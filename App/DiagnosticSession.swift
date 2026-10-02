import CoreGraphics
import Foundation
import Observation
import StitchKit

struct PairID: Hashable {
    var a: Int
    var b: Int

    init(_ x: Int, _ y: Int) {
        a = min(x, y)
        b = max(x, y)
    }
}

enum DetailTab: Hashable {
    case graph
    case pair
}

/// Download of the learned models: nothing going on, in progress (no report yet when nil), or failed.
enum ModelDownload: Equatable {
    case idle
    case running(ModelInstallProgress?)
    case failed(String)
}

/// State of one diagnostic session: imported photos, settings, last report and selections.
@MainActor
@Observable
final class DiagnosticSession {
    private(set) var images: [SourceImage] = []
    var excluded: Set<Int> = []
    var configuration: PipelineConfiguration
    private(set) var report: MatchReport?
    private(set) var isRunning = false
    private(set) var progress: ProgressEvent?
    private(set) var lastDuration: Duration?
    var errorMessage: String?
    private(set) var modelDownload: ModelDownload = .idle

    var tab: DetailTab = .graph
    var selectedImage: Int?
    var selectedPair: PairID?
    var filter: EvidenceFilter = .best

    private(set) var thumbnails: [Int: CGImage] = [:]
    private(set) var previews: [Int: CGImage] = [:]
    private let engine = StitchEngine()
    /// Bumped by `reset()`: results of work started before it (analysis, thumbnails) are dropped.
    private var generation = 0
    private var analysis: Task<Void, Never>?
    private var modelTask: Task<Void, Never>?

    init() {
        var configuration = PipelineConfiguration()
        configuration.sources = configuration.learnedModels == nil ? [.rootSIFT] : [.rootSIFT, .racoLightGlue]
        self.configuration = configuration
    }

    var lightGlueAvailable: Bool { configuration.learnedModels != nil }

    /// The models in use are the ones the app downloaded, so the app may remove them.
    var modelsAreRemovable: Bool {
        configuration.learnedModels?.directory.resolvingSymlinksInPath().path
            == ModelInstaller.standardDestination.resolvingSymlinksInPath().path
    }

    /// Matchers that will actually run.
    var activeSources: [FeatureSource] {
        configuration.sources.filter { $0 != .racoLightGlue || lightGlueAvailable }
    }

    var canAnalyze: Bool { images.count >= 2 && !isRunning && !activeSources.isEmpty }

    /// The report is stale once photos or exclusions change.
    var reportIsCurrent: Bool {
        guard let report else { return false }
        return report.images.map(\.url) == images.map(\.url) && Set(report.excludedByUser) == excluded
    }

    func add(_ urls: [URL]) {
        var known = Set(images.map(\.url))
        var failed: [String] = []
        for url in urls where !known.contains(url) {
            known.insert(url)
            do {
                let image = try Thumbnails.describe(url, id: images.count)
                images.append(image)
                loadThumbnail(for: image)
            } catch {
                failed.append(url.lastPathComponent)
            }
        }
        if !failed.isEmpty {
            errorMessage = String(localized: "These files could not be read and were not added: \(failed.joined(separator: ", "))")
        }
    }

    func reset() {
        generation += 1
        analysis?.cancel()
        analysis = nil
        isRunning = false
        progress = nil
        images = []
        excluded = []
        report = nil
        thumbnails = [:]
        previews = [:]
        selectedImage = nil
        selectedPair = nil
        tab = .graph
    }

    func toggleExclusion(_ id: Int) {
        if excluded.contains(id) {
            excluded.remove(id)
        } else {
            excluded.insert(id)
        }
    }

    func setSource(_ source: FeatureSource, enabled: Bool) {
        var sources = configuration.sources.filter { $0 != source }
        if enabled { sources.append(source) }
        configuration.sources = FeatureSource.allCases.filter { sources.contains($0) }
    }

    /// Starts an analysis of the current photos; `cancel()` or `reset()` stops it.
    func analyze() {
        guard canAnalyze else {
            if activeSources.isEmpty { errorMessage = String(localized: "Turn on at least one matcher.") }
            return
        }
        analysis = Task { await run() }
    }

    func cancel() {
        analysis?.cancel()
    }

    private func run() async {
        let generation = generation
        isRunning = true
        defer {
            if generation == self.generation {
                isRunning = false
                progress = nil
                analysis = nil
            }
        }
        var configuration = configuration
        configuration.sources = activeSources
        let start = ContinuousClock.now
        do {
            let result = try await engine.analyze(
                urls: images.map(\.url), configuration: configuration, excluded: excluded
            ) { [weak self] event in
                Task { @MainActor in
                    if self?.generation == generation { self?.progress = event }
                }
            }
            guard generation == self.generation else { return }
            report = result
            lastDuration = ContinuousClock.now - start
            if let selectedPair, result.evidence(selectedPair.a, selectedPair.b, source: .rootSIFT) == nil,
               result.evidence(selectedPair.a, selectedPair.b, source: .racoLightGlue) == nil {
                self.selectedPair = nil
            }
            if selectedPair == nil {
                selectedPair = result.evidence(filter: filter).first { $0.verdict == .verified }
                    .map { PairID($0.a, $0.b) }
            }
        } catch is CancellationError {
            // Stopped by the user or by a new session.
        } catch {
            if generation == self.generation { errorMessage = error.localizedDescription }
        }
    }

    func openPair(_ pair: PairID) {
        selectedPair = pair
        tab = .pair
        loadPreview(for: pair.a)
        loadPreview(for: pair.b)
    }

    /// Evidence for the selected pair under the current filter.
    func evidence(for pair: PairID) -> PairEvidence? {
        guard let report else { return nil }
        switch filter {
        case .best:
            return report.evidence(filter: .best).first { PairID($0.a, $0.b) == pair }
        case .only(let source):
            return report.evidence(pair.a, pair.b, source: source)
        }
    }

    func name(of id: Int) -> String {
        images.first { $0.id == id }?.name ?? "#\(id)"
    }

    // MARK: - Models

    /// Downloads and installs the learned models, then turns RaCo + LightGlue on.
    func downloadModels() {
        guard modelTask == nil else { return }
        modelDownload = .running(nil)
        modelTask = Task { await installModels() }
    }

    /// Cancels a download in progress. The returned task ends once its partial files are gone.
    @discardableResult
    func cancelModelDownload() -> Task<Void, Never>? {
        modelTask?.cancel()
        return modelTask
    }

    private func installModels() async {
        defer { modelTask = nil }
        do {
            try await ModelInstaller().install { [weak self] progress in
                Task { @MainActor in
                    guard let self, case .running = self.modelDownload else { return }
                    self.modelDownload = .running(progress)
                }
            }
            modelDownload = .idle
            reloadModels()
            if lightGlueAvailable { setSource(.racoLightGlue, enabled: true) }
        } catch is CancellationError {
            modelDownload = .idle
        } catch {
            modelDownload = .failed(error.localizedDescription)
        }
    }

    func removeModels() {
        // Not while an analysis uses the models or a download is about to replace them.
        guard !isRunning, modelTask == nil else { return }
        do {
            try ModelInstaller().remove()
        } catch {
            errorMessage = error.localizedDescription
        }
        reloadModels()
        // Without models RootSIFT is the only matcher left.
        if !lightGlueAvailable { setSource(.rootSIFT, enabled: true) }
    }

    private func reloadModels() {
        configuration.learnedModels = LearnedModelSet.standard(keypoints: configuration.learnedModels?.keypoints ?? 2048)
    }

    // MARK: - Images

    /// Whether `id` still names the photo at `url` in the current session.
    private func stillShows(_ id: Int, _ url: URL, _ generation: Int) -> Bool {
        generation == self.generation && images.first { $0.id == id }?.url == url
    }

    private func loadThumbnail(for image: SourceImage) {
        let url = image.url, id = image.id, generation = generation
        Task.detached(priority: .utility) {
            let thumbnail = try? Thumbnails.load(url, longSide: 360)
            await MainActor.run { [weak self] in
                guard let self, let thumbnail, self.stillShows(id, url, generation) else { return }
                self.thumbnails[id] = thumbnail
            }
        }
    }

    func loadPreview(for id: Int) {
        guard previews[id] == nil, let image = images.first(where: { $0.id == id }) else { return }
        let url = image.url, generation = generation
        Task.detached(priority: .userInitiated) {
            let preview = try? Thumbnails.load(url, longSide: 1800)
            await MainActor.run { [weak self] in
                guard let self, let preview, self.stillShows(id, url, generation) else { return }
                self.previews[id] = preview
            }
        }
    }
}
