import CoreGraphics
import Foundation
import Observation
import StitchKit
import Synchronization

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
    case panorama
}

/// Hands engine events to the main actor: the newest event of each stage, with at most one hop in flight, so that
/// a burst of events costs one update of the views.
final class ProgressRelay: Sendable {
    private let state = Mutex<(pending: [ProgressEvent], scheduled: Bool)>(([], false))
    private let deliver: @MainActor @Sendable ([ProgressEvent]) -> Void

    init(_ deliver: @escaping @MainActor @Sendable ([ProgressEvent]) -> Void) {
        self.deliver = deliver
    }

    func send(_ event: ProgressEvent) {
        let schedule = state.withLock { state -> Bool in
            if let index = state.pending.firstIndex(where: { $0.stage == event.stage }) {
                state.pending[index] = event
            } else {
                state.pending.append(event)
            }
            guard !state.scheduled else { return false }
            state.scheduled = true
            return true
        }
        guard schedule else { return }
        Task { @MainActor in
            let events = self.state.withLock { state -> [ProgressEvent] in
                defer {
                    state.pending = []
                    state.scheduled = false
                }
                return state.pending
            }
            self.deliver(events)
        }
    }
}

/// Why the panorama on screen no longer matches the session.
enum PanoramaStaleness {
    /// Photos, exclusions or the scene mode changed: a new analysis is needed.
    case photos
    /// The photos were analysed again since.
    case analysis
    /// The stitch settings changed.
    case settings
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

    var stitchRequest = StitchRequest()
    private(set) var panorama: Panorama?
    /// Show and export the panorama cropped to its largest rectangle without empty pixels.
    var cropsPanorama = true
    private(set) var isStitching = false
    private(set) var stitchProgress: ProgressEvent?
    /// Stop was pressed and the work has not ended yet.
    private(set) var isStopping = false
    /// Share of the current stage done, when it is known; it never goes back within a stage.
    private(set) var displayFraction: Double?
    private var phase: String?
    /// The latest event of each stage of a phase whose stages run side by side (the extractors, the matchers),
    /// and how many extractors and matchers run.
    private var combined: [String: ProgressEvent] = [:]
    private var extractors = 1
    private(set) var isExporting = false
    private(set) var lastExport: URL?

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
    private var stitching: Task<Void, Never>?
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

    var isBusy: Bool { isRunning || isStitching }

    var canAnalyze: Bool { images.count >= 2 && !isBusy && !activeSources.isEmpty }

    /// Stitching analyses first when the report is missing, stale, or was made in another scene mode.
    var canStitch: Bool { images.count - excluded.count >= 2 && !isBusy && !activeSources.isEmpty }

    /// A report that can be stitched as it is.
    var reportReadyForStitch: Bool { reportIsCurrent && report?.configuration.mode == configuration.mode }

    /// Photos in the report's main group, 0 when it has fewer than two.
    var stitchableCount: Int {
        report?.graph.components.first.map { $0.count >= 2 ? $0.count : 0 } ?? 0
    }

    var panoramaStaleness: PanoramaStaleness? {
        guard let panorama else { return nil }
        if !reportReadyForStitch { return .photos }
        if panorama.reportCreatedAt != report?.createdAt { return .analysis }
        return panorama.request == stitchRequest ? nil : .settings
    }

    /// The report is stale once photos or exclusions change.
    var reportIsCurrent: Bool {
        guard let report else { return false }
        return report.images.map(\.url) == images.map(\.url) && Set(report.excludedByUser) == excluded
    }

    /// Adds photos, and the frames chosen from each video among `urls`. `then` runs once the photos and the
    /// frames are in (not when the frames were stopped or could not be read).
    func add(_ urls: [URL], then: (@MainActor () -> Void)? = nil) {
        let videos = urls.filter(VideoFrames.isVideo)
        addPhotos(urls.filter { !VideoFrames.isVideo($0) })
        guard !videos.isEmpty else {
            then?()
            return
        }
        guard !isBusy else {
            errorMessage = String(localized: "A video can be added when the work in progress has ended.")
            return
        }
        extractFrames(videos, then: then)
    }

    /// Chooses the sharpest frames of each video, a third of a frame apart, and adds them as photos.
    private func extractFrames(_ videos: [URL], then: (@MainActor () -> Void)?) {
        let generation = generation
        isRunning = true
        analysis = Task {
            defer {
                if generation == self.generation {
                    isRunning = false
                    isStopping = false
                    progress = nil
                    displayFraction = nil
                    phase = nil
                    analysis = nil
                }
            }
            let relay = ProgressRelay { [weak self] events in self?.apply(events, .analysis, generation) }
            do {
                for video in videos {
                    // Detached: the extraction reads and measures every frame on the calling thread.
                    let frames = try await Task.detached(priority: .userInitiated) {
                        try await VideoFrames.extract(video: video) { relay.send($0) }
                    }.value
                    try Task.checkCancellation()
                    guard generation == self.generation else { return }
                    addPhotos(frames)
                }
            } catch is CancellationError {
                return
            } catch {
                if generation == self.generation { errorMessage = error.localizedDescription }
                return
            }
            guard generation == self.generation, let then else { return }
            // After this task has ended and the session is no longer busy.
            Task { then() }
        }
    }

    private func addPhotos(_ urls: [URL]) {
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
        stitching?.cancel()
        stitching = nil
        isRunning = false
        isStitching = false
        isStopping = false
        progress = nil
        stitchProgress = nil
        displayFraction = nil
        phase = nil
        panorama = nil
        lastExport = nil
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
        if isBusy { isStopping = true }
        analysis?.cancel()
        stitching?.cancel()
    }

    private enum Stream { case analysis, stitch }

    /// Shows the events of the current run, unless it was stopped or replaced.
    private func apply(_ events: [ProgressEvent], _ stream: Stream, _ generation: Int) {
        guard generation == self.generation, !isStopping, stream == .analysis ? isRunning : isStitching else { return }
        for var event in events {
            // The two extractors run side by side with LightGlue's matching of the pairs compared in any case
            // (consecutive shots, or every pair), and the two matchers side by side: one phase each.
            let extracting = ["sift", "lightglue-extract", "lightglue-early"].contains(event.stage)
            let matching = ["sift-match", "lightglue-match"].contains(event.stage)
            let phase = extracting ? "features" : matching ? "match" : event.stage
            if phase != self.phase {
                self.phase = phase
                displayFraction = nil
                combined = [:]
            }
            var fraction = event.fraction ?? (event.total > 0 ? Double(event.completed) / Double(event.total) : nil)
            if extracting || matching {
                // Counted against every extractor or matcher from the start, so that a fast one does not fill
                // the bar alone.
                combined[event.stage] = event
                let started = combined.keys.filter { $0 != "lightglue-early" }.count
                let each = combined.values.first { $0.stage != "lightglue-early" }?.total ?? 0
                let total = combined.values.reduce(0) { $0 + $1.total } + max(0, extractors - started) * each
                let completed = combined.values.reduce(0) { $0 + $1.completed }
                fraction = total > 0 ? Double(completed) / Double(total) : nil
                // One count for both matchers, which report in turn.
                if matching { event = ProgressEvent(stage: "match", completed: completed, total: total) }
            }
            if let fraction { displayFraction = max(displayFraction ?? 0, min(fraction, 1)) }
            switch stream {
            case .analysis: if progress != event { progress = event }
            case .stitch: if stitchProgress != event { stitchProgress = event }
            }
        }
    }

    /// Joins the main group into one image, analysing first when needed. A failed or stopped stitch keeps
    /// the previous result.
    func stitch() {
        guard canStitch else { return }
        tab = .panorama
        let generation = generation
        isStitching = true
        stitching = Task {
            defer {
                if generation == self.generation {
                    isStitching = false
                    isStopping = false
                    stitchProgress = nil
                    displayFraction = nil
                    phase = nil
                    stitching = nil
                }
            }
            if !reportReadyForStitch {
                await run()
                guard generation == self.generation, reportReadyForStitch, !Task.isCancelled else { return }
            }
            await composite(generation)
        }
    }

    private func composite(_ generation: Int) async {
        guard let report else { return }
        guard stitchableCount >= 2 else {
            errorMessage = String(localized: "No two photos were joined: the Graph view shows why.")
            return
        }
        do {
            let relay = ProgressRelay { [weak self] events in self?.apply(events, .stitch, generation) }
            let result = try await engine.stitch(report, request: stitchRequest) { relay.send($0) }
            guard generation == self.generation else { return }
            panorama = result
            lastExport = nil
            cropsPanorama = result.cropsByDefault
        } catch is CancellationError {
            // Stopped by the user or by a new session.
        } catch {
            if generation == self.generation { errorMessage = error.localizedDescription }
        }
    }

    /// Asks where to save the panorama and writes it there.
    func exportPanorama() {
        guard let panorama, !isExporting else { return }
        Task { await export(panorama) }
    }

    private func export(_ panorama: Panorama) async {
        let photos = panorama.imageIDs.compactMap { id in images.first { $0.id == id } }
        let base = Panorama.suggestedName(first: photos.first?.name ?? "Tessera", last: photos.last?.name ?? "")
        let crop = cropsPanorama && !panorama.crop.isEmpty ? panorama.crop : nil
        let size = crop.map { PixelSize(width: $0.width, height: $0.height) } ?? panorama.size
        guard let (url, options) = await ExportPanel.run(
            name: String(localized: "\(base) panorama"), directory: photos.first?.url.deletingLastPathComponent(),
            transparent: crop == nil && !panorama.pixels.opaque, size: size
        ) else { return }
        isExporting = true
        defer { isExporting = false }
        let pixels = panorama.pixels
        do {
            try await Task.detached(priority: .userInitiated) {
                try PanoramaWriter.write(pixels, crop: crop, to: url, options: options)
            }.value
            lastExport = url
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func run() async {
        let generation = generation
        isRunning = true
        defer {
            if generation == self.generation {
                isRunning = false
                progress = nil
                analysis = nil
                // A stitch runs the analysis first: its Stop lasts until the stitch ends.
                if !isStitching {
                    isStopping = false
                    displayFraction = nil
                    phase = nil
                }
            }
        }
        var configuration = configuration
        configuration.sources = activeSources
        extractors = max(1, configuration.sources.count)
        let start = ContinuousClock.now
        do {
            let relay = ProgressRelay { [weak self] events in self?.apply(events, .analysis, generation) }
            let result = try await engine.analyze(
                urls: images.map(\.url), configuration: configuration, excluded: excluded
            ) { relay.send($0) }
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
        // A choice the new set cannot run (an ONNX matcher, fp32) goes back to the defaults.
        if let models = configuration.learnedModels, !models.supports(configuration) {
            let defaults = PipelineConfiguration()
            configuration.extractorBackend = defaults.extractorBackend
            configuration.matcherBackend = defaults.matcherBackend
            configuration.matcherPrecision = defaults.matcherPrecision
        }
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
