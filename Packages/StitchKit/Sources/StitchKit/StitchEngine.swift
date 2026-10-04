import CStitchCore
import Foundation
import simd

public struct ProgressEvent: Sendable, Equatable {
    public var stage: String
    public var completed: Int
    public var total: Int
    /// Share of the stage done, for stages that measure work instead of counting items (align, layout).
    public var fraction: Double? = nil
}

/// Owns a native SIFT feature set.
private final class SIFTFeatures: @unchecked Sendable {
    let pointer: OpaquePointer
    /// Working pixels per original pixel, per axis.
    let scale: SIMD2<Double>

    init(pointer: OpaquePointer, scale: SIMD2<Double>) {
        self.pointer = pointer
        self.scale = scale
    }

    deinit { sc_features_free(pointer) }
}

/// Runs the diagnostic pipeline: features, tentative matches, verification, graph.
public actor StitchEngine {
    private var learned: LearnedSession?
    /// A session being built, so concurrent analyses wait for it instead of building their own.
    private var loading: (key: String, task: Task<LearnedSession, Error>)?

    public init() {}

    /// Photos that cannot be opened or decoded are reported as unreadable instead of stopping the run.
    /// Cancelling the calling task stops the analysis at the next stage boundary.
    public func analyze(
        urls: [URL],
        configuration: PipelineConfiguration,
        excluded: Set<Int> = [],
        progress: (@Sendable (ProgressEvent) -> Void)? = nil
    ) async throws -> MatchReport {
        guard !configuration.sources.isEmpty else { throw StitchError.noMatcher }
        var unreadable = Set<Int>()
        let images = urls.enumerated().map { index, url in
            do {
                return try ImageLoader.describe(url: url, id: index)
            } catch {
                unreadable.insert(index)
                return SourceImage(id: index, url: url, name: url.lastPathComponent, pixelSize: PixelSize(width: 0, height: 0),
                                   captureDate: nil, focalLength35mm: nil)
            }
        }
        let readable = images.filter { !excluded.contains($0.id) && !unreadable.contains($0.id) }
        try Task.checkCancellation()

        // 1. Features of every photo, computed once. RootSIFT runs on the CPU cores while the
        //    learned extractor runs on the GPU, so the two proceed side by side.
        //    Pairs that are candidates whatever the affinity (consecutive shots, or every pair) are
        //    matched by LightGlue as soon as both photos are extracted, while extraction goes on.
        let (session, learnedProblem) = try await learnedSession(configuration)
        let early = session == nil ? [] : Self.earlyPairs(readable, configuration)
        let (ready, readyPairs) = AsyncStream<(Extraction, Extraction)>.makeStream()
        defer { readyPairs.finish() }
        // Counted with the extraction ("lightglue-early"): with every pair compared it goes on after it.
        if !early.isEmpty { progress?(ProgressEvent(stage: "lightglue-early", completed: 0, total: early.count)) }
        async let earlyResult = Self.matchLearned(ready, session, configuration, total: early.count, progress)
        async let siftResult = configuration.sources.contains(.rootSIFT)
            ? Self.extractSIFT(readable, configuration, progress) : ([:], [])
        async let learnedResult = Self.extractLearned(readable, session, progress) { done, latest in
            for key in early where key.a == latest.image.id || key.b == latest.image.id {
                if let a = done[key.a], let b = done[key.b] { readyPairs.yield((a, b)) }
            }
        }
        let (sift, siftFailed) = try await siftResult
        let (learned, learnedFailed) = try await learnedResult
        readyPairs.finish()
        let earlyEvidence = try await earlyResult
        // Photos the header described but whose pixels could not be decoded.
        let failed = siftFailed.union(learnedFailed)
        unreadable.formUnion(failed)
        let active = readable.filter { !failed.contains($0.id) }
        let matchedEarly = Set(earlyEvidence.map { PairProposal.Key($0.a, $0.b) })
        try Task.checkCancellation()

        // 2. Which pairs to match: all of them, or candidates ranked by a quick descriptor affinity.
        progress?(ProgressEvent(stage: "affinity", completed: 0, total: 0))
        var (candidates, affinity) = await Self.candidates(active, sift: sift, learned: learned,
                                                           configuration: configuration, progress: progress)
        try Task.checkCancellation()
        let lookup = Dictionary(uniqueKeysWithValues: active.map { ($0.id, $0) })

        // 3-4. Matching and geometric verification, only on the candidates.
        func examine(_ chosen: [CandidatePair], learnedAlready: [PairEvidence] = []) async throws -> [PairEvidence] {
            let pairs = chosen.compactMap { pair in lookup[pair.a].flatMap { a in lookup[pair.b].map { (a, $0) } } }
            var evidence: [PairEvidence] = []
            if !sift.isEmpty {
                evidence += try await matchSIFT(pairs, sift, configuration, progress)
            }
            if let session {
                let remaining = pairs.filter { !matchedEarly.contains(PairProposal.Key($0.0.id, $0.1.id)) }
                evidence += learnedAlready
                evidence += try await matchLearned(remaining, learned, session, configuration, progress)
            }
            try Task.checkCancellation()
            progress?(ProgressEvent(stage: "verify", completed: 0, total: evidence.count))
            let verified = await Self.verifyAll(evidence, images: images, configuration: configuration, progress: progress)
            try Task.checkCancellation()
            return verified
        }
        let chosen = Set(candidates.map { PairProposal.Key($0.a, $0.b) })
        var verified = try await examine(
            candidates, learnedAlready: earlyEvidence.filter {
                chosen.contains(PairProposal.Key($0.a, $0.b)) && !failed.contains($0.a) && !failed.contains($0.b)
            }
        )

        // 5. If the verified pairs leave separate groups, try the best untried pairs between them.
        var tried = Set(candidates.map { PairProposal.Key($0.a, $0.b) })
        if configuration.pairSelection == .proposed {
            for _ in 0..<3 {
                let joined = verified.filter { $0.verdict == .verified }.map { PairProposal.Key($0.a, $0.b) }
                let groups = PairProposal.components(active.map(\.id), edges: joined)
                guard groups.count > 1 else { break }
                let bridges = PairProposal.bridges(groups: groups, affinity: affinity, tried: tried,
                                                   perGroupPair: configuration.bridgePairs, limit: active.count)
                guard !bridges.isEmpty else { break }
                progress?(ProgressEvent(stage: "bridge", completed: 0, total: bridges.count))
                candidates += bridges
                tried.formUnion(bridges.map { PairProposal.Key($0.a, $0.b) })
                verified += try await examine(bridges)
            }
        }

        let features = active.compactMap { sift[$0.id]?.1 } + active.compactMap { learned[$0.id]?.features }

        // 6. Pairs the layout predicts. The candidates join a large set almost as a tree, where a false pair
        //    has nothing to disagree with: 728 pairs for 511 drone photos that each overlap six or eight
        //    others. A provisional alignment of the main group places the photos, and pairs that overlap
        //    there but were never compared are matched too; the loops they close let the global alignment
        //    find the false pairs.
        if configuration.pairSelection == .proposed, active.count > Self.overlapRounds.minimumPhotos {
            for _ in 0..<Self.overlapRounds.count {
                try Task.checkCancellation()
                let graph = MatchGraphBuilder.build(images: images, features: features, pairs: verified, excluded: excluded,
                                                    unreadable: unreadable, configuration: configuration)
                let provisional = MatchReport(
                    createdAt: Date(), configuration: configuration, images: images, features: [], pairs: verified,
                    graph: graph, excludedByUser: excluded.sorted(), candidates: nil, learnedPipeline: nil, learnedProblem: nil)
                let meter = WorkMeter(stage: "layout", emit: progress)
                let predicted = try Self.overlapping(provisional, tried: tried, perPhoto: Self.overlapRounds.perPhoto,
                                                     progress: meter.begin())
                meter.finish()
                guard !predicted.isEmpty else { break }
                progress?(ProgressEvent(stage: "overlap", completed: 0, total: predicted.count))
                candidates += predicted
                tried.formUnion(predicted.map { PairProposal.Key($0.a, $0.b) })
                verified += try await examine(predicted)
            }
        }

        // Pairs arrive in completion order; a fixed order makes identical runs give identical reports.
        verified.sort { ($0.a, $0.b, $0.source.rawValue) < ($1.a, $1.b, $1.source.rawValue) }
        let graph = MatchGraphBuilder.build(
            images: images, features: features, pairs: verified, excluded: excluded, unreadable: unreadable,
            configuration: configuration
        )
        return MatchReport(
            createdAt: Date(), configuration: configuration, images: images, features: features,
            pairs: verified, graph: graph, excludedByUser: excluded.sorted(), candidates: candidates,
            learnedPipeline: session?.pipeline, learnedProblem: learnedProblem
        )
    }

    /// Layout-predicted pairs: above `minimumPhotos` photos, `count` rounds of at most `perPhoto` new pairs.
    static let overlapRounds = (minimumPhotos: 8, count: 2, perPhoto: 6)

    private enum LayoutStep: Hashable, Sendable { case layout, document }

    /// Untried pairs of the main group that a provisional alignment of `report` places on top of each other,
    /// most overlapping first: for each photo its `perPhoto` best, at most four per photo in all.
    /// Throws only when cancelled. The provisional alignments advance `progress`.
    nonisolated static func overlapping(_ report: MatchReport, tried: Set<PairProposal.Key>, perPhoto: Int,
                                        progress: WorkSpan = .none) throws -> [CandidatePair] {
        let component = report.graph.components.first ?? []
        let estimate = AlignmentCost.size(report, component: component)
        let layout = AlignmentCost.build * Double(estimate.pairs)
            + AlignmentCost.align(report.configuration.mode, estimate, provisional: true)
        let documentCost = AlignmentCost.build * Double(estimate.pairs)
            + AlignmentCost.align(.document, estimate, provisional: true)
        var plan = WorkPlan<LayoutStep>(progress, [(.layout, layout), (.document, 0.5 * documentCost)])
        var outcome: AlignmentOutcome
        do {
            outcome = try alignment(for: report, straighten: false, provisional: true, progress: plan.next(.layout))
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return []
        }
        // The layout only has to be right locally: when homographies fit the pairs better than the model
        // the mode picks (whose stretch rule guards the whole panorama), they predict the overlaps.
        try Task.checkCancellation()
        if outcome.alignment.model != .rotation, outcome.alignment.model != .homography {
            plan.revise(.document, documentCost)
            var flat = report
            flat.configuration.mode = .document
            func median(_ values: [Double]) -> Double { values.sorted().dropFirst(values.count / 2).first ?? .infinity }
            let document = try? alignment(for: flat, straighten: false, provisional: true, progress: plan.next(.document))
            try Task.checkCancellation()
            if let document, median(document.alignment.pairRMS) < median(outcome.alignment.pairRMS) {
                outcome = document
            }
        }
        let images = outcome.problem.images, alignment = outcome.alignment
        var scores: [(key: PairProposal.Key, overlap: Double)] = []
        if alignment.model == .rotation {
            // Optical axes (world ray of the photo centre) and half fields of view along the short side.
            let axes = alignment.transforms.map { r in simd_normalize(SIMD3(r[2], r[5], r[8])) }
            let halves = images.indices.map { i in
                atan(Double(min(images[i].pixelSize.width, images[i].pixelSize.height)) / (2 * alignment.focals[i]))
            }
            for i in images.indices {
                for j in images.indices where j > i {
                    let angle = acos(max(-1, min(1, simd_dot(axes[i], axes[j]))))
                    let reach = halves[i] + halves[j]
                    if angle < reach { scores.append((PairProposal.Key(images[i].id, images[j].id), 1 - angle / reach)) }
                }
            }
        } else {
            // Each photo's outline on the mosaic, a convex quadrilateral while it stays in front of the plane.
            let outlines: [[SIMD2<Double>]?] = images.indices.map { i in
                let m = PlaneGeometry.matrix(alignment.transforms[i])
                let w = Double(images[i].pixelSize.width), h = Double(images[i].pixelSize.height)
                let corners = [(0.0, 0.0), (w, 0), (w, h), (0, h)].compactMap { PlaneGeometry.apply(m, $0.0, $0.1) }
                return corners.count == 4 ? corners : nil
            }
            let areas = outlines.map { $0.map { abs(PlaneGeometry.signedArea($0)) } ?? 0 }
            let boxes = outlines.map { outline -> (SIMD2<Double>, SIMD2<Double>)? in
                outline.map { points in (points.reduce(points[0]) { simd_min($0, $1) }, points.reduce(points[0]) { simd_max($0, $1) }) }
            }
            for i in images.indices {
                guard let a = outlines[i], let boxA = boxes[i], areas[i] > 0 else { continue }
                for j in images.indices where j > i {
                    guard let b = outlines[j], let boxB = boxes[j], areas[j] > 0,
                          boxA.0.x < boxB.1.x, boxB.0.x < boxA.1.x, boxA.0.y < boxB.1.y, boxB.0.y < boxA.1.y
                    else { continue }
                    let shared = abs(PlaneGeometry.signedArea(PlaneGeometry.clip(a, by: b)))
                    let overlap = shared / min(areas[i], areas[j])
                    if overlap > 0.05 { scores.append((PairProposal.Key(images[i].id, images[j].id), overlap)) }
                }
            }
        }
        let untried = scores.filter { !tried.contains($0.key) }.sorted { ($0.overlap, $1.key) > ($1.overlap, $0.key) }
        var chosen: [PairProposal.Key] = [], taken = Set<PairProposal.Key>(), count: [Int: Int] = [:]
        for score in untried where count[score.key.a, default: 0] < perPhoto || count[score.key.b, default: 0] < perPhoto {
            guard chosen.count < 4 * images.count, taken.insert(score.key).inserted else { continue }
            chosen.append(score.key)
            count[score.key.a, default: 0] += 1
            count[score.key.b, default: 0] += 1
        }
        return chosen.map { CandidatePair(a: $0.a, b: $0.b, affinity: nil, reasons: [.overlap]) }
    }

    /// The learned session when RaCo + LightGlue is among the sources. When it cannot be loaded (no
    /// models for the chosen backends, a backend this build lacks) and RootSIFT runs too, the analysis goes
    /// on with RootSIFT alone and the reason is returned instead.
    private func learnedSession(_ configuration: PipelineConfiguration) async throws -> (LearnedSession?, String?) {
        guard configuration.sources.contains(.racoLightGlue) else { return (nil, nil) }
        do {
            return (try await loadLearned(configuration), nil)
        } catch where !(error is CancellationError) && configuration.sources.contains(.rootSIFT) {
            return (nil, error.localizedDescription)
        }
    }

    // MARK: - RootSIFT

    /// Features per photo, and the ids of photos that could not be decoded.
    private nonisolated static func extractSIFT(
        _ images: [SourceImage], _ configuration: PipelineConfiguration,
        _ progress: (@Sendable (ProgressEvent) -> Void)?
    ) async throws -> ([Int: (SIFTFeatures, ImageFeatures)], Set<Int>) {
        try await withThrowingTaskGroup(of: (Int, (SIFTFeatures, ImageFeatures)?).self) { group in
            for image in images {
                group.addTask {
                    try Task.checkCancellation()
                    do {
                        let (id, handle, features) = try Self.extractSIFT(image, configuration)
                        return (id, (handle, features))
                    } catch let error as StitchError where error.isUnreadableImage {
                        return (image.id, nil)
                    }
                }
            }
            var results: [Int: (SIFTFeatures, ImageFeatures)] = [:]
            var failed = Set<Int>()
            for try await (id, result) in group {
                if let result { results[id] = result } else { failed.insert(id) }
                progress?(ProgressEvent(stage: "sift", completed: results.count + failed.count, total: images.count))
            }
            return (results, failed)
        }
    }

    private func matchSIFT(
        _ pairs: [(SourceImage, SourceImage)], _ extracted: [Int: (SIFTFeatures, ImageFeatures)],
        _ configuration: PipelineConfiguration, _ progress: (@Sendable (ProgressEvent) -> Void)?
    ) async throws -> [PairEvidence] {
        let evidence = try await withThrowingTaskGroup(of: PairEvidence.self) { group in
            for (a, b) in pairs {
                guard let (handleA, _) = extracted[a.id], let (handleB, _) = extracted[b.id] else { continue }
                group.addTask {
                    try Task.checkCancellation()
                    return try Self.matchSIFT(a.id, handleA, b.id, handleB, configuration)
                }
            }
            var results: [PairEvidence] = []
            for try await item in group {
                results.append(item)
                progress?(ProgressEvent(stage: "sift-match", completed: results.count, total: pairs.count))
            }
            return results
        }
        return evidence.sorted { ($0.a, $0.b) < ($1.a, $1.b) }
    }

    private nonisolated static func extractSIFT(
        _ image: SourceImage, _ configuration: PipelineConfiguration
    ) throws -> (Int, SIFTFeatures, ImageFeatures) {
        let start = ContinuousClock.now
        let working = try ImageLoader.gray(for: image, megapixels: configuration.siftMegapixels)
        var message = [CChar](repeating: 0, count: 512)
        let pointer = working.gray.withUnsafeBufferPointer { gray in
            sc_sift_extract(
                gray.baseAddress, Int32(working.size.width), Int32(working.size.height),
                Int32(working.size.width), Int32(configuration.siftMaxFeatures), 1, &message, message.count
            )
        }
        guard let pointer else { throw StitchError.engine(errorText(message)) }
        let handle = SIFTFeatures(pointer: pointer, scale: working.scale)
        let count = Int(sc_features_count(pointer))
        let inverse = SIMD2<Float>(1 / working.scale)
        let inverseSize = (inverse.x + inverse.y) / 2
        var keypoints: [Keypoint] = []
        if let raw = sc_features_keypoints(pointer) {
            keypoints = (0..<count).map { index in
                let k = raw[index]
                return Keypoint(x: siftToOriginal(k.x, inverse.x), y: siftToOriginal(k.y, inverse.y),
                                size: k.size * inverseSize, angle: k.angle,
                                response: k.response)
            }
        }
        let seconds = (ContinuousClock.now - start).seconds
        let features = ImageFeatures(
            imageID: image.id, source: .rootSIFT, workingSize: working.size, keypoints: keypoints,
            extractionSeconds: seconds
        )
        return (image.id, handle, features)
    }

    private nonisolated static func matchSIFT(
        _ a: Int, _ handleA: SIFTFeatures, _ b: Int, _ handleB: SIFTFeatures, _ configuration: PipelineConfiguration
    ) throws -> PairEvidence {
        let start = ContinuousClock.now
        var raw: UnsafeMutablePointer<sc_match>?
        var message = [CChar](repeating: 0, count: 512)
        let count = sc_match_descriptors(
            handleA.pointer, handleB.pointer, configuration.ratio, configuration.mutualCheck ? 1 : 0, &raw,
            &message, message.count
        )
        defer { sc_free(raw) }
        guard count >= 0, let raw else { throw StitchError.engine(errorText(message)) }

        let keypointsA = sc_features_keypoints(handleA.pointer)
        let keypointsB = sc_features_keypoints(handleB.pointer)
        let inverseA = SIMD2<Float>(1 / handleA.scale)
        let inverseB = SIMD2<Float>(1 / handleB.scale)
        var matches: [TentativeMatch] = []
        matches.reserveCapacity(Int(count))
        for index in 0..<Int(count) {
            let m = raw[index]
            guard let ka = keypointsA?[Int(m.index_a)], let kb = keypointsB?[Int(m.index_b)] else { continue }
            matches.append(TentativeMatch(
                a: Point2(x: siftToOriginal(ka.x, inverseA.x), y: siftToOriginal(ka.y, inverseA.y)),
                b: Point2(x: siftToOriginal(kb.x, inverseB.x), y: siftToOriginal(kb.y, inverseB.y)),
                score: m.score
            ))
        }
        return PairEvidence(
            a: a, b: b, source: .rootSIFT, matches: matches, fits: [], chosenModel: nil, inlierCoverage: 0,
            confidence: 0, verdict: .tooFewMatches, matchingSeconds: (ContinuousClock.now - start).seconds,
            verificationSeconds: 0
        )
    }

    // MARK: - RaCo-ALIKED + LightGlue

    private func loadLearned(_ configuration: PipelineConfiguration) async throws -> LearnedSession {
        guard let models = configuration.learnedModels, models.isUsable else { throw StitchError.modelMissing }
        let key = LearnedSession.key(models, configuration)
        if let learned, learned.key == key { return learned }
        // The actor is free while the session loads; a second analysis waits for the same task.
        if let loading, loading.key == key { return try await loading.task.value }
        learned = nil
        let task = Task { try await LearnedSession(models: models, configuration: configuration) }
        loading = (key, task)
        defer { if loading?.key == key { loading = nil } }
        let session = try await task.value
        if loading?.key == key { learned = session }
        return session
    }

    /// Network output for one photo, in the pixels of its fixed-size canvas.
    private struct Extraction: Sendable {
        var image: SourceImage
        var canvas: PixelSize
        /// Size of the photo inside the canvas (the rest is zero padding).
        var content: PixelSize
        /// Canvas pixels per original pixel, per axis.
        var scale: SIMD2<Double>
        var keypoints: [Float]
        var descriptors: [Float]
        var descriptorSize: Int
        var seconds: Double

        var count: Int { keypoints.count / 2 }

        var features: ImageFeatures {
            let inverse = SIMD2<Float>(1 / scale)
            let points = (0..<count).filter(isContent).map { index in
                Keypoint(x: learnedToOriginal(keypoints[2 * index], inverse.x, canvas: canvas.width),
                         y: learnedToOriginal(keypoints[2 * index + 1], inverse.y, canvas: canvas.height), size: 0,
                         angle: 0, response: 0)
            }
            return ImageFeatures(imageID: image.id, source: .racoLightGlue, workingSize: content, keypoints: points,
                                 extractionSeconds: seconds)
        }

        /// Descriptors of the first `limit` keypoints inside the photo (the extractor ranks them best first).
        func topDescriptors(_ limit: Int) -> [Float] {
            var result: [Float] = []
            result.reserveCapacity(limit * descriptorSize)
            var taken = 0
            for index in 0..<count where taken < limit && isContent(index) {
                result += descriptors[(index * descriptorSize)..<((index + 1) * descriptorSize)]
                taken += 1
            }
            return result
        }

        func isContent(_ index: Int) -> Bool {
            keypoints[2 * index] < Float(content.width) && keypoints[2 * index + 1] < Float(content.height)
        }

        /// LightGlue normalises each image by its own long edge around its centre.
        var normalisedKeypoints: [Float] {
            let half = Float(max(canvas.width, canvas.height)) / 2
            let cx = Float(canvas.width) / 2, cy = Float(canvas.height) / 2
            var result = keypoints
            for index in 0..<count {
                result[2 * index] = (keypoints[2 * index] - cx) / half
                result[2 * index + 1] = (keypoints[2 * index + 1] - cy) / half
            }
            return result
        }
    }

    /// Extractions per photo, and the ids of photos that could not be decoded.
    private nonisolated static func extractLearned(
        _ images: [SourceImage], _ session: LearnedSession?, _ progress: (@Sendable (ProgressEvent) -> Void)?,
        onExtracted: @escaping @Sendable (_ done: [Int: Extraction], _ latest: Extraction) -> Void = { _, _ in }
    ) async throws -> ([Int: Extraction], Set<Int>) {
        guard let session else { return ([:], []) }
        // A short pipeline: while the GPU runs the dense half for one photo, the next photo is decoded
        // and the previous one goes through the sparse half on the CPU.
        let depth = 3
        return try await withThrowingTaskGroup(of: (Int, Extraction?).self) { group in
            func start(_ image: SourceImage) {
                group.addTask {
                    try Task.checkCancellation()
                    do {
                        return (image.id, try Self.extract(image, session))
                    } catch let error as StitchError where error.isUnreadableImage {
                        return (image.id, nil)
                    }
                }
            }
            var pending = images.makeIterator()
            for _ in 0..<depth {
                guard let image = pending.next() else { break }
                start(image)
            }
            var extractions: [Int: Extraction] = [:]
            var failed = Set<Int>()
            for try await (id, extraction) in group {
                if let extraction {
                    extractions[id] = extraction
                    onExtracted(extractions, extraction)
                } else {
                    failed.insert(id)
                }
                progress?(ProgressEvent(stage: "lightglue-extract", completed: extractions.count + failed.count,
                                        total: images.count))
                try Task.checkCancellation()
                if let image = pending.next() { start(image) }
            }
            return (extractions, failed)
        }
    }

    /// Pairs that will be candidates whatever the affinity says.
    private nonisolated static func earlyPairs(
        _ images: [SourceImage], _ configuration: PipelineConfiguration
    ) -> [PairProposal.Key] {
        if configuration.pairSelection == .all || images.count <= 4 {
            return allPairs(images).map { PairProposal.Key($0.0.id, $0.1.id) }
        }
        let order = PairProposal.shootingOrder(images)
        return zip(order, order.dropFirst()).map { PairProposal.Key($0.0.id, $0.1.id) }
    }

    /// Matches pairs as they arrive, until the stream ends.
    private nonisolated static func matchLearned(
        _ pairs: AsyncStream<(Extraction, Extraction)>, _ session: LearnedSession?,
        _ configuration: PipelineConfiguration, total: Int, _ progress: (@Sendable (ProgressEvent) -> Void)?
    ) async throws -> [PairEvidence] {
        var evidence: [PairEvidence] = []
        for await (a, b) in pairs {
            guard let session else { continue }
            try Task.checkCancellation()
            evidence.append(try await Task.detached(priority: .userInitiated) {
                try Self.matchLearned(a, b, session, threshold: configuration.matchThreshold)
            }.value)
            progress?(ProgressEvent(stage: "lightglue-early", completed: evidence.count, total: total))
        }
        return evidence
    }

    private func matchLearned(
        _ pairs: [(SourceImage, SourceImage)], _ extractions: [Int: Extraction], _ session: LearnedSession,
        _ configuration: PipelineConfiguration, _ progress: (@Sendable (ProgressEvent) -> Void)?
    ) async throws -> [PairEvidence] {
        var evidence: [PairEvidence] = []
        for (index, (a, b)) in pairs.enumerated() {
            guard let ea = extractions[a.id], let eb = extractions[b.id] else { continue }
            try Task.checkCancellation()
            evidence.append(try await Task.detached(priority: .userInitiated) {
                try Self.matchLearned(ea, eb, session, threshold: configuration.matchThreshold)
            }.value)
            progress?(ProgressEvent(stage: "lightglue-match", completed: index + 1, total: pairs.count))
        }
        return evidence
    }

    private nonisolated static func extract(_ image: SourceImage, _ session: LearnedSession) throws -> Extraction {
        let start = ContinuousClock.now
        let canvas = LearnedModelSet.canvas(for: image)
        // Fit inside the canvas without distorting the aspect ratio.
        let fit = min(Double(canvas.width) / Double(image.pixelSize.width),
                      Double(canvas.height) / Double(image.pixelSize.height))
        let longSide = Int((Double(max(image.pixelSize.width, image.pixelSize.height)) * fit).rounded(.down))
        let working = try ImageLoader.planarRGB(for: image, longSide: longSide, canvas: canvas)
        let (keypoints, descriptors, size) = try session.extract(working.planar, canvas: canvas)
        return Extraction(
            image: image, canvas: canvas, content: working.size, scale: working.scale, keypoints: keypoints,
            descriptors: descriptors, descriptorSize: size, seconds: (ContinuousClock.now - start).seconds
        )
    }

    private nonisolated static func matchLearned(
        _ a: Extraction, _ b: Extraction, _ session: LearnedSession, threshold: Float
    ) throws -> PairEvidence {
        let start = ContinuousClock.now
        let count = min(a.count, b.count)
        let keypoints = Array(a.normalisedKeypoints.prefix(count * 2)) + Array(b.normalisedKeypoints.prefix(count * 2))
        let descriptors = Array(a.descriptors.prefix(count * a.descriptorSize)) +
            Array(b.descriptors.prefix(count * b.descriptorSize))
        let (partner, confidence) = try session.match(keypoints: keypoints, descriptors: descriptors, count: count,
                                                      descriptorSize: a.descriptorSize)
        let inverseA = SIMD2<Float>(1 / a.scale), inverseB = SIMD2<Float>(1 / b.scale)
        var matches: [TentativeMatch] = []
        for index in 0..<count where confidence[index] > threshold {
            let other = Int(partner[index])
            guard other >= 0, other < count, a.isContent(index), b.isContent(other) else { continue }
            matches.append(TentativeMatch(
                a: Point2(x: learnedToOriginal(a.keypoints[2 * index], inverseA.x, canvas: a.canvas.width),
                          y: learnedToOriginal(a.keypoints[2 * index + 1], inverseA.y, canvas: a.canvas.height)),
                b: Point2(x: learnedToOriginal(b.keypoints[2 * other], inverseB.x, canvas: b.canvas.width),
                          y: learnedToOriginal(b.keypoints[2 * other + 1], inverseB.y, canvas: b.canvas.height)),
                score: confidence[index]
            ))
        }
        return PairEvidence(
            a: a.image.id, b: b.image.id, source: .racoLightGlue, matches: matches, fits: [], chosenModel: nil,
            inlierCoverage: 0, confidence: 0, verdict: .tooFewMatches,
            matchingSeconds: (ContinuousClock.now - start).seconds, verificationSeconds: 0
        )
    }

    // MARK: - Candidate pairs

    /// Number of descriptors per photo used for the affinity between two photos.
    static let affinityDescriptors = 512

    private nonisolated static func candidates(
        _ images: [SourceImage], sift: [Int: (SIFTFeatures, ImageFeatures)], learned: [Int: Extraction],
        configuration: PipelineConfiguration, progress: (@Sendable (ProgressEvent) -> Void)?
    ) async -> ([CandidatePair], [PairProposal.Key: Int]) {
        let pairs = allPairs(images)
        // With few photos every pair is cheap, and seeing all of them is more informative.
        guard configuration.pairSelection == .proposed, images.count > 4 else {
            return (pairs.map { CandidatePair(a: $0.0.id, b: $0.1.id, affinity: nil, reasons: [.all]) }, [:])
        }
        // Learned descriptors when available (they are what LightGlue matches), RootSIFT otherwise.
        var subsets: [Int: (values: [Float], size: Int)] = [:]
        for image in images {
            if let e = learned[image.id] {
                subsets[image.id] = (e.topDescriptors(affinityDescriptors), e.descriptorSize)
            } else if let (handle, features) = sift[image.id] {
                subsets[image.id] = topSIFTDescriptors(handle, features, limit: affinityDescriptors)
            }
        }
        let affinity = await withTaskGroup(of: (PairProposal.Key, Int).self) { group in
            var added = 0
            for (a, b) in pairs {
                guard let da = subsets[a.id], let db = subsets[b.id], da.size == db.size, da.size > 0 else { continue }
                added += 1
                group.addTask {
                    let key = PairProposal.Key(a.id, b.id)
                    guard !Task.isCancelled else { return (key, 0) }
                    return (key, mutualMatches(da.values, db.values, size: da.size, ratio: 0.8))
                }
            }
            // 137,000 pairs for 524 photos: one event per half percent.
            let every = max(1, added / 200)
            var result: [PairProposal.Key: Int] = [:]
            for await (key, count) in group {
                result[key] = count
                if result.count % every == 0 || result.count == added {
                    progress?(ProgressEvent(stage: "affinity", completed: result.count, total: added))
                }
            }
            return result
        }
        return (PairProposal.propose(images: images, affinity: affinity, neighbours: configuration.proposalNeighbours),
                affinity)
    }

    private nonisolated static func topSIFTDescriptors(
        _ handle: SIFTFeatures, _ features: ImageFeatures, limit: Int
    ) -> (values: [Float], size: Int) {
        var size: Int32 = 0
        guard let raw = sc_features_descriptors(handle.pointer, &size), size > 0 else { return ([], 0) }
        let order = features.keypoints.indices.sorted { features.keypoints[$0].response > features.keypoints[$1].response }
        var values: [Float] = []
        values.reserveCapacity(min(limit, order.count) * Int(size))
        for index in order.prefix(limit) {
            values += UnsafeBufferPointer(start: raw + index * Int(size), count: Int(size))
        }
        return (values, Int(size))
    }

    /// Mutual nearest neighbours passing the ratio test between two descriptor sets.
    private nonisolated static func mutualMatches(_ a: [Float], _ b: [Float], size: Int, ratio: Float) -> Int {
        var raw: UnsafeMutablePointer<sc_match>?
        var message = [CChar](repeating: 0, count: 256)
        let count = sc_match_raw(a, Int32(a.count / size), b, Int32(b.count / size), Int32(size), ratio, 1, &raw,
                                 &message, message.count)
        sc_free(raw)
        return max(0, Int(count))
    }

    // MARK: - Verification

    private nonisolated static func verifyAll(
        _ pairs: [PairEvidence], images: [SourceImage], configuration: PipelineConfiguration,
        progress: (@Sendable (ProgressEvent) -> Void)?
    ) async -> [PairEvidence] {
        let lookup = Dictionary(uniqueKeysWithValues: images.map { ($0.id, $0) })
        return await withTaskGroup(of: (Int, PairEvidence).self) { group in
            for (index, pair) in pairs.enumerated() {
                group.addTask {
                    guard !Task.isCancelled, let a = lookup[pair.a], let b = lookup[pair.b] else { return (index, pair) }
                    return (index, Verifier.verify(pair, imageA: a, imageB: b, configuration: configuration))
                }
            }
            var results = pairs
            let every = max(1, pairs.count / 200)
            var done = 0
            for await (index, pair) in group {
                results[index] = pair
                done += 1
                if done % every == 0 || done == pairs.count {
                    progress?(ProgressEvent(stage: "verify", completed: done, total: pairs.count))
                }
            }
            return results
        }
    }

    private nonisolated static func allPairs(_ images: [SourceImage]) -> [(SourceImage, SourceImage)] {
        var pairs: [(SourceImage, SourceImage)] = []
        for i in images.indices {
            for j in images.indices where j > i {
                pairs.append((images[i], images[j]))
            }
        }
        return pairs
    }
}

/// Original-pixel coordinates follow OpenCV: pixel centres at integers. SIFT works in the same convention
/// on the downscaled image, so a point maps through the pixel centres.
func siftToOriginal(_ value: Float, _ inverseScale: Float) -> Float { (value + 0.5) * inverseScale - 0.5 }

/// The span RaCo's keypoints gain across the canvas, in canvas pixels. Its coarser feature levels (1/2, 1/8,
/// 1/32) are upsampled with align_corners, which stretches each by s - 1 pixels from edge to edge about the
/// canvas centre; the score map inherits a mix of them. Measured on synthetic photos shifted by known amounts:
/// the same on both axes, on 1024 and on 768 pixels alike, between 0.37 and 0.47 with the scene and the aspect
/// ratio; 0.42 is the middle, which leaves at most 0.05 / 1024 of the distance between two photos.
let racoStretch: Float = 0.42

/// RaCo-ALIKED keypoints are in edge coordinates of the canvas (pixel i spans [i, i + 1)): pulled back by
/// `racoStretch` towards the centre of the canvas side `canvas`, then mapped to original pixel centres.
func learnedToOriginal(_ value: Float, _ inverseScale: Float, canvas: Int) -> Float {
    let side = Float(canvas), centre = side / 2
    return (centre + (value - centre) * side / (side + racoStretch)) * inverseScale - 0.5
}

/// Reads a NUL-terminated message written by the C layer.
func errorText(_ buffer: [CChar]) -> String {
    String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

extension Duration {
    var seconds: Double {
        let (whole, fraction) = components
        return Double(whole) + Double(fraction) / 1e18
    }
}
