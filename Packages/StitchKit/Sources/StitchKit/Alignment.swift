import CStitchCore
import Foundation
import simd

/// The correspondences that drive global alignment, one set per verified pair of the main group.
struct AlignmentProblem: Sendable {
    struct Pair: Sendable {
        /// Indices into `images`, a < b.
        var a: Int
        var b: Int
        /// (ax, ay, bx, by) per correspondence, full-resolution pixels.
        var points: [Float]
        var sigmas: [Float]
        /// a -> b, row-major, from the refit; starts the homography solver.
        var homography: [Double]
        /// RootSIFT inliers alone, when there are enough: rotation bundle adjustment weights every point
        /// equally and SIFT localises better than the learned keypoints.
        var siftPoints: [Float]
        var siftSigmas: [Float]

        var count: Int { points.count / 4 }
    }

    var images: [SourceImage]
    var pairs: [Pair]
    /// Verification threshold per image, in its full-resolution pixels.
    var thresholds: [Double]

    /// Union of the verified matchers' inliers per pair, without duplicates, checked by one homography
    /// refit and capped to `cap` points spread over the photo.
    static func build(_ report: MatchReport, component: [Int], cap: Int = 300) -> AlignmentProblem {
        let images = component.compactMap { id in report.images.first { $0.id == id } }
        let index = Dictionary(uniqueKeysWithValues: images.enumerated().map { ($1.id, $0) })
        let configuration = report.configuration
        let thresholds = images.map {
            configuration.inlierThreshold * (Double($0.pixelSize.width * $0.pixelSize.height) / 1_000_000).squareRoot()
        }
        var grouped: [PairProposal.Key: [PairEvidence]] = [:]
        for evidence in report.pairs where evidence.verdict == .verified && evidence.chosenFit != nil {
            guard let a = index[evidence.a], let b = index[evidence.b] else { continue }
            grouped[PairProposal.Key(a, b), default: []].append(evidence)
        }

        var pairs: [Pair] = []
        for key in grouped.keys.sorted() {
            let sources = grouped[key]!.sorted { $0.source.rawValue < $1.source.rawValue }
            var candidates: [(point: SIMD4<Float>, sigma: Float)] = []
            var perSource: [(source: FeatureSource, points: [SIMD4<Float>], sigma: Float, transform: [Double])] = []
            for evidence in sources {
                guard let fit = evidence.chosenFit else { continue }
                let sigma = Float(max(0.5, fit.rmse))
                let flipped = index[evidence.a] != key.a
                let points = fit.inliers.compactMap { i -> SIMD4<Float>? in
                    guard evidence.matches.indices.contains(Int(i)) else { return nil }
                    let m = evidence.matches[Int(i)]
                    let (p, q) = flipped ? (m.b, m.a) : (m.a, m.b)
                    return SIMD4(p.x, p.y, q.x, q.y)
                }
                let transform = flipped ? inverse(fit.transform) : fit.transform
                perSource.append((evidence.source, points, sigma, transform))
                candidates += points.map { ($0, sigma) }
            }
            guard let best = perSource.max(by: { $0.points.count < $1.points.count }) else { continue }
            let threshold = thresholds[key.b]
            var union = deduplicate(candidates, radius: Float(0.5 * threshold))
            var homography = best.transform
            if let refit = refit(union, threshold: threshold, seed: configuration.seed) {
                union = refit.kept
                homography = refit.homography
            }
            // When the matchers disagree, the refit keeps few points: trust the best one alone.
            if Double(union.count) < 0.9 * Double(best.points.count) {
                union = best.points.map { ($0, best.sigma) }
                homography = best.transform
            }
            let spread = stratified(union, size: images[key.a].pixelSize, cap: cap)
            let sift = perSource.first { $0.source == .rootSIFT }
            let siftSpread = sift.flatMap { set in
                set.points.count >= 40
                    ? stratified(set.points.map { ($0, set.sigma) }, size: images[key.a].pixelSize, cap: cap) : nil
            }
            pairs.append(Pair(
                a: key.a, b: key.b,
                points: spread.flatMap { [$0.point.x, $0.point.y, $0.point.z, $0.point.w] },
                sigmas: spread.map(\.sigma), homography: homography,
                siftPoints: siftSpread?.flatMap { [$0.point.x, $0.point.y, $0.point.z, $0.point.w] } ?? [],
                siftSigmas: siftSpread?.map(\.sigma) ?? []
            ))
        }
        return AlignmentProblem(images: images, pairs: pairs, thresholds: thresholds)
    }

    /// Keeps the more precise of two correspondences that land within `radius` of each other in both photos.
    static func deduplicate(_ points: [(point: SIMD4<Float>, sigma: Float)], radius: Float)
        -> [(point: SIMD4<Float>, sigma: Float)] {
        guard radius > 0 else { return points }
        var kept: [(point: SIMD4<Float>, sigma: Float)] = []
        var cells: [SIMD2<Int>: [Int]] = [:]
        for candidate in points.sorted(by: { $0.sigma < $1.sigma }) {
            let cell = SIMD2(Int((candidate.point.x / radius).rounded(.down)), Int((candidate.point.y / radius).rounded(.down)))
            var duplicate = false
            search: for dx in -1...1 {
                for dy in -1...1 {
                    for k in cells[cell &+ SIMD2(dx, dy), default: []] {
                        let d = abs(kept[k].point - candidate.point)
                        if max(d.x, d.y) < radius && max(d.z, d.w) < radius {
                            duplicate = true
                            break search
                        }
                    }
                }
            }
            guard !duplicate else { continue }
            cells[cell, default: []].append(kept.count)
            kept.append(candidate)
        }
        return kept
    }

    /// One homography over the union; its inliers and the matrix, or nil when the fit fails.
    static func refit(_ points: [(point: SIMD4<Float>, sigma: Float)], threshold: Double, seed: Int32)
        -> (kept: [(point: SIMD4<Float>, sigma: Float)], homography: [Double])? {
        guard points.count >= 8 else { return nil }
        let a = points.flatMap { [$0.point.x, $0.point.y] }, b = points.flatMap { [$0.point.z, $0.point.w] }
        var mask = [UInt8](repeating: 0, count: points.count)
        let fit = sc_fit_model(a, b, Int32(points.count), SC_MODEL_HOMOGRAPHY, threshold, 5000, 0.999, seed, &mask)
        guard fit.ok != 0 else { return nil }
        let transform = withUnsafeBytes(of: fit.transform) { Array($0.bindMemory(to: Double.self)) }
        return (points.indices.filter { mask[$0] != 0 }.map { points[$0] }, transform)
    }

    /// At most `cap` points, taken in turn from an 8 x 8 grid over photo a so they cover it evenly.
    static func stratified(_ points: [(point: SIMD4<Float>, sigma: Float)], size: PixelSize, cap: Int)
        -> [(point: SIMD4<Float>, sigma: Float)] {
        guard points.count > cap else { return points }
        var buckets = [[(point: SIMD4<Float>, sigma: Float)]](repeating: [], count: 64)
        for p in points {
            let gx = min(7, max(0, Int(p.point.x / Float(max(size.width, 1)) * 8)))
            let gy = min(7, max(0, Int(p.point.y / Float(max(size.height, 1)) * 8)))
            buckets[gy * 8 + gx].append(p)
        }
        for i in buckets.indices { buckets[i].sort { $0.sigma < $1.sigma } }
        var result: [(point: SIMD4<Float>, sigma: Float)] = []
        var round = 0
        while result.count < cap {
            var added = false
            for bucket in buckets where round < bucket.count && result.count < cap {
                result.append(bucket[round])
                added = true
            }
            if !added { break }
            round += 1
        }
        return result
    }

    static func inverse(_ rowMajor: [Double]) -> [Double] {
        let m = PlaneGeometry.matrix(rowMajor)
        guard abs(m.determinant) > 1e-12 else { return rowMajor }
        let i = m.inverse
        return [i[0, 0], i[1, 0], i[2, 0], i[0, 1], i[1, 1], i[2, 1], i[0, 2], i[1, 2], i[2, 2]]
    }

    /// Whether the pairs connect every image.
    func isConnected(without removed: Set<Int> = []) -> Bool {
        let edges = pairs.indices.filter { !removed.contains($0) }.map { PairProposal.Key(pairs[$0].a, pairs[$0].b) }
        return PairProposal.components(Array(images.indices), edges: edges).count == 1
    }

    /// The problem restricted to `keep` (indices into `images`, ascending), with the pairs among them.
    func subset(_ keep: [Int]) -> AlignmentProblem {
        let index = Dictionary(uniqueKeysWithValues: keep.enumerated().map { ($1, $0) })
        let kept = pairs.compactMap { pair -> Pair? in
            guard let a = index[pair.a], let b = index[pair.b] else { return nil }
            var copy = pair
            copy.a = a
            copy.b = b
            return copy
        }
        return AlignmentProblem(images: keep.map { images[$0] }, pairs: kept, thresholds: keep.map { thresholds[$0] })
    }

    /// Going round a loop of three photos through their pairs brings a point back where it started when
    /// the three pairs are right. A false pair, between two rows of identical labels or parking spaces,
    /// throws it hundreds of pixels off, and so do the triangles it is part of. A pair is left out when more
    /// of its triangles disagree than agree, and at least two do: one triangle alone cannot tell which of its
    /// pairs is wrong. Pairs in no triangle are kept. A triangle agrees when, starting from any of its pairs'
    /// matches, the way round lands within eight inlier thresholds of the match.
    /// Returns the largest group the remaining pairs join, the pairs left out and the photos left out
    /// (indices into this problem).
    func consistentLoops() -> (problem: AlignmentProblem, removed: [Int], leftOut: [Int]) {
        var index: [PairProposal.Key: Int] = [:]
        for (p, pair) in pairs.enumerated() { index[PairProposal.Key(pair.a, pair.b)] = p }
        var neighbours = [Set<Int>](repeating: [], count: images.count)
        for pair in pairs {
            neighbours[pair.a].insert(pair.b)
            neighbours[pair.b].insert(pair.a)
        }
        let maps = pairs.map { PlaneGeometry.matrix($0.homography) }
        func map(_ x: Int, _ y: Int) -> simd_double3x3 {
            let p = index[PairProposal.Key(x, y)]!
            return pairs[p].a == x ? maps[p] : maps[p].inverse
        }
        // The matches of the pair x-y, thinned evenly to 16-31 when there are more, as (point in x, point in y).
        func matches(_ x: Int, _ y: Int) -> [(SIMD2<Double>, SIMD2<Double>)] {
            let pair = pairs[index[PairProposal.Key(x, y)]!]
            let step = max(1, pair.count / 16)
            return stride(from: 0, to: pair.count, by: step).map { k in
                let a = SIMD2(Double(pair.points[4 * k]), Double(pair.points[4 * k + 1]))
                let b = SIMD2(Double(pair.points[4 * k + 2]), Double(pair.points[4 * k + 3]))
                return pair.a == x ? (a, b) : (b, a)
            }
        }
        // Whether the matches of x-z land on their place in z through y.
        func closes(_ x: Int, _ y: Int, _ z: Int) -> Bool {
            let round = map(y, z) * map(x, y)
            let errors = matches(x, z).map { p, q in
                PlaneGeometry.apply(round, p.x, p.y).map { simd_distance($0, q) } ?? .infinity
            }.sorted()
            guard let median = errors.dropFirst(errors.count / 2).first else { return false }
            return median <= 8 * thresholds[z]
        }
        var triangles: [(edges: [Int], agrees: Bool)] = []
        for pair in pairs {
            let i = pair.a, j = pair.b
            for k in neighbours[i].intersection(neighbours[j]) where k > max(i, j) {
                let edges = [index[PairProposal.Key(i, j)]!, index[PairProposal.Key(j, k)]!, index[PairProposal.Key(i, k)]!]
                triangles.append((edges, closes(i, j, k) || closes(i, k, j) || closes(j, i, k)))
            }
        }
        var agree = [Int](repeating: 0, count: pairs.count), disagree = agree
        var trianglesOf = [[Int]](repeating: [], count: pairs.count)
        for (t, triangle) in triangles.enumerated() {
            for e in triangle.edges {
                trianglesOf[e].append(t)
                if triangle.agrees { agree[e] += 1 } else { disagree[e] += 1 }
            }
        }
        var alive = [Bool](repeating: true, count: triangles.count)
        var removed: [Int] = []
        while let worst = pairs.indices.filter({ disagree[$0] >= 2 && disagree[$0] > agree[$0] })
            .max(by: { (disagree[$0] - agree[$0], disagree[$0]) < (disagree[$1] - agree[$1], disagree[$1]) }) {
            removed.append(worst)
            for t in trianglesOf[worst] where alive[t] {
                alive[t] = false
                for e in triangles[t].edges {
                    if triangles[t].agrees { agree[e] -= 1 } else { disagree[e] -= 1 }
                }
            }
        }
        guard !removed.isEmpty else { return (self, [], []) }
        let gone = Set(removed)
        let edges = pairs.indices.filter { !gone.contains($0) }.map { PairProposal.Key(pairs[$0].a, pairs[$0].b) }
        let groups = PairProposal.components(Array(images.indices), edges: edges)
        let largest = groups.max { $0.count < $1.count } ?? Set(images.indices)
        var kept = self
        kept.pairs = pairs.indices.filter { !gone.contains($0) }.map { pairs[$0] }
        let keep = largest.sorted()
        return (kept.subset(keep), removed, images.indices.filter { !largest.contains($0) })
    }

    /// The `count` images reached first from `centre` through the pairs (strongest pairs first), with the
    /// pairs among them, and the centre's index in it; nil when that leaves no pair.
    func neighbourhood(of centre: Int, count: Int) -> (problem: AlignmentProblem, centre: Int)? {
        var chosen = [centre], seen: Set<Int> = [centre]
        var frontier = [centre]
        let strongest = pairs.sorted { $0.count > $1.count }
        while chosen.count < count, !frontier.isEmpty {
            var next: [Int] = []
            for image in frontier {
                for pair in strongest where pair.a == image || pair.b == image {
                    let other = pair.a == image ? pair.b : pair.a
                    guard chosen.count < count, seen.insert(other).inserted else { continue }
                    chosen.append(other)
                    next.append(other)
                }
            }
            frontier = next
        }
        let order = chosen.sorted()
        let trial = subset(order)
        guard !trial.pairs.isEmpty, let local = order.firstIndex(of: centre) else { return nil }
        return (trial, local)
    }
}

/// The outcome of global alignment.
struct Alignment: Sendable {
    var model: GlobalModel
    var anchor: Int
    /// Per photo, row-major: planar models map photo pixels to the mosaic (the anchor's pixels);
    /// rotation holds cv::detail's camera rotation.
    var transforms: [[Double]]
    var focals: [Double]
    var rms: Double
    var pairRMS: [Double]
    /// Rotation: 0 ray bundle adjustment, 1 focal fixed at the prior.
    var method: Int32
}

enum Aligner {
    /// Solves `model` once. Throws with the solver's message when it fails.
    static func solve(_ model: GlobalModel, _ problem: AlignmentProblem, anchor: Int, straighten: Bool = true)
        throws -> Alignment {
        let n = problem.images.count
        let native: sc_align_model = switch model {
        case .translation: SC_ALIGN_TRANSLATION
        case .similarity: SC_ALIGN_SIMILARITY
        case .affine: SC_ALIGN_AFFINE
        case .homography: SC_ALIGN_HOMOGRAPHY
        case .rotation: SC_ALIGN_ROTATION
        }
        // Rotation: SIFT points alone where a pair has enough, and fewer points per pair as the group grows,
        // because the bundle adjuster's Jacobian is dense.
        let useSIFT = model == .rotation
        let cap = model == .rotation ? min(300, max(40, Int(1.4e6 / Double(max(1, n * problem.pairs.count))))) : Int.max
        var flat: [Float] = [], sigmas: [Float] = []
        var ranges: [(offset: Int, count: Int)] = []
        for pair in problem.pairs {
            let points = useSIFT && !pair.siftPoints.isEmpty ? pair.siftPoints : pair.points
            let weights = useSIFT && !pair.siftPoints.isEmpty ? pair.siftSigmas : pair.sigmas
            let count = min(points.count / 4, cap)
            ranges.append((sigmas.count, count))
            flat += points.prefix(count * 4)
            sigmas += weights.prefix(count)
        }
        let images = problem.images.map { image -> sc_align_image in
            // 35 mm equivalent focal length by the diagonal (CIPA): 43.27 mm for the full-frame diagonal.
            let diagonal = (Double(image.pixelSize.width * image.pixelSize.width
                                   + image.pixelSize.height * image.pixelSize.height)).squareRoot()
            let focal = image.focalLength35mm.map { $0 * diagonal / 43.2666 } ?? 0
            return sc_align_image(width: Int32(image.pixelSize.width), height: Int32(image.pixelSize.height), focal: focal)
        }
        var transforms = [Double](repeating: 0, count: 9 * n)
        var focals = [Double](repeating: 0, count: n)
        var pairRMS = [Double](repeating: 0, count: problem.pairs.count)
        var result = sc_align_result()
        var message = [CChar](repeating: 0, count: 512)
        let status = flat.withUnsafeBufferPointer { points in
            sigmas.withUnsafeBufferPointer { weights in
                let pairs = zip(problem.pairs, ranges).map { pair, range in
                    let h = pair.homography
                    return sc_align_pair(a: Int32(pair.a), b: Int32(pair.b), count: Int32(range.count),
                                         points: points.baseAddress! + 4 * range.offset,
                                         sigma: weights.baseAddress! + range.offset,
                                         homography: (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]))
                }
                return sc_align(native, images, Int32(n), pairs, Int32(pairs.count), Int32(anchor),
                                model == .rotation && straighten ? 2 : -1, &transforms, &focals, &pairRMS, &result,
                                &message, message.count)
            }
        }
        guard status == 0, result.ok != 0 else { throw StitchError.engine(errorText(message)) }
        // Errors are measured in the anchor's pixels, so a similarity or affine fit can lower them by
        // shrinking photos towards a point: on 511 drone photos most fell to a thousandth of their size.
        // More photos scaled beyond `overstretch` either way than the misplaced-photo budget is no layout.
        if model == .similarity || model == .affine {
            let scaled = (0..<n).filter { i in
                let s = abs(transforms[9 * i] * transforms[9 * i + 4] - transforms[9 * i + 1] * transforms[9 * i + 3]).squareRoot()
                return !(s.isFinite && s > 0) || max(s, 1 / s) > overstretch
            }
            if scaled.count > max(1, n / 20) { throw StitchError.engine(String(localized: "No global model fits these photos")) }
        }
        return Alignment(model: model, anchor: anchor, transforms: (0..<n).map { Array(transforms[(9 * $0)..<(9 * $0 + 9)]) },
                         focals: focals, rms: result.rms, pairRMS: pairRMS, method: result.iterations)
    }

    /// Worst corner stretch of a planar solution seen from `anchor`'s plane.
    static func stretch(_ alignment: Alignment, _ problem: AlignmentProblem, anchor: Int) -> Double {
        let flat = alignment.transforms.flatMap { $0 }
        let sizes = problem.images.flatMap { [Int32($0.pixelSize.width), Int32($0.pixelSize.height)] }
        return sc_alignment_stretch(flat, sizes, Int32(problem.images.count), Int32(anchor))
    }

    /// Homographies onto the photo whose plane stretches the others least (min-max over corners).
    static func homographies(_ problem: AlignmentProblem, anchor: Int) throws -> (Alignment, stretch: Double) {
        var alignment = try solve(.homography, problem, anchor: anchor)
        let ranked = problem.images.indices.map { ($0, stretch(alignment, problem, anchor: $0)) }
        if let best = ranked.min(by: { $0.1 < $1.1 }), best.0 != anchor, best.1.isFinite {
            if let better = try? solve(.homography, problem, anchor: best.0) { alignment = better }
        }
        return (alignment, stretch(alignment, problem, anchor: alignment.anchor))
    }

    /// Picks the global model for `mode` (strategy: the simplest model within a tolerance of the best fit,
    /// a rotation only when its focal length is plausible), then drops pairs that disagree with the rest while
    /// the group stays connected: up to two, or a quarter of the pairs beyond a spanning tree when that is more.
    /// Returns the problem without the pairs it dropped.
    /// The pairs it drops are given by photo id, with their error, and so are the photos it leaves out.
    /// `provisional` alignments, which only lay the photos out, throw `ProvisionalSkip` instead of solving
    /// the rotation for more than `rotationTrial.limit` photos. `automaticTrial` marks Automatic mode's
    /// own Document trial, which may leave photos out as Automatic mode does.
    static func align(_ problem: AlignmentProblem, mode: StitchMode, centre: Int, straighten: Bool,
                      provisional: Bool = false, automaticTrial: Bool = false) throws
        -> (alignment: Alignment, problem: AlignmentProblem, notes: [String], dropped: [(a: Int, b: Int, off: Double)],
            misplaced: [Int]) {
        var problem = problem
        var centre = centre
        var modelNotes: [String] = []
        var alignment = try choose(problem, mode: mode, centre: centre, straighten: straighten, provisional: provisional,
                                   notes: &modelNotes)
        // Homographies that fit most pairs far better than the model chosen, but that a few false pairs or
        // misplaced photos stretch out of the running (median pair error 8.8 px against 79 px for the affine
        // on 511 drone photos): Document mode's own clean-up may rescue them. They are kept when they then
        // earn their perspective, or when they leave the planar fits four times behind and stretch no photo
        // more than twice the usual limit: a survey of uneven ground drifts into a gentle, even perspective
        // (13.8 px against 700 px, stretched 4.9 times at the far end).
        if mode == .auto, alignment.model != .homography, alignment.model != .rotation,
           let (document, _) = try? homographies(problem, anchor: centre),
           median(document.pairRMS) < 0.5 * median(alignment.pairRMS) {
            try Task.checkCancellation()
            if let trial = try? align(problem, mode: .document, centre: centre, straighten: straighten,
                                      provisional: provisional, automaticTrial: true),
               trial.alignment.model == .homography,
               let planar = [GlobalModel.affine, .similarity, .translation].compactMap({
                   try? solve($0, trial.problem, anchor: trial.alignment.anchor)
               }).min(by: { $0.rms < $1.rms }) {
                let spread = stretch(trial.alignment, trial.problem, anchor: trial.alignment.anchor)
                if earnsPerspective(documentRMS: trial.alignment.rms, stretch: spread, planarRMS: planar.rms)
                    || (trial.alignment.rms < 0.25 * planar.rms && spread <= 2 * overstretch) {
                    // Document mode's hint at Rotation mode does not apply to a set Automatic found flat.
                    let hint = String(localized: "Some photos are stretched a lot on the reference plane: Rotation mode may suit them better")
                    return (trial.alignment, trial.problem, trial.notes.filter { $0 != hint }, trial.dropped, trial.misplaced)
                }
            }
        }
        // A new choice of model, or nil when it fails for any reason but cancellation.
        func chooseAgain(_ problem: AlignmentProblem, centre: Int) throws -> (Alignment, [String])? {
            var notes: [String] = []
            do {
                return (try choose(problem, mode: mode, centre: centre, straighten: straighten, provisional: provisional,
                                   notes: &notes), notes)
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                return nil
            }
        }
        // A few photos that the homographies stretch beyond belief would rule the homographies out for the
        // whole set (the first photos of a drone survey, taken while it climbed): in Automatic mode they are
        // left out, with any photo only they joined, when the homographies then win. Before dropping pairs,
        // which otherwise go by the errors of a model the homographies would beat, and after. Not for a
        // rotating camera, whose outer photos the homographies stretch by nature, nor in Document mode chosen
        // by hand, where the photos stay and the note suggests Rotation mode.
        var misplaced: [Int] = []
        let budget = max(1, problem.images.count / 20)
        func leaveOutMisplaced() throws -> Bool {
            // In a small set one stretched photo is as likely the outer photo of a turning camera.
            guard mode == .auto || automaticTrial, alignment.model != .rotation, budget >= 2 else { return false }
            try Task.checkCancellation()
            let stretched = Set(overstretched(problem, anchor: centre))
            guard !stretched.isEmpty, misplaced.count + stretched.count <= budget else { return false }
            let remaining = problem.subset(problem.images.indices.filter { !stretched.contains($0) })
            let groups = PairProposal.components(Array(remaining.images.indices),
                                                 edges: remaining.pairs.map { PairProposal.Key($0.a, $0.b) })
            // Photos only the stretched ones joined count against the budget too: a stretched photo between
            // two halves of the set would otherwise take one half with it.
            guard let largest = groups.max(by: { $0.count < $1.count }), largest.count >= 2,
                  misplaced.count + problem.images.count - largest.count <= budget else { return false }
            let reduced = remaining.subset(largest.sorted())
            let centreID = problem.images[centre].id
            let newCentre = reduced.images.firstIndex { $0.id == centreID } ?? 0
            // Kept only when the homographies win without them: a planar model placed them as well as the rest.
            guard let (chosen, notes) = try chooseAgain(reduced, centre: newCentre), chosen.model == .homography else {
                return false
            }
            let kept = Set(reduced.images.map(\.id))
            misplaced += problem.images.map(\.id).filter { !kept.contains($0) }
            problem = reduced
            centre = newCentre
            alignment = chosen
            modelNotes = notes
            return true
        }
        _ = try leaveOutMisplaced()

        var dropped: [(a: Int, b: Int, off: Double)] = []
        var stale = false
        let limit = max(2, (problem.pairs.count - (problem.images.count - 1)) / 4)
        while dropped.count < limit {
            try Task.checkCancellation()
            let sorted = alignment.pairRMS.sorted()
            guard problem.pairs.count >= problem.images.count, let median = sorted.dropFirst(sorted.count / 2).first,
                  let worstError = sorted.last else { break }
            // The worst pair and, with it, every pair as far off as half of it: many independent outliers go
            // in one solve instead of one solve each (37 on 524 drone photos), a single one stays alone.
            let offenders = alignment.pairRMS.indices
                .filter { alignment.pairRMS[$0] > max(3 * median, problem.thresholds[problem.pairs[$0].b], worstError / 2) }
                .sorted { alignment.pairRMS[$0] > alignment.pairRMS[$1] }
            var removing: [Int] = []
            for index in offenders where dropped.count + removing.count < limit {
                if problem.isConnected(without: Set(removing + [index])) { removing.append(index) }
            }
            guard !removing.isEmpty else { break }
            let gone = Set(removing)
            var reduced = problem
            reduced.pairs = problem.pairs.indices.filter { !gone.contains($0) }.map { problem.pairs[$0] }
            // Without outliers the same model fits better. When it fits worse, it leaned on them (a rotation
            // a false pair made look plausible, say): choose the model again.
            let next: Alignment
            if let resolved = try? solve(alignment.model, reduced, anchor: alignment.anchor, straighten: straighten),
               resolved.rms <= alignment.rms {
                next = resolved
                stale = true
            } else if let (chosen, notes) = try chooseAgain(reduced, centre: centre) {
                next = chosen
                modelNotes = notes
                stale = false
            } else {
                break
            }
            dropped += removing.map {
                (problem.images[problem.pairs[$0].a].id, problem.images[problem.pairs[$0].b].id, alignment.pairRMS[$0])
            }
            problem = reduced
            alignment = next
        }
        // The model was chosen with the pairs now left out.
        if stale, let (chosen, notes) = try chooseAgain(problem, centre: centre) {
            alignment = chosen
            modelNotes = notes
        }
        // Leaving photos out can bring others over the limit; a few more rounds, within the same budget.
        for _ in 0..<3 {
            guard try leaveOutMisplaced() else { break }
        }
        // Each pair was verified to within its own inlier threshold: a global fit looser than that has not joined them.
        if !alignment.pairRMS.isEmpty {
            let ratios = zip(alignment.pairRMS, problem.pairs).map { $0 / problem.thresholds[$1.b] }
            let excess = (ratios.map { $0 * $0 }.reduce(0, +) / Double(ratios.count)).squareRoot()
            if excess > 1 {
                modelNotes.append(String(format: String(localized: "No model fits these photos closely: the alignment error is %.1f times the error allowed within each pair"),
                                         locale: .current, excess))
            }
        }
        return (alignment, problem, modelNotes, dropped, misplaced)
    }

    private static func median(_ values: [Double]) -> Double { values.sorted().dropFirst(values.count / 2).first ?? .infinity }

    /// Beyond this stretch of a photo on the reference plane a homography is not trusted.
    static let overstretch = 4.0

    /// Photos that the homographies onto `anchor`'s plane stretch more than `overstretch` times at a corner.
    static func overstretched(_ problem: AlignmentProblem, anchor: Int) -> [Int] {
        guard let (fit, _) = try? homographies(problem, anchor: anchor) else { return [] }
        let reference = PlaneGeometry.matrix(fit.transforms[fit.anchor]).inverse
        return problem.images.indices.filter { i in
            let m = reference * PlaneGeometry.matrix(fit.transforms[i])
            let w = Double(problem.images[i].pixelSize.width), h = Double(problem.images[i].pixelSize.height)
            return [(0.0, 0.0), (w, 0), (w, h), (0, h)].contains { corner in
                guard let j = PlaneGeometry.jacobian(m, at: SIMD2(corner.0, corner.1)) else { return true }
                let s = abs(j.determinant).squareRoot()
                return !(s.isFinite && s > 0) || max(s, 1 / s) > overstretch
            }
        }
    }

    /// Automatic mode tries the rotation on `size` photos first when there are more than `limit`.
    static let rotationTrial = (limit: 60, size: 40)

    /// Whether a homography earns its perspective over the planar fit it is compared with: it stretches no
    /// photo more than `overstretch` times and halves the planar error. Parallax (specimens on pins, a
    /// hand-held camera) lowers a homography's error a little while the perspective builds up along a chain
    /// of photos; a real perspective leaves the planar fits far behind.
    static func earnsPerspective(documentRMS: Double, stretch: Double, planarRMS: Double) -> Bool {
        stretch.isFinite && stretch <= overstretch && documentRMS < 0.5 * planarRMS
    }

    /// Thrown by provisional alignments instead of solving the rotation for a large set.
    struct ProvisionalSkip: Error {}

    private static func choose(_ problem: AlignmentProblem, mode: StitchMode, centre: Int, straighten: Bool,
                               provisional: Bool = false, notes: inout [String]) throws -> Alignment {
        func simplest(_ fits: [Alignment]) -> Alignment? {
            guard let best = fits.map(\.rms).min() else { return nil }
            let tolerance = 1.25 * best + 0.5
            return fits.first { $0.rms <= tolerance }
        }
        switch mode {
        case .plane:
            let fits = [GlobalModel.translation, .similarity, .affine].compactMap { try? solve($0, problem, anchor: centre) }
            try Task.checkCancellation()
            guard let chosen = simplest(fits) else { throw StitchError.engine(String(localized: "The tiles could not be aligned")) }
            // Against the best planar fit: the simplest one may leave an error that an affine fit removes.
            if let (document, stretch) = try? homographies(problem, anchor: centre),
               let best = fits.min(by: { $0.rms < $1.rms }),
               earnsPerspective(documentRMS: document.rms, stretch: stretch, planarRMS: best.rms),
               best.rms - document.rms > 1 {
                notes.append(String(format: String(localized: "The photos show perspective: Document mode would align them to %.1f px instead of %.1f px"),
                                    locale: .current, document.rms, chosen.rms))
            }
            return chosen
        case .document:
            let (document, stretch) = try homographies(problem, anchor: centre)
            guard stretch.isFinite else {
                throw StitchError.engine(String(localized: "The photos do not lie on one plane: try Rotation mode"))
            }
            if stretch > overstretch {
                notes.append(String(localized: "Some photos are stretched a lot on the reference plane: Rotation mode may suit them better"))
            }
            return document
        case .rotation:
            if provisional, problem.images.count > Self.rotationTrial.limit { throw ProvisionalSkip() }
            do {
                return try solve(.rotation, problem, anchor: centre, straighten: straighten)
            } catch {
                guard let (document, stretch) = try? homographies(problem, anchor: centre), stretch < overstretch else {
                    throw error
                }
                notes.append(String(localized: "The photos do not fit a rotating camera; they were joined as a flat document"))
                return document
            }
        case .auto:
            let planar = [GlobalModel.translation, .similarity, .affine].compactMap { try? solve($0, problem, anchor: centre) }
            try Task.checkCancellation()
            // The tolerance is at least half a pixel and at most 1.25 times the best planar fit plus half a pixel:
            // these choices do not need the homographies or the rotation, the slowest of the fits.
            if let t = planar.first(where: { $0.model == .translation }), t.rms <= 0.5 { return t }
            if let lowest = planar.map(\.rms).min(), let s = planar.first(where: { $0.model == .similarity }), s.rms <= 0.5,
               planar.first(where: { $0.model == .translation }).map({ $0.rms > 1.25 * lowest + 0.5 }) ?? true {
                return s
            }
            let document = try? homographies(problem, anchor: centre)
            try Task.checkCancellation()
            let documentStretched = document.map { !$0.stretch.isFinite || $0.stretch > overstretch } ?? true
            // The rotation's bundle adjustment is dense: with hundreds of photos it runs for hours. Above
            // `rotationTrial` photos it is tried on the ones nearest the centre first, and solved for all
            // of them only when it wins there, or when the homographies of the whole set are stretched and
            // it beats the affine there: a wide turn stretches the whole set's homographies, not the
            // centre's, while a drone survey's photos do not fit a rotation even nearby.
            let rotation: Alignment?
            if problem.images.count > Self.rotationTrial.limit,
               let trial = problem.neighbourhood(of: centre, count: Self.rotationTrial.size) {
                var ignored: [String] = []
                let model = try? choose(trial.problem, mode: .auto, centre: trial.centre, straighten: straighten, notes: &ignored)
                try Task.checkCancellation()
                var wins = model?.model == .rotation
                if !wins, documentStretched,
                   let local = try? solve(.rotation, trial.problem, anchor: trial.centre, straighten: straighten),
                   let affine = try? solve(.affine, trial.problem, anchor: trial.centre) {
                    wins = local.rms < affine.rms
                }
                try Task.checkCancellation()
                if wins, provisional { throw ProvisionalSkip() }
                rotation = wins ? try? solve(.rotation, problem, anchor: centre, straighten: straighten) : nil
            } else {
                rotation = try? solve(.rotation, problem, anchor: centre, straighten: straighten)
            }
            try Task.checkCancellation()
            let all = planar.map(\.rms) + [document?.0.rms, rotation?.rms].compactMap { $0 }
            guard let best = all.min() else { throw StitchError.engine(String(localized: "No global model fits these photos")) }
            let tolerance = 1.25 * best + 0.5
            let byModel = Dictionary(uniqueKeysWithValues: planar.map { ($0.model, $0) })
            if let t = byModel[.translation], t.rms <= tolerance { return t }
            if let s = byModel[.similarity], s.rms <= tolerance { return s }
            // Stretched homographies point to a turning camera, unless the rotation fits worse than an affine:
            // then a misplaced photo stretched them, and align(_:) leaves it out.
            let affineCloser = byModel[.affine].map { affine in rotation.map { affine.rms < $0.rms } ?? true } ?? false
            if let rotation, rotation.rms <= max(tolerance, 1.5 * (document?.0.rms ?? .infinity) + 1)
                || (documentStretched && !affineCloser) {
                return rotation
            }
            if let a = byModel[.affine] {
                let earned = document.map {
                    earnsPerspective(documentRMS: $0.0.rms, stretch: $0.stretch, planarRMS: a.rms)
                } ?? false
                if a.rms <= tolerance || !earned { return a }
            }
            if let document, !documentStretched { return document.0 }
            if let rotation { return rotation }
            // The similarity and the affine can be ruled out for shrinking the photos: the translation is left.
            if let last = planar.min(by: { $0.rms < $1.rms }) { return last }
            throw StitchError.engine(String(localized: "No global model fits these photos"))
        }
    }
}

/// The global alignment of a report's main group.
struct AlignmentOutcome {
    var problem: AlignmentProblem
    var alignment: Alignment
    var notes: [String]
    var leftOutPairs: [LeftOutPair]
    /// Photos of the main group joined only through pairs whose loops do not close.
    var leftOutImages: [Int]
    /// Photos the homographies stretched more than `Aligner.overstretch` times, and those only they joined.
    var misplacedImages: [Int] = []
}

/// A verified pair of photos that the global alignment did not use.
public struct LeftOutPair: Sendable, Hashable {
    public enum Reason: Sendable, Hashable {
        /// More of the loops of three photos it is part of fail to close than close.
        case brokenLoops
        /// It stayed this far off the other pairs after alignment, in pixels.
        case offBy(Double)
    }

    /// Photo ids.
    public var a: Int
    public var b: Int
    public var reason: Reason

    public func text(_ name: (Int) -> String) -> String {
        switch reason {
        case .brokenLoops:
            String(format: String(localized: "The pair %@ ↔ %@ was left out of the alignment: its loops with other pairs do not close"),
                   locale: .current, name(a), name(b))
        case .offBy(let pixels):
            String(format: String(localized: "The pair %@ ↔ %@ was left out of the alignment (%.1f px off the others)"),
                   locale: .current, name(a), name(b), pixels)
        }
    }
}

/// What global alignment found, for diagnostics.
public struct AlignmentSummary: Sendable {
    public var model: GlobalModel
    public var rms: Double
    public var anchor: String
    /// (photo a, photo b, RMS transfer error) per pair used.
    public var pairs: [(String, String, Double)]
    public var focals: [Double]
    public var notes: [String]
}

extension StitchEngine {
    /// Aligns the main group of `report` without compositing.
    public nonisolated func align(_ report: MatchReport, straighten: Bool = true) throws -> AlignmentSummary {
        let outcome = try Self.alignment(for: report, straighten: straighten)
        let problem = outcome.problem, alignment = outcome.alignment
        let names = Dictionary(uniqueKeysWithValues: report.images.map { ($0.id, $0.name) })
        return AlignmentSummary(
            model: alignment.model, rms: alignment.rms, anchor: problem.images[alignment.anchor].name,
            pairs: zip(problem.pairs, alignment.pairRMS).map { (problem.images[$0.a].name, problem.images[$0.b].name, $1) },
            focals: alignment.focals,
            notes: outcome.notes + outcome.leftOutPairs.map { $0.text { names[$0] ?? "\($0)" } }
                + outcome.leftOutImages.map { String(format: String(localized: "%@ was left out: it is joined only through pairs that disagree with the others"),
                                                     locale: .current, names[$0] ?? "\($0)") }
                + outcome.misplacedImages.map { String(format: String(localized: "%@ was left out: its pairs place it implausibly"),
                                                       locale: .current, names[$0] ?? "\($0)") }
        )
    }

    /// Aligns the main group of `report`, without the pairs whose loops do not close and the photos they
    /// alone joined.
    static func alignment(for report: MatchReport, straighten: Bool, provisional: Bool = false) throws -> AlignmentOutcome {
        guard let component = report.graph.components.first, component.count >= 2 else {
            throw StitchError.nothingToStitch
        }
        let full = AlignmentProblem.build(report, component: component)
        guard full.pairs.count >= full.images.count - 1, full.isConnected() else {
            throw StitchError.nothingToStitch
        }
        let (problem, broken, separated) = full.consistentLoops()
        guard problem.images.count >= 2 else { throw StitchError.nothingToStitch }
        let brokenKeys = Set(broken.map { PairProposal.Key(full.images[full.pairs[$0].a].id, full.images[full.pairs[$0].b].id) })
        let verified = report.evidence(filter: .best).filter {
            $0.verdict == .verified && !brokenKeys.contains(PairProposal.Key($0.a, $0.b))
        }
        func centre(of problem: AlignmentProblem) -> Int {
            let ids = Set(problem.images.map(\.id))
            let edges = verified.filter { ids.contains($0.a) && ids.contains($0.b) }
            let centreID = GraphLayout.centre(of: problem.images.map(\.id), edges: edges)
            return problem.images.firstIndex { $0.id == centreID } ?? 0
        }
        let mode = report.configuration.mode
        let result = try Aligner.align(problem, mode: mode, centre: centre(of: problem), straighten: straighten,
                                       provisional: provisional)
        let leftOutPairs = broken.map { LeftOutPair(a: full.images[full.pairs[$0].a].id, b: full.images[full.pairs[$0].b].id,
                                                    reason: .brokenLoops) }
            + result.dropped.map { LeftOutPair(a: $0.a, b: $0.b, reason: .offBy($0.off)) }
        return AlignmentOutcome(problem: result.problem, alignment: result.alignment, notes: result.notes,
                                leftOutPairs: leftOutPairs, leftOutImages: separated.map { full.images[$0].id },
                                misplacedImages: result.misplaced)
    }
}
