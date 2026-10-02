import CStitchCore
import Foundation

enum Verifier {
    /// Share of the best inlier count a simpler model must keep to be preferred.
    static let simplerModelInlierShare = 0.92

    static func verify(
        _ pair: PairEvidence, imageA: SourceImage, imageB: SourceImage, configuration: PipelineConfiguration
    ) -> PairEvidence {
        let start = ContinuousClock.now
        var result = pair
        defer { result.verificationSeconds = (ContinuousClock.now - start).seconds }

        let count = pair.matches.count
        guard count >= configuration.minimumInliers else {
            result.verdict = .tooFewMatches
            return result
        }

        // Residuals are measured in B, so the threshold follows B's resolution.
        let megapixels = Double(imageB.pixelSize.width * imageB.pixelSize.height) / 1_000_000
        let threshold = configuration.inlierThreshold * megapixels.squareRoot()
        var pointsA = [Float](repeating: 0, count: count * 2)
        var pointsB = [Float](repeating: 0, count: count * 2)
        for (index, match) in pair.matches.enumerated() {
            pointsA[2 * index] = match.a.x
            pointsA[2 * index + 1] = match.a.y
            pointsB[2 * index] = match.b.x
            pointsB[2 * index + 1] = match.b.y
        }

        // In plane mode a homography is fitted too, only to tell when the photos are not tiles of a plane.
        let candidates = configuration.mode.candidateModels
        let diagnostic: [MotionModel] = configuration.mode == .plane ? [.homography] : []
        var fits: [ModelFit] = []
        var mask = [UInt8](repeating: 0, count: count)
        for model in candidates + diagnostic {
            let fit = sc_fit_model(
                pointsA, pointsB, Int32(count), nativeModel(model), threshold, Int32(configuration.maxIterations),
                configuration.ransacConfidence, configuration.seed, &mask
            )
            guard fit.ok != 0 else { continue }
            let inliers = mask.indices.compactMap { mask[$0] != 0 ? Int32($0) : nil }
            let transform = PlaneGeometry.facingPoints(
                withUnsafeBytes(of: fit.transform) { Array($0.bindMemory(to: Double.self)) },
                inliers.map { SIMD2(Double(pointsA[2 * Int($0)]), Double(pointsA[2 * Int($0) + 1])) }
            )
            fits.append(ModelFit(model: model, transform: transform, inliers: inliers, rmse: fit.rmse,
                                 medianError: fit.median_error))
        }
        result.fits = fits

        guard let chosen = chooseModel(fits.filter { candidates.contains($0.model) }, threshold: threshold) else {
            result.verdict = .noConsistentModel
            return result
        }
        result.chosenModel = chosen.model
        let inliers = chosen.inliers.count
        let perspective = fits.first { diagnostic.contains($0.model) }.map(\.inliers.count) ?? 0

        // Brown and Lowe count only the features inside the overlap: matches elsewhere are
        // outliers by construction and would make large, repetitive scenes look unreliable.
        let overlap = PlaneGeometry.overlapInA(aToB: chosen.transform, sizeA: imageA.pixelSize,
                                               sizeB: imageB.pixelSize)
        let forward = PlaneGeometry.matrix(chosen.transform)
        result.overlapA = overlap.map { Point2(x: Float($0.x), y: Float($0.y)) }
        result.overlapB = overlap.compactMap { PlaneGeometry.apply(forward, $0.x, $0.y) }
            .map { Point2(x: Float($0.x), y: Float($0.y)) }
        let inOverlap = pair.matches.filter {
            PlaneGeometry.contains(overlap, SIMD2(Double($0.a.x), Double($0.a.y)))
        }.count
        result.matchesInOverlap = max(inOverlap, inliers)
        result.confidence = Double(inliers) / (8 + 0.3 * Double(result.matchesInOverlap))
        let overlapArea = abs(PlaneGeometry.signedArea(overlap))
        // Without its outermost layer, so a single stray inlier cannot stretch a strip into an area.
        let hull = ConvexHull.peeledArea(chosen.inliers.map { pair.matches[Int($0)].a })
        let areaA = Double(imageA.pixelSize.width * imageA.pixelSize.height)
        result.inlierCoverage = overlapArea > 0 ? min(1, hull / overlapArea) : 0
        // The threshold is in B's pixels, and so is the area a random match can land in.
        let overlapAreaB = abs(PlaneGeometry.signedArea(result.overlapB.map { SIMD2(Double($0.x), Double($0.y)) }))
        result.log10NFA = log10NumberOfFalseAlarms(
            matches: count, inliers: inliers, sampleSize: chosen.model.minimumSample, threshold: threshold,
            area: overlapAreaB > 0 ? overlapAreaB : Double(imageB.pixelSize.width * imageB.pixelSize.height)
        )

        if perspective >= configuration.minimumInliers && Double(perspective) >= 1.5 * Double(inliers) {
            // A homography explains far more matches: perspective or a rotating camera, not plane tiles.
            result.verdict = .modelMismatch
        } else if overlap.isEmpty || inliers < configuration.minimumInliers {
            result.verdict = .noConsistentModel
        } else if !isPlausible(chosen, sizeA: imageA.pixelSize, sizeB: imageB.pixelSize, mode: configuration.mode,
                               overlap: overlap) {
            result.verdict = .implausibleModel
        } else if hull < configuration.minimumHullFraction * areaA
                    || result.inlierCoverage < configuration.minimumCoverage {
            // Inliers on a thin strip or a single spot: typical of repeated labels or text.
            result.verdict = .degenerateInliers
        } else if result.log10NFA >= 0 {
            result.verdict = .lowConfidence
        } else {
            result.verdict = .verified
        }
        return result
    }

    /// A-contrario significance (Moisan, Moulon and Monasse 2012):
    /// NFA = (n - s) * C(n, k) * C(k, s) * alpha^(k - s), with alpha the probability that a random
    /// match lands within `threshold` of its prediction. The pair is meaningful when NFA < 1.
    static func log10NumberOfFalseAlarms(
        matches n: Int, inliers k: Int, sampleSize s: Int, threshold: Double, area: Double
    ) -> Double {
        guard n > s, k > s, area > 0 else { return .infinity }
        let alpha = min(1, Double.pi * threshold * threshold / area)
        func log10Choose(_ n: Int, _ k: Int) -> Double {
            (lgamma(Double(n + 1)) - lgamma(Double(k + 1)) - lgamma(Double(n - k + 1))) / log(10)
        }
        return log10(Double(n - s)) + log10Choose(n, k) + log10Choose(k, s) + Double(k - s) * log10(alpha)
    }

    /// Rejects folds everywhere, and implausible scale or shear where the subject is a plane.
    static func isPlausible(
        _ fit: ModelFit, sizeA: PixelSize, sizeB: PixelSize, mode: StitchMode, overlap: [SIMD2<Double>]
    ) -> Bool {
        let m = PlaneGeometry.matrix(fit.transform)
        // Inside the overlap z > 0, where the Jacobian's determinant is det(H) / z^3: a fold has det(H) < 0.
        guard !overlap.isEmpty, m.determinant > 0 else { return false }
        let planar = mode == .plane || mode == .document || fit.model != .homography
        guard planar else { return true }
        let center = overlap.reduce(SIMD2<Double>(0, 0), +) / Double(overlap.count)
        guard let jacobian = PlaneGeometry.jacobian(m, at: center) else { return false }
        let (large, small) = PlaneGeometry.singularValues(jacobian)
        // Scale relative to the ratio of the two resolutions, so a downscaled copy is not suspicious.
        let resolution = (Double(sizeB.width * sizeB.height) / Double(sizeA.width * sizeA.height)).squareRoot()
        let scale = (large * small).squareRoot() / resolution
        return small > 0 && large / small < 2.0 && scale > 1.0 / 3 && scale < 3
    }

    /// The simplest model whose inlier count stays close to the best one and that fits about as tightly:
    /// a translation that ignores a small rotation keeps most inliers but leaves a systematic error.
    static func chooseModel(_ fits: [ModelFit], threshold: Double) -> ModelFit? {
        guard let best = fits.max(by: { $0.inliers.count < $1.inliers.count }), !best.inliers.isEmpty else { return nil }
        let floor = Double(best.inliers.count) * simplerModelInlierShare
        let ceiling = 1.5 * best.rmse + 0.1 * threshold
        return fits.sorted { $0.model < $1.model }.first { Double($0.inliers.count) >= floor && $0.rmse <= ceiling }
    }

    static func nativeModel(_ model: MotionModel) -> sc_model {
        switch model {
        case .translation: SC_MODEL_TRANSLATION
        case .similarity: SC_MODEL_SIMILARITY
        case .affine: SC_MODEL_AFFINE
        case .homography: SC_MODEL_HOMOGRAPHY
        }
    }
}

enum ConvexHull {
    /// Area of the convex hull.
    static func area(_ points: [Point2]) -> Double {
        let hull = vertices(points)
        guard hull.count >= 3 else { return 0 }
        var twice = 0.0
        for index in hull.indices {
            let p = points[hull[index]], q = points[hull[(index + 1) % hull.count]]
            twice += Double(p.x) * Double(q.y) - Double(q.x) * Double(p.y)
        }
        return abs(twice) / 2
    }

    /// Area of the hull of the points left after removing the hull's own vertices.
    static func peeledArea(_ points: [Point2]) -> Double {
        let outer = Set(vertices(points))
        return area(points.indices.filter { !outer.contains($0) }.map { points[$0] })
    }

    /// Indices of the hull vertices in order (Andrew's monotone chain).
    static func vertices(_ points: [Point2]) -> [Int] {
        guard points.count >= 3 else { return [] }
        let order = points.indices.sorted { (points[$0].x, points[$0].y) < (points[$1].x, points[$1].y) }
        func cross(_ o: Int, _ a: Int, _ b: Int) -> Double {
            let o = points[o], a = points[a], b = points[b]
            return Double(a.x - o.x) * Double(b.y - o.y) - Double(a.y - o.y) * Double(b.x - o.x)
        }
        var lower: [Int] = []
        for p in order {
            while lower.count >= 2, cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 { lower.removeLast() }
            lower.append(p)
        }
        var upper: [Int] = []
        for p in order.reversed() {
            while upper.count >= 2, cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 { upper.removeLast() }
            upper.append(p)
        }
        let hull = Array(lower.dropLast()) + Array(upper.dropLast())
        return hull.count >= 3 ? hull : []
    }
}

