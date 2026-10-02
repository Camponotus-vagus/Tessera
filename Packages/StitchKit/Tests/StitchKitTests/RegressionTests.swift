import CoreGraphics
import CStitchCore
import Foundation
import simd
import Testing

@testable import StitchKit

/// Cases found by the bug hunt before the first release; each test names the failure it guards against.
@Suite("Regressions")
struct RegressionTests {
    // MARK: Geometry

    /// Camera rotation about the vertical axis, H = K R K^-1 normalised as OpenCV does (h33 = 1).
    private func rotation(degrees: Double, focal: Double, size: PixelSize) -> [Double] {
        let k = simd_double3x3(rows: [SIMD3(focal, 0, Double(size.width) / 2), SIMD3(0, focal, Double(size.height) / 2),
                                      SIMD3(0, 0, 1)])
        let t = degrees * .pi / 180
        let r = simd_double3x3(rows: [SIMD3(cos(t), 0, -sin(t)), SIMD3(0, 1, 0), SIMD3(sin(t), 0, cos(t))])
        let h = k * r * k.inverse
        let rows = [h[0, 0], h[1, 0], h[2, 0], h[0, 1], h[1, 1], h[2, 1], h[0, 2], h[1, 2], h[2, 2]]
        return rows.map { $0 / rows[8] }
    }

    @Test("Overlap stays correct when a corner of A maps behind the camera")
    func overlapBehindCamera() {
        let size = PixelSize(width: 4000, height: 3000)
        // Past about 36 degrees h33 is negative, so normalising it to 1 also flips the sign; the verifier
        // restores it from the inliers, which lie in front of both cameras.
        let normalised = rotation(degrees: 40, focal: 1456, size: size)
        #expect(normalised[6] * 2000 + normalised[8] < 0)
        let h = PlaneGeometry.facingPoints(normalised, [SIMD2(2000, 1500), SIMD2(3500, 1500), SIMD2(3000, 500)])
        let overlap = PlaneGeometry.overlapInA(aToB: h, sizeA: size, sizeB: size)
        #expect(overlap.count >= 3)
        #expect(PlaneGeometry.signedArea(overlap) > 0.05 * Double(size.width * size.height))
        let m = PlaneGeometry.matrix(h)
        for vertex in overlap {
            let p = PlaneGeometry.apply(m, vertex.x, vertex.y)
            #expect(p != nil)
            if let p {
                #expect(p.x > -1 && p.x < Double(size.width) + 1 && p.y > -1 && p.y < Double(size.height) + 1)
            }
        }
        let fit = ModelFit(model: .homography, transform: h, inliers: [], rmse: 1, medianError: 1)
        #expect(Verifier.isPlausible(fit, sizeA: size, sizeB: size, mode: .rotation, overlap: overlap))
    }

    @Test("The sign of a homography is chosen so that its inliers are in front of the camera")
    func homographySign() {
        let size = PixelSize(width: 4000, height: 3000)
        let h = rotation(degrees: 30, focal: 1456, size: size)
        let points = [SIMD2(3000.0, 1500.0), SIMD2(3500, 1000), SIMD2(3800, 2500)]
        #expect(PlaneGeometry.facingPoints(h.map { -$0 }, points) == h)
        #expect(PlaneGeometry.facingPoints(h, points) == h)
    }

    @Test("One stray inlier does not turn a strip into an area")
    func peeledHull() {
        var points = (0..<50).map { Point2(x: Float($0) * 20, y: Float($0 % 5)) }
        points.append(Point2(x: 500, y: 800))
        #expect(ConvexHull.area(points) > 300_000)
        #expect(ConvexHull.peeledArea(points) < 5_000)
    }

    @Test("A degenerate polygon contains nothing")
    func degenerateContains() {
        let point = SIMD2(5.0, 5.0)
        #expect(!PlaneGeometry.contains([point, point, point], point))
    }

    // MARK: Verification

    /// Matches through `map` with uniform noise of `noise` pixels in B, plus 15% random outliers.
    private func matches(
        _ count: Int, sizeA: PixelSize, sizeB: PixelSize, noise: Float, seed: UInt64,
        map: (SIMD2<Double>) -> SIMD2<Double>?
    ) -> [TentativeMatch] {
        var rng = SplitMix64(state: seed)
        var result: [TentativeMatch] = []
        while result.count < count {
            let a = SIMD2(Double.random(in: 0..<Double(sizeA.width), using: &rng),
                          Double.random(in: 0..<Double(sizeA.height), using: &rng))
            guard var b = map(a), b.x >= 0, b.y >= 0, b.x < Double(sizeB.width), b.y < Double(sizeB.height) else { continue }
            if result.count % 7 == 0 {
                b = SIMD2(Double.random(in: 0..<Double(sizeB.width), using: &rng),
                          Double.random(in: 0..<Double(sizeB.height), using: &rng))
            }
            result.append(TentativeMatch(
                a: Point2(x: Float(a.x), y: Float(a.y)),
                b: Point2(x: Float(b.x) + Float.random(in: -noise...noise, using: &rng),
                          y: Float(b.y) + Float.random(in: -noise...noise, using: &rng)),
                score: 1
            ))
        }
        return result
    }

    private func image(_ id: Int, _ size: PixelSize, name: String? = nil, date: Date? = nil) -> SourceImage {
        SourceImage(id: id, url: URL(fileURLWithPath: "/tmp/\(id).jpg"), name: name ?? "\(id).jpg", pixelSize: size,
                    captureDate: date, focalLength35mm: nil)
    }

    private func evidence(_ a: Int, _ b: Int, _ matches: [TentativeMatch]) -> PairEvidence {
        PairEvidence(a: a, b: b, source: .rootSIFT, matches: matches, fits: [], chosenModel: nil, inlierCoverage: 0,
                     confidence: 0, verdict: .tooFewMatches, matchingSeconds: 0, verificationSeconds: 0)
    }

    @Test("A pair of photos at different resolutions gets the same verdict in both orders")
    func orderIndependence() {
        let large = PixelSize(width: 4000, height: 3000), small = PixelSize(width: 1000, height: 750)
        let forward = matches(400, sizeA: large, sizeB: small, noise: 0.5, seed: 3) { 0.25 * ($0 - SIMD2(800, 0)) }
        let backward = forward.map { TentativeMatch(a: $0.b, b: $0.a, score: $0.score) }
        var configuration = PipelineConfiguration()
        configuration.mode = .auto
        let one = Verifier.verify(evidence(0, 1, forward), imageA: image(0, large), imageB: image(1, small),
                                  configuration: configuration)
        let two = Verifier.verify(evidence(0, 1, backward), imageA: image(0, small), imageB: image(1, large),
                                  configuration: configuration)
        #expect(one.verdict == .verified)
        #expect(two.verdict == .verified)
        #expect(abs(one.inlierCount - two.inlierCount) < 30)
    }

    @Test("Plane mode reports a perspective pair instead of verifying it with the wrong model")
    func planeModeMismatch() {
        let size = PixelSize(width: 2000, height: 1500)
        let h = PlaneGeometry.matrix([1, 0, -500, 0, 1, 0, 0.00025, 0, 1])
        let pairs = matches(400, sizeA: size, sizeB: size, noise: 0.5, seed: 5) { PlaneGeometry.apply(h, $0.x, $0.y) }
        var configuration = PipelineConfiguration()
        configuration.mode = .plane
        let plane = Verifier.verify(evidence(0, 1, pairs), imageA: image(0, size), imageB: image(1, size),
                                    configuration: configuration)
        #expect(plane.verdict == .modelMismatch)
        configuration.mode = .document
        let document = Verifier.verify(evidence(0, 1, pairs), imageA: image(0, size), imageB: image(1, size),
                                       configuration: configuration)
        #expect(document.verdict == .verified)
        #expect(document.chosenModel == .homography)
    }

    @Test("A translation does not win over a similarity that fits a real rotation much better")
    func modelChoiceUsesResiduals() {
        let translation = ModelFit(model: .translation, transform: [], inliers: Array(0..<95), rmse: 4, medianError: 4)
        let similarity = ModelFit(model: .similarity, transform: [], inliers: Array(0..<100), rmse: 0.8,
                                  medianError: 0.7)
        #expect(Verifier.chooseModel([translation, similarity], threshold: 10)?.model == .similarity)
        let close = ModelFit(model: .translation, transform: [], inliers: Array(0..<95), rmse: 1, medianError: 1)
        #expect(Verifier.chooseModel([close, similarity], threshold: 10)?.model == .translation)
    }

    // MARK: Graph

    private func verified(_ a: Int, _ b: Int, inliers: Int, transform: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1],
                          model: MotionModel = .translation, overlap: [Point2] = []) -> PairEvidence {
        var pair = evidence(a, b, [])
        pair.fits = [ModelFit(model: model, transform: transform, inliers: (0..<inliers).map(Int32.init), rmse: 1,
                              medianError: 1)]
        pair.chosenModel = model
        pair.verdict = .verified
        pair.overlapA = overlap
        return pair
    }

    @Test("Equal groups are ordered the same way whatever the order of the pairs")
    func deterministicComponents() {
        let size = PixelSize(width: 100, height: 100)
        let images = (0..<4).map { image($0, size) }
        let pairs = [verified(2, 3, inliers: 100), verified(0, 1, inliers: 100)]
        for order in [pairs, pairs.reversed()] {
            let graph = MatchGraphBuilder.build(images: images, features: [], pairs: order, excluded: [],
                                                configuration: PipelineConfiguration())
            #expect(graph.components == [[0, 1], [2, 3]])
        }
    }

    @Test("With nothing verified no photo is shown as joined")
    func singletonMainGroup() {
        let size = PixelSize(width: 100, height: 100)
        var pair = evidence(0, 1, [])
        pair.verdict = .noConsistentModel
        let graph = MatchGraphBuilder.build(images: [image(0, size), image(1, size)], features: [], pairs: [pair],
                                            excluded: [], configuration: PipelineConfiguration())
        #expect(graph.nodes.allSatisfy { $0.exclusion != nil })
    }

    @Test("Shooting order is by name when some photos have no date")
    func shootingOrderWithMissingDates() {
        let size = PixelSize(width: 100, height: 100)
        let images = [image(0, size, name: "b.jpg", date: Date(timeIntervalSince1970: 10)),
                      image(1, size, name: "a.jpg"),
                      image(2, size, name: "c.jpg", date: Date(timeIntervalSince1970: 5))]
        #expect(PairProposal.shootingOrder(images).map(\.id) == [1, 0, 2])
        #expect(PairProposal.shootingOrder(images.reversed()).map(\.id) == [1, 0, 2])
    }

    @Test("Bridge rounds stay within their budget and skip pairs without affinity")
    func boundedBridges() {
        let groups = (0..<30).map { Set([$0]) }
        var affinity: [PairProposal.Key: Int] = [:]
        for a in 0..<30 { for b in (a + 1)..<30 where (a + b) % 3 != 0 { affinity[PairProposal.Key(a, b)] = a + b } }
        let bridges = PairProposal.bridges(groups: groups, affinity: affinity, tried: [], perGroupPair: 2, limit: 30)
        #expect(bridges.count == 30)
        #expect(bridges.allSatisfy { (affinity[PairProposal.Key($0.a, $0.b)] ?? 0) > 0 })
        #expect(PairProposal.bridges(groups: groups, affinity: [:], tried: [], perGroupPair: 2, limit: 30).isEmpty)
    }

    @Test("Layout of a long rotation chain keeps every footprint, in order and bounded")
    func rotationChainLayout() {
        let size = PixelSize(width: 2000, height: 1500)
        let count = 8
        let h = rotation(degrees: 25, focal: 1456, size: size)
        let overlap = PlaneGeometry.overlapInA(aToB: h, sizeA: size, sizeB: size).map { Point2(x: Float($0.x), y: Float($0.y)) }
        let pairs = (0..<(count - 1)).map { verified($0, $0 + 1, inliers: 80, transform: h, model: .homography, overlap: overlap) }
        let images = (0..<count).map { image($0, size) }
        let graph = MatchGraphBuilder.build(images: images, features: [], pairs: pairs, excluded: [],
                                            configuration: PipelineConfiguration())
        let report = MatchReport(createdAt: Date(), configuration: PipelineConfiguration(), images: images, features: [],
                                 pairs: pairs, graph: graph, excludedByUser: [], candidates: nil)
        let placement = GraphLayout.geometric(report, filter: .best)
        #expect(placement.outlines.count == count)
        #expect(placement.outlines.values.allSatisfy { $0.count == 4 })
        let xs = (0..<count).compactMap { placement.centers[$0]?.x }
        #expect(xs.count == count)
        #expect(zip(xs, xs.dropFirst()).allSatisfy { $0 < $1 })
        let extent = placement.outlines.values.flatMap { $0 }.map(\.x)
        #expect((extent.max()! - extent.min()!) < Double(count * size.width))
    }

    // MARK: Native layer

    private func values<T>(_ tuple: T) -> [Double] {
        withUnsafeBytes(of: tuple) { Array($0.bindMemory(to: Double.self)) }
    }

    @Test("Model fitting ignores non-finite points and never returns a non-finite transform")
    func finiteFits() {
        var a: [Float] = [], b: [Float] = []
        for i in 0..<40 {
            let x = Float(i * 37 % 500), y = Float(i * 53 % 400)
            a += [x, y]
            b += [x + 12, y - 7]
        }
        a += [.nan, 3, 4, .infinity]
        b += [1, 2, 3, 4]
        var mask = [UInt8](repeating: 0, count: 42)
        let fit = sc_fit_model(a, b, 42, SC_MODEL_SIMILARITY, 2, 1000, 0.999, 1, &mask)
        #expect(fit.ok == 1)
        #expect(mask[40] == 0 && mask[41] == 0)
        let finite = values(fit.transform).allSatisfy { $0.isFinite }
        #expect(finite)

        let same = [Float](repeating: 10, count: 40)
        var small = [UInt8](repeating: 0, count: 20)
        let degenerate = sc_fit_model(same, same, 20, SC_MODEL_SIMILARITY, 2, 1000, 0.999, 1, &small)
        let degenerateFinite = values(degenerate.transform).allSatisfy { $0.isFinite }
        #expect(degenerate.ok == 0 || degenerateFinite)
    }

    @Test("A damaged descriptor head file is rejected with a message")
    func damagedDescriptorHead() throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        func header(_ sizes: [Int32]) -> Data {
            var data = Data("ALKH".utf8)
            for var value in sizes { data.append(Data(bytes: &value, count: 4)) }
            return data
        }
        let cases = [header([100_000, 16, 3, 128]), header([128, 16, 3, 128]) + Data(count: 100), Data("ALK".utf8)]
        for (index, data) in cases.enumerated() {
            let url = directory.appendingPathComponent("head\(index).bin")
            try data.write(to: url)
            var message = [CChar](repeating: 0, count: 256)
            let head = sc_descriptor_head_load(url.path, &message, message.count)
            #expect(head == nil)
            #expect(!errorText(message).isEmpty)
            sc_descriptor_head_free(head)
        }
    }

    // MARK: Engine

    private func tiles(in directory: URL) throws -> [URL] {
        let scene = Synthetic.scene(width: 2400, height: 1000, seed: 21)
        return try (0..<2).map { index in
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: CGPoint(x: index * 600, y: 50),
                                               size: CGSize(width: 1200, height: 900)), to: url)
            return url
        }
    }

    @Test("Unreadable files are reported, and the other photos are still analysed")
    func unreadablePhotos() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let good = try tiles(in: directory)
        let garbage = directory.appendingPathComponent("garbage.jpg")
        try Data((0..<4096).map { UInt8($0 % 251) }).write(to: garbage)
        let missing = directory.appendingPathComponent("missing.jpg")

        let report = try await StitchEngine().analyze(urls: [good[0], garbage, good[1], missing],
                                                      configuration: PipelineConfiguration())
        #expect(report.graph.nodes[1].exclusion == .unreadable)
        #expect(report.graph.nodes[3].exclusion == .unreadable)
        #expect(report.graph.components.first == [0, 2])
    }

    @Test("An analysis without any matcher is refused")
    func noMatcher() async throws {
        var configuration = PipelineConfiguration()
        configuration.sources = []
        await #expect(throws: StitchError.self) {
            _ = try await StitchEngine().analyze(urls: [], configuration: configuration)
        }
    }

    @Test("A cancelled analysis stops instead of returning a report")
    func cancellation() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let urls = try tiles(in: directory)
        let engine = StitchEngine()
        let task = Task { try await engine.analyze(urls: urls, configuration: PipelineConfiguration()) }
        task.cancel()
        await #expect(throws: CancellationError.self) { _ = try await task.value }
    }
}
