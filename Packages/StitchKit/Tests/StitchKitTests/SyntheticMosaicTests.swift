import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import StitchKit

/// Deterministic generator so the synthetic scene is identical on every run.
struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

enum Synthetic {
    /// A cluttered scene of random shapes: plenty of corners and blobs at several scales.
    static func scene(width: Int, height: Int, seed: UInt64) -> CGImage {
        var rng = SplitMix64(state: seed)
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.setFillColor(CGColor(gray: 0.5, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        for _ in 0..<2500 {
            let size = Double.random(in: 6...90, using: &rng)
            let rect = CGRect(x: Double.random(in: 0...Double(width), using: &rng),
                              y: Double.random(in: 0...Double(height), using: &rng), width: size,
                              height: size * Double.random(in: 0.4...1.6, using: &rng))
            context.setFillColor(CGColor(red: .random(in: 0...1, using: &rng), green: .random(in: 0...1, using: &rng),
                                         blue: .random(in: 0...1, using: &rng), alpha: 1))
            if Bool.random(using: &rng) {
                context.fillEllipse(in: rect)
            } else {
                context.fill(rect)
            }
        }
        return context.makeImage()!
    }

    /// Crops `size` at `origin` (top-left, y down), optionally rotated by `degrees` about the tile centre.
    static func tile(of scene: CGImage, origin: CGPoint, size: CGSize, degrees: Double = 0) -> CGImage {
        let context = CGContext(
            data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        context.interpolationQuality = .high
        // Work in a y-down frame matching image pixel coordinates.
        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1, y: -1)
        context.translateBy(x: size.width / 2, y: size.height / 2)
        context.rotate(by: degrees * .pi / 180)
        context.translateBy(x: -size.width / 2 - origin.x, y: -size.height / 2 - origin.y)
        // Draw the scene upright in the flipped frame.
        context.saveGState()
        context.translateBy(x: 0, y: CGFloat(scene.height))
        context.scaleBy(x: 1, y: -1)
        context.draw(scene, in: CGRect(x: 0, y: 0, width: scene.width, height: scene.height))
        context.restoreGState()
        return context.makeImage()!
    }

    static func write(_ image: CGImage, to url: URL) throws {
        let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw StitchError.engine("write failed") }
    }

    static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("stitchkit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

@Suite("Synthetic mosaics")
struct SyntheticMosaicTests {
    @Test("Translation between plane tiles is recovered within half a pixel")
    func translationTiles() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2600, height: 1800, seed: 7)
        let size = CGSize(width: 1200, height: 900)
        let origins = [CGPoint(x: 0, y: 0), CGPoint(x: 820, y: 30), CGPoint(x: 850, y: 640), CGPoint(x: 60, y: 700)]
        var urls: [URL] = []
        for (index, origin) in origins.enumerated() {
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: origin, size: size), to: url)
            urls.append(url)
        }

        var configuration = PipelineConfiguration()
        configuration.mode = .plane
        let report = try await StitchEngine().analyze(urls: urls, configuration: configuration)

        #expect(report.graph.components.first?.count == origins.count)
        for (a, b) in [(0, 1), (0, 3), (1, 2)] {
            let pair = try #require(report.evidence(a, b, source: .rootSIFT))
            #expect(pair.verdict == .verified)
            let fit = try #require(pair.chosenFit)
            // A point at p in tile a sits at p + origin_a - origin_b in tile b.
            let expectedX = Double(origins[a].x - origins[b].x)
            let expectedY = Double(origins[a].y - origins[b].y)
            #expect(abs(fit.transform[2] - expectedX) < 0.5, "pair \(a)-\(b) tx \(fit.transform[2]) vs \(expectedX)")
            #expect(abs(fit.transform[5] - expectedY) < 0.5, "pair \(a)-\(b) ty \(fit.transform[5]) vs \(expectedY)")
            #expect(fit.model == .translation)
        }
    }

    @Test("A rotated tile is fitted by a similarity with the right angle")
    func rotatedTile() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2600, height: 1800, seed: 11)
        let size = CGSize(width: 1200, height: 900)
        let a = directory.appendingPathComponent("a.png")
        let b = directory.appendingPathComponent("b.png")
        try Synthetic.write(Synthetic.tile(of: scene, origin: CGPoint(x: 300, y: 300), size: size), to: a)
        try Synthetic.write(Synthetic.tile(of: scene, origin: CGPoint(x: 700, y: 450), size: size, degrees: 12), to: b)

        var configuration = PipelineConfiguration()
        configuration.mode = .plane
        let report = try await StitchEngine().analyze(urls: [a, b], configuration: configuration)
        let pair = try #require(report.evidence(0, 1, source: .rootSIFT))
        #expect(pair.verdict == .verified)
        let fit = try #require(pair.chosenFit)
        #expect(fit.model == .similarity)
        let angle = atan2(fit.transform[3], fit.transform[0]) * 180 / .pi
        // The tile content is rotated by +12 degrees, so A maps onto B with -12 degrees in a y-down frame.
        #expect(abs(abs(angle) - 12) < 0.2, "angle \(angle)")
        let scale = (fit.transform[0] * fit.transform[0] + fit.transform[3] * fit.transform[3]).squareRoot()
        #expect(abs(scale - 1) < 0.005)
    }

    @Test("An unrelated photo ends up outside the main group with a reason")
    func unrelatedImageIsExcluded() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2600, height: 1800, seed: 3)
        let other = Synthetic.scene(width: 1200, height: 900, seed: 99)
        let size = CGSize(width: 1200, height: 900)
        var urls: [URL] = []
        for (index, origin) in [CGPoint(x: 0, y: 0), CGPoint(x: 800, y: 0)].enumerated() {
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: origin, size: size), to: url)
            urls.append(url)
        }
        let stranger = directory.appendingPathComponent("stranger.png")
        try Synthetic.write(other, to: stranger)
        urls.append(stranger)

        let report = try await StitchEngine().analyze(urls: urls, configuration: PipelineConfiguration())
        #expect(report.graph.components.first == [0, 1])
        let node = try #require(report.graph.nodes.first { $0.id == 2 })
        #expect(node.exclusion != nil)
        #expect(node.component != 0)
    }

    @Test("Excluded photos are reported as such and skipped")
    func userExclusion() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2600, height: 1800, seed: 5)
        let size = CGSize(width: 1200, height: 900)
        var urls: [URL] = []
        for (index, origin) in [CGPoint(x: 0, y: 0), CGPoint(x: 800, y: 0), CGPoint(x: 1300, y: 600)].enumerated() {
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: origin, size: size), to: url)
            urls.append(url)
        }
        let report = try await StitchEngine().analyze(urls: urls, configuration: PipelineConfiguration(), excluded: [1])
        #expect(report.graph.nodes.first { $0.id == 1 }?.exclusion == .excludedByUser)
        #expect(report.pairs.allSatisfy { $0.a != 1 && $0.b != 1 })
        #expect(report.excludedByUser == [1])
    }

    @Test("The JSON report survives a round trip")
    func reportRoundTrip() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2000, height: 1200, seed: 21)
        let size = CGSize(width: 1000, height: 800)
        var urls: [URL] = []
        for (index, origin) in [CGPoint(x: 0, y: 0), CGPoint(x: 700, y: 200)].enumerated() {
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: origin, size: size), to: url)
            urls.append(url)
        }
        let report = try await StitchEngine().analyze(urls: urls, configuration: PipelineConfiguration())
        let decoded = try MatchReport.decode(report.jsonData())
        #expect(decoded.version == MatchReport.currentVersion)
        #expect(decoded.pairs.count == report.pairs.count)
        #expect(decoded.pairs.first?.matches == report.pairs.first?.matches)
        #expect(decoded.pairs.first?.chosenFit?.inliers == report.pairs.first?.chosenFit?.inliers)
        #expect(decoded.graph.components == report.graph.components)
    }
}

@Suite("Verification helpers")
struct VerificationHelperTests {
    @Test("Many inliers in a large area are significant, a handful are not")
    func numberOfFalseAlarms() {
        let strong = Verifier.log10NumberOfFalseAlarms(matches: 600, inliers: 150, sampleSize: 4, threshold: 10,
                                                       area: 4_000_000)
        let weak = Verifier.log10NumberOfFalseAlarms(matches: 600, inliers: 6, sampleSize: 4, threshold: 10,
                                                     area: 4_000_000)
        #expect(strong < -100)
        #expect(weak > 0)
    }

    @Test("Overlap of a pure translation is the shared rectangle")
    func overlapPolygon() {
        let size = PixelSize(width: 1000, height: 800)
        let translation: [Double] = [1, 0, -600, 0, 1, -100, 0, 0, 1]
        let polygon = PlaneGeometry.overlapInA(aToB: translation, sizeA: size, sizeB: size)
        #expect(abs(abs(PlaneGeometry.signedArea(polygon)) - 400 * 700) < 1)
    }

    @Test("Convex hull area of a square")
    func hull() {
        let points = [Point2(x: 0, y: 0), Point2(x: 10, y: 0), Point2(x: 10, y: 10), Point2(x: 0, y: 10),
                      Point2(x: 5, y: 5)]
        #expect(abs(ConvexHull.area(points) - 100) < 1e-9)
    }
}
