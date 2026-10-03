import CoreGraphics
import Foundation
import Testing

@testable import StitchKit

@Suite("Global alignment")
struct GlobalAlignmentTests {
    /// Row-major 3 x 3 product.
    private func multiply(_ a: [Double], _ b: [Double]) -> [Double] {
        (0..<9).map { k in (0..<3).reduce(0) { $0 + a[3 * (k / 3) + $1] * b[3 * $1 + k % 3] } }
    }

    private func apply(_ m: [Double], _ x: Double, _ y: Double) -> (Double, Double) {
        let w = m[6] * x + m[7] * y + m[8]
        return ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }

    private func tile(of scene: CGImage, size: (width: Int, height: Int), toScene: [Double]) -> CGImage {
        tile(of: scene, size: size) { apply(toScene, $0, $1) }
    }

    /// A tile whose pixel (x, y) shows the scene at `toScene`(x, y), sampled bilinearly at pixel centres.
    private func tile(of scene: CGImage, size: (width: Int, height: Int),
                      toScene: (Double, Double) -> (Double, Double)) -> CGImage {
        let sw = scene.width, sh = scene.height
        var source = [UInt8](repeating: 0, count: sw * sh * 4)
        let context = CGContext(data: &source, width: sw, height: sh, bitsPerComponent: 8, bytesPerRow: sw * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.draw(scene, in: CGRect(x: 0, y: 0, width: sw, height: sh))
        var out = [UInt8](repeating: 255, count: size.width * size.height * 4)
        source.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                for y in 0..<size.height {
                    for x in 0..<size.width {
                        let (u, v) = toScene(Double(x), Double(y))
                        let x0 = Int(u.rounded(.down)), y0 = Int(v.rounded(.down))
                        guard x0 >= 0, y0 >= 0, x0 + 1 < sw, y0 + 1 < sh else { continue }
                        let fx = u - Double(x0), fy = v - Double(y0)
                        for c in 0..<3 {
                            let p00 = Double(s[(y0 * sw + x0) * 4 + c]), p10 = Double(s[(y0 * sw + x0 + 1) * 4 + c])
                            let p01 = Double(s[((y0 + 1) * sw + x0) * 4 + c]), p11 = Double(s[((y0 + 1) * sw + x0 + 1) * 4 + c])
                            let value = (p00 * (1 - fx) + p10 * fx) * (1 - fy) + (p01 * (1 - fx) + p11 * fx) * fy
                            o[(y * size.width + x) * 4 + c] = UInt8(value.rounded())
                        }
                    }
                }
            }
        }
        let provider = CGDataProvider(data: Data(out) as CFData)!
        return CGImage(width: size.width, height: size.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size.width * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    @Test("A flat document photographed at different tilts is joined with homographies in Automatic mode")
    func tiltedDocument() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2600, height: 1900, seed: 31)
        let width = 1200, height = 900
        // Tile -> scene: a perspective about the tile centre (as from a tilted camera), then the tile's place.
        let views: [(x: Double, y: Double, px: Double, py: Double)] = [
            (150, 120, 1.6e-4, -0.8e-4), (950, 140, -1.4e-4, 1.0e-4), (180, 840, -1.0e-4, -1.5e-4), (980, 860, 1.2e-4, 1.2e-4),
        ]
        var toScene: [[Double]] = []
        var urls: [URL] = []
        for (index, view) in views.enumerated() {
            let cx = Double(width) / 2, cy = Double(height) / 2
            let centre: [Double] = [1, 0, cx, 0, 1, cy, 0, 0, 1], back: [Double] = [1, 0, -cx, 0, 1, -cy, 0, 0, 1]
            let tilt: [Double] = [1, 0, 0, 0, 1, 0, view.px, view.py, 1]
            let place: [Double] = [1, 0, view.x, 0, 1, view.y, 0, 0, 1]
            let m = multiply(place, multiply(centre, multiply(tilt, back)))
            toScene.append(m)
            let url = directory.appendingPathComponent("view\(index).png")
            try Synthetic.write(tile(of: scene, size: (width, height), toScene: m), to: url)
            urls.append(url)
        }
        var configuration = PipelineConfiguration()
        configuration.sources = [.rootSIFT]
        let report = try await StitchEngine().analyze(urls: urls, configuration: configuration)
        let (problem, alignment, _) = try StitchEngine.alignment(for: report, straighten: true)
        #expect(alignment.model == .homography, "chose \(alignment.model) at \(alignment.rms) px")
        #expect(alignment.rms < 1, "rms \(alignment.rms)")
        // Every pair placed as the truth places it, over the whole of photo a that lands in photo b.
        for pair in problem.pairs {
            let ida = problem.images[pair.a].id, idb = problem.images[pair.b].id
            let truth = multiply(AlignmentProblem.inverse(toScene[idb]), toScene[ida])
            let found = multiply(AlignmentProblem.inverse(alignment.transforms[pair.b]), alignment.transforms[pair.a])
            var worst = 0.0
            for gx in 0...10 {
                for gy in 0...10 {
                    let x = Double(width - 1) * Double(gx) / 10, y = Double(height - 1) * Double(gy) / 10
                    let (tx, ty) = apply(truth, x, y)
                    guard tx >= 0, ty >= 0, tx < Double(width), ty < Double(height) else { continue }
                    let (fx, fy) = apply(found, x, y)
                    worst = max(worst, hypot(fx - tx, fy - ty))
                }
            }
            #expect(worst < 1.5, "pair \(ida)-\(idb) off by \(worst) px")
        }
    }

    /// Four photos of a camera that turns about the vertical axis (f = 1400 px), with a band of the scene
    /// that drifts `drift` px to the right from one shot to the next, like water or a crowd. Returns the
    /// alignment and the worst error, away from the band, with which it places one photo of a pair on the other.
    private func turningCamera(seed: UInt64, drift: Double) async throws -> (Alignment, worst: Double) {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // The scene is what a reference camera sees.
        let scene = Synthetic.scene(width: 3200, height: 1600, seed: seed)
        let width = 1200, height = 900, focal = 1400.0
        let toRay: [Double] = [1 / focal, 0, -Double(width) / (2 * focal), 0, 1 / focal, -Double(height) / (2 * focal), 0, 0, 1]
        let toScenePixels: [Double] = [focal, 0, 1600, 0, focal, 800, 0, 0, 1]
        let yaws = [-24.0, -8, 8, 24]
        func turn(_ degrees: Double) -> [Double] {
            let r = degrees * .pi / 180
            return [cos(r), 0, sin(r), 0, 1, 0, -sin(r), 0, cos(r)]
        }
        let band = 1000.0...1110.0
        var urls: [URL] = []
        for (index, yaw) in yaws.enumerated() {
            let m = multiply(toScenePixels, multiply(turn(yaw), toRay))
            let image = tile(of: scene, size: (width, height)) { x, y in
                let (u, v) = apply(m, x, y)
                return band.contains(v) ? (u - drift * Double(index), v) : (u, v)
            }
            let url = directory.appendingPathComponent("turn\(index).png")
            try Synthetic.write(image, to: url)
            urls.append(url)
        }
        var configuration = PipelineConfiguration()
        configuration.sources = [.rootSIFT]
        let report = try await StitchEngine().analyze(urls: urls, configuration: configuration)
        let (problem, alignment, _) = try StitchEngine.alignment(for: report, straighten: false)
        func intrinsics(_ i: Int) -> [Double] {
            [alignment.focals[i], 0, Double(width) / 2, 0, alignment.focals[i], Double(height) / 2, 0, 0, 1]
        }
        func transposed(_ m: [Double]) -> [Double] { [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]] }
        // Photo a onto photo b as the true rotations place it, and as the cameras found do: K_b R_b^T R_a K_a^-1.
        var worst = 0.0
        for pair in problem.pairs {
            let ida = problem.images[pair.a].id, idb = problem.images[pair.b].id
            let truth = multiply(AlignmentProblem.inverse(toRay), multiply(turn(-yaws[idb]), multiply(turn(yaws[ida]), toRay)))
            let found = multiply(intrinsics(pair.b), multiply(transposed(alignment.transforms[pair.b]),
                                 multiply(alignment.transforms[pair.a], AlignmentProblem.inverse(intrinsics(pair.a)))))
            let sceneOfA = multiply(toScenePixels, multiply(turn(yaws[ida]), toRay))
            for gx in 0...12 {
                for gy in 0...9 {
                    let x = Double(width - 1) * Double(gx) / 12, y = Double(height - 1) * Double(gy) / 9
                    let (tx, ty) = apply(truth, x, y)
                    guard tx >= 0, ty >= 0, tx < Double(width), ty < Double(height), !band.contains(apply(sceneOfA, x, y).1)
                    else { continue }
                    let (fx, fy) = apply(found, x, y)
                    worst = max(worst, hypot(fx - tx, fy - ty))
                }
            }
        }
        return (alignment, worst)
    }

    @Test("A rotating camera is solved with its focal length, and a band that moves between shots does not bend it",
          arguments: [41, 44, 48] as [UInt64])
    func rotatingCamera(seed: UInt64) async throws {
        let (alignment, worst) = try await turningCamera(seed: seed, drift: 3)
        #expect(alignment.model == .rotation, "chose \(alignment.model) at \(alignment.rms) px")
        for f in alignment.focals { #expect(abs(f / 1400 - 1) < 0.01, "focal \(f)") }
        // Over eleven scenes (seeds 41-51) the worst error is 0.11-0.26 px, and 0.49-0.99 px when the refit
        // keeps every match; 44 and 48 are the hardest.
        #expect(worst < 0.4, "off by \(worst) px")
    }

    @Test("A pair made mostly of matches that moved is left out of the refit, and errors stay with their pairs")
    func rotationCorrespondences() throws {
        let width = 1200, height = 900, focal = 1400.0
        let toRay: [Double] = [1 / focal, 0, -Double(width) / (2 * focal), 0, 1 / focal, -Double(height) / (2 * focal), 0, 0, 1]
        let yaws = [-24.0, -8, 8, 24]
        func turn(_ degrees: Double) -> [Double] {
            let r = degrees * .pi / 180
            return [cos(r), 0, sin(r), 0, 1, 0, -sin(r), 0, cos(r)]
        }
        func truth(_ a: Int, _ b: Int) -> [Double] {
            multiply(AlignmentProblem.inverse(toRay), multiply(turn(-yaws[b]), multiply(turn(yaws[a]), toRay)))
        }
        var rng = SplitMix64(state: 7)
        func noise() -> Double { Double.random(in: -0.4...0.4, using: &rng) }
        // Matches of a and b with 0.4 px of noise; `moved` of them are 6 px off, as if the subject had moved.
        func pair(_ a: Int, _ b: Int, count: Int, moved: Int) -> AlignmentProblem.Pair {
            let h = truth(a, b)
            var points: [Float] = []
            while points.count < 4 * count {
                let x = Double.random(in: 0..<Double(width), using: &rng), y = Double.random(in: 0..<Double(height), using: &rng)
                let (u, v) = apply(h, x, y)
                guard u >= 0, v >= 0, u < Double(width), v < Double(height) else { continue }
                let shift = points.count < 4 * moved ? 6.0 : 0
                points += [Float(x + noise()), Float(y + noise()), Float(u + shift + noise()), Float(v + noise())]
            }
            return AlignmentProblem.Pair(a: a, b: b, points: points, sigmas: Array(repeating: 1, count: count),
                                         homography: h, siftPoints: [], siftSigmas: [])
        }
        let images = (0..<4).map {
            SourceImage(id: $0, url: URL(fileURLWithPath: "/dev/null"), name: "turn\($0)",
                        pixelSize: PixelSize(width: width, height: height), captureDate: nil, focalLength35mm: nil)
        }
        // Pair 0-3 does not overlap and carries only non-finite points: it must get an error of 0 in its own place.
        var empty = pair(0, 1, count: 8, moved: 0)
        empty.b = 3
        empty.points = Array(repeating: .nan, count: 32)
        let pairs = [pair(0, 1, count: 150, moved: 0), pair(1, 2, count: 150, moved: 0), empty, pair(1, 3, count: 40, moved: 36),
                     pair(2, 3, count: 150, moved: 0), pair(0, 2, count: 150, moved: 0)]
        let problem = AlignmentProblem(images: images, pairs: pairs, thresholds: Array(repeating: 3, count: 4))
        let alignment = try Aligner.solve(.rotation, problem, anchor: 0, straighten: false)
        #expect(alignment.pairRMS.count == pairs.count)
        #expect(alignment.pairRMS[2] == 0)
        #expect(alignment.pairRMS.indices.max { alignment.pairRMS[$0] < alignment.pairRMS[$1] } == 3, "\(alignment.pairRMS)")
        for f in alignment.focals { #expect(abs(f / focal - 1) < 0.005, "focal \(f)") }
        func intrinsics(_ i: Int) -> [Double] {
            [alignment.focals[i], 0, Double(width) / 2, 0, alignment.focals[i], Double(height) / 2, 0, 0, 1]
        }
        func transposed(_ m: [Double]) -> [Double] { [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]] }
        var worst = 0.0
        for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)] {
            let found = multiply(intrinsics(b), multiply(transposed(alignment.transforms[b]),
                                 multiply(alignment.transforms[a], AlignmentProblem.inverse(intrinsics(a)))))
            for gx in 0...12 {
                for gy in 0...9 {
                    let x = Double(width - 1) * Double(gx) / 12, y = Double(height - 1) * Double(gy) / 9
                    let (tx, ty) = apply(truth(a, b), x, y)
                    guard tx >= 0, ty >= 0, tx < Double(width), ty < Double(height) else { continue }
                    let (fx, fy) = apply(found, x, y)
                    worst = max(worst, hypot(fx - tx, fy - ty))
                }
            }
        }
        // 2.5 px when the refit keeps every match.
        #expect(worst < 0.3, "off by \(worst) px")
    }

    @Test("A homography earns its perspective by halving the planar error, without stretching any photo more than four times")
    func perspectiveRule() {
        // The insect drawer: parallax lets a chain of homographies bend the row of photos.
        #expect(!Aligner.earnsPerspective(documentRMS: 4.41, stretch: 1.95, planarRMS: 6.49))
        // The tilted page of tiltedDocument.
        #expect(Aligner.earnsPerspective(documentRMS: 0.72, stretch: 1.60, planarRMS: 21.25))
        // A box of pinned ants: little perspective and a small gain.
        #expect(!Aligner.earnsPerspective(documentRMS: 5.71, stretch: 1.08, planarRMS: 7.47))
        // Stretched too far, or not at all measurable.
        #expect(!Aligner.earnsPerspective(documentRMS: 0.5, stretch: 4.5, planarRMS: 30))
        #expect(!Aligner.earnsPerspective(documentRMS: 0.5, stretch: .infinity, planarRMS: 30))
    }

    private func photos(_ count: Int, focal35mm: [Double?]? = nil) -> [SourceImage] {
        (0..<count).map {
            SourceImage(id: $0, url: URL(fileURLWithPath: "/dev/null"), name: "photo\($0)",
                        pixelSize: PixelSize(width: 1200, height: 900), captureDate: nil, focalLength35mm: focal35mm?[$0] ?? nil)
        }
    }

    /// `count` matches of a and b inside both 1200 x 900 photos, through `h` (a to b), with noise up to `noise` px.
    private func pair(_ a: Int, _ b: Int, _ h: [Double], count: Int, noise: Double, rng: inout SplitMix64) -> AlignmentProblem.Pair {
        var points: [Float] = []
        while points.count < 4 * count {
            let x = Double.random(in: 0..<1200, using: &rng), y = Double.random(in: 0..<900, using: &rng)
            let (u, v) = apply(h, x, y)
            guard u >= 0, v >= 0, u < 1200, v < 900 else { continue }
            let du = Double.random(in: -noise...noise, using: &rng), dv = Double.random(in: -noise...noise, using: &rng)
            points += [Float(x), Float(y), Float(u + du), Float(v + dv)]
        }
        return AlignmentProblem.Pair(a: a, b: b, points: points, sigmas: Array(repeating: 1, count: count), homography: h,
                                     siftPoints: [], siftSigmas: [])
    }

    @Test("Automatic keeps the affine fit when homographies lower its error by less than half")
    func affineOverWeakPerspective() throws {
        // Four tiles in a row with a small perspective under heavy noise, the shape that parallax gives the
        // insect drawer: the homographies explain part of the error, not half of it.
        var rng = SplitMix64(state: 3)
        let p = 1e-4
        let tiles: [[Double]] = [(0.0, 1.0), (800, -1.0), (1600, 1.0), (2400, -1.0)].map { x, sign in
            multiply([1, 0, x + 600, 0, 1, 450, 0, 0, 1], multiply([1, 0, 0, 0, 1, 0, sign * p, 0.6 * p, 1], [1, 0, -600, 0, 1, -450, 0, 0, 1]))
        }
        let pairs = (0..<3).map { pair($0, $0 + 1, multiply(AlignmentProblem.inverse(tiles[$0 + 1]), tiles[$0]), count: 200, noise: 5.2, rng: &rng) }
        let problem = AlignmentProblem(images: photos(4), pairs: pairs, thresholds: Array(repeating: 10, count: 4))
        let affine = try Aligner.solve(.affine, problem, anchor: 1)
        let (document, _) = try Aligner.homographies(problem, anchor: 1)
        // The case this test is about: the affine is outside the tolerance, and the homographies do not halve it.
        try #require(affine.rms > 1.25 * document.rms + 0.5 && document.rms > 0.5 * affine.rms,
                     "affine \(affine.rms), homographies \(document.rms)")
        let (alignment, _, notes) = try Aligner.align(problem, mode: .auto, centre: 1, straighten: true)
        #expect(alignment.model == .affine, "chose \(alignment.model)")
        #expect(notes.isEmpty, "\(notes)")
    }

    @Test("With no usable homography and no rotation, Automatic joins the photos with an affine fit and says it fits badly")
    func affineAsLastResort() throws {
        // A camera turning by 100 degrees: the homographies fit, but stretch the outer photos more than four
        // times. The EXIF focal lengths rule out the rotation: they disagree by 20%, and the matches point to
        // a focal length less than half as long.
        var rng = SplitMix64(state: 5)
        let focal = 1400.0
        let toRay: [Double] = [1 / focal, 0, -600 / focal, 0, 1 / focal, -450 / focal, 0, 0, 1]
        func turn(_ degrees: Double) -> [Double] {
            let r = degrees * .pi / 180
            return [cos(r), 0, sin(r), 0, 1, 0, -sin(r), 0, cos(r)]
        }
        let yaws = [-50.0, -25, 0, 25, 50]
        let pairs = (0..<4).map { a in
            let h = multiply(AlignmentProblem.inverse(toRay), multiply(turn(-yaws[a + 1]), multiply(turn(yaws[a]), toRay)))
            return pair(a, a + 1, h, count: 150, noise: 0.3, rng: &rng)
        }
        let problem = AlignmentProblem(images: photos(5, focal35mm: [100, 120, 100, 120, 100]), pairs: pairs,
                                       thresholds: Array(repeating: 3, count: 5))
        #expect((try? Aligner.solve(.rotation, problem, anchor: 2)) == nil)
        let (document, stretch) = try Aligner.homographies(problem, anchor: 2)
        #expect(document.rms < 1 && stretch.isFinite && stretch > Aligner.overstretch, "\(document.rms) px, stretch \(stretch)")
        let (alignment, _, notes) = try Aligner.align(problem, mode: .auto, centre: 2, straighten: true)
        #expect(alignment.model == .affine, "chose \(alignment.model)")
        #expect(notes.contains { $0.hasPrefix("No model fits these photos closely") }, "\(notes)")
    }

    /// Tiles of a 2 x 2 grid, 800 px apart across and 600 px down, joined by their four sides and one diagonal.
    /// Pair 0-1 is replaced by a false link turned by `degrees` and scaled by `scale`, like matches between two
    /// rows of identical labels.
    private func gridWithFalseLink(degrees: Double, scale: Double) -> AlignmentProblem {
        var rng = SplitMix64(state: 9)
        let origins: [(Double, Double)] = [(0, 0), (800, 0), (0, 600), (800, 600)]
        func toB(_ a: Int, _ b: Int) -> [Double] {
            [1, 0, origins[a].0 - origins[b].0, 0, 1, origins[a].1 - origins[b].1, 0, 0, 1]
        }
        let r = degrees * .pi / 180
        let wrong = multiply([scale * cos(r), -scale * sin(r), 0, scale * sin(r), scale * cos(r), 0, 0, 0, 1], toB(0, 1))
        let pairs = [pair(0, 1, wrong, count: 120, noise: 0.3, rng: &rng)]
            + [(0, 2), (1, 3), (2, 3), (0, 3)].map { pair($0.0, $0.1, toB($0.0, $0.1), count: 120, noise: 0.3, rng: &rng) }
        return AlignmentProblem(images: photos(4), pairs: pairs, thresholds: Array(repeating: 3, count: 4))
    }

    @Test("A false link is left out and the model is chosen again without it")
    func chooseAgainAfterDrop() throws {
        let problem = gridWithFalseLink(degrees: 4, scale: 1.04)
        // With the false link a translation does not fit, and a model that bends to meet it is chosen.
        let first = try Aligner.align(AlignmentProblem(images: problem.images, pairs: problem.pairs, thresholds: [1e9, 1e9, 1e9, 1e9]),
                                      mode: .auto, centre: 0, straighten: true)
        try #require(first.0.model != .translation, "chose \(first.0.model) with the false link")
        let (alignment, kept, notes) = try Aligner.align(problem, mode: .auto, centre: 0, straighten: true)
        #expect(kept.pairs.count == 4 && !kept.pairs.contains { $0.a == 0 && $0.b == 1 })
        #expect(alignment.model == .translation, "chose \(alignment.model) at \(alignment.rms) px")
        #expect(alignment.rms < 0.5)
        #expect(notes.count == 1 && notes[0].hasPrefix("The pair photo0 ↔ photo1 was left out"), "\(notes)")
    }

    @Test("Plane mode suggests Document mode only when homographies halve the planar error")
    func planeModePerspectiveNote() throws {
        var rng = SplitMix64(state: 11)
        func chain(perspective p: Double, noise: Double) -> AlignmentProblem {
            let tiles: [[Double]] = [(0.0, 1.0), (800, -1.0), (1600, 1.0), (2400, -1.0)].map { x, sign in
                multiply([1, 0, x + 600, 0, 1, 450, 0, 0, 1], multiply([1, 0, 0, 0, 1, 0, sign * p, 0.6 * p, 1], [1, 0, -600, 0, 1, -450, 0, 0, 1]))
            }
            let pairs = (0..<3).map { pair($0, $0 + 1, multiply(AlignmentProblem.inverse(tiles[$0 + 1]), tiles[$0]), count: 200, noise: noise, rng: &rng) }
            return AlignmentProblem(images: photos(4), pairs: pairs, thresholds: Array(repeating: 10, count: 4))
        }
        // A strong perspective under little noise: the homographies leave the planar fits far behind.
        let tilted = try Aligner.align(chain(perspective: 1.5e-4, noise: 0.3), mode: .plane, centre: 1, straighten: true)
        #expect(tilted.notes.contains { $0.hasPrefix("The photos show perspective") }, "\(tilted.notes)")
        // A small perspective under heavy noise: they do better, not twice as well.
        let weak = try Aligner.align(chain(perspective: 1e-4, noise: 5.2), mode: .plane, centre: 1, straighten: true)
        #expect(!weak.notes.contains { $0.hasPrefix("The photos show perspective") }, "\(weak.notes)")
    }
}
