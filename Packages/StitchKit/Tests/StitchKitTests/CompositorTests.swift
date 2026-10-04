import CoreGraphics
import CStitchCore
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import StitchKit

/// The native compositor driven directly, with synthetic photos whose positions are known exactly.
@Suite("Compositor")
struct CompositorTests {
    /// A scene sampled at any point, so that photos placed at fractional offsets still agree.
    typealias Scene = @Sendable (Double, Double) -> (UInt16, UInt16, UInt16)

    static let smooth: Scene = { x, y in
        func channel(_ a: Double, _ b: Double, _ c: Double) -> UInt16 {
            UInt16(32000 + 28000 * sin(a * x + 0.3) * cos(b * y + c))
        }
        return (channel(0.031, 0.017, 0.1), channel(0.013, 0.041, 0.7), channel(0.023, 0.029, 1.9))
    }

    /// Dark grey with saturated 2 x 2 spots about every 12 pixels, like highlights on a dark subject.
    static let stars: Scene = { x, y in
        let cx = Int((x / 12).rounded(.down)), cy = Int((y / 12).rounded(.down))
        var hash = UInt64(bitPattern: Int64(cx &* 73_856_093 ^ cy &* 19_349_663))
        hash = (hash ^ (hash >> 13)) &* 0x5bd1_e995
        let sx = Double(cx * 12) + Double(hash % 10), sy = Double(cy * 12) + Double((hash >> 8) % 10)
        let lit = x >= sx && x < sx + 2 && y >= sy && y < sy + 2
        let v: UInt16 = lit ? 65535 : 3000
        return (v, v, v)
    }

    struct Photo {
        var width: Int
        var height: Int
        /// Translation from the photo to the mosaic, full-resolution pixels.
        var dx: Double
        var dy: Double
    }

    struct Result {
        var width: Int
        var height: Int
        var crop: PixelRect
        var opaque: Bool
        var rgba: [UInt16]
    }

    /// RGBA16 pixels of `photo` at `size`, each pixel the scene at its centre (area-consistent sampling).
    static func pixels(_ photo: Photo, size: (width: Int, height: Int), scene: Scene) -> [UInt16] {
        var out = [UInt16](repeating: 65535, count: size.width * size.height * 4)
        let rx = Double(photo.width) / Double(size.width), ry = Double(photo.height) / Double(size.height)
        for y in 0..<size.height {
            for x in 0..<size.width {
                let fx = (Double(x) + 0.5) * rx - 0.5 + photo.dx, fy = (Double(y) + 0.5) * ry - 0.5 + photo.dy
                let (r, g, b) = scene(fx, fy)
                let i = (y * size.width + x) * 4
                out[i] = r
                out[i + 1] = g
                out[i + 2] = b
            }
        }
        return out
    }

    static func options(blend: sc_blend, seam: sc_seam, scale: Double = 1, seamScale: Double = 0.1) -> sc_compose_options {
        var options = sc_compose_options()
        options.projection = SC_PROJECTION_PLANE
        options.scale = scale
        options.seam_scale = seamScale
        options.seam = seam
        options.exposure = SC_EXPOSURE_NONE
        options.similarity = 0.15
        options.blend = blend
        options.interpolation = 2
        return options
    }

    static func images(_ photos: [Photo]) -> [sc_compose_image] {
        photos.map {
            sc_compose_image(width: Int32($0.width), height: Int32($0.height),
                             transform: (1, 0, $0.dx, 0, 1, $0.dy, 0, 0, 1), focal: 0)
        }
    }

    /// Runs the compositor through every stage; nil and the message when a stage fails.
    static func run(_ photos: [Photo], options: sc_compose_options, scene: Scene) -> (Result?, String) {
        var message = [CChar](repeating: 0, count: 512)
        var options = options
        let composeImages = images(photos)
        guard let c = composeImages.withUnsafeBufferPointer({
            sc_compositor_create($0.baseAddress, Int32($0.count), &options, &message, message.count)
        }) else { return (nil, errorText(message)) }
        defer { sc_compositor_free(c) }
        for (index, photo) in photos.enumerated() {
            var w: Int32 = 0, h: Int32 = 0
            sc_compositor_seam_size(c, Int32(index), &w, &h)
            let data = pixels(photo, size: (Int(w), Int(h)), scene: scene)
            let status = data.withUnsafeBufferPointer {
                sc_compositor_add_seam_image(c, Int32(index), $0.baseAddress, w, h, w * 8, 1, &message, message.count)
            }
            if status != 0 { return (nil, errorText(message)) }
        }
        if sc_compositor_prepare(c, nil, &message, message.count) != 0 { return (nil, errorText(message)) }
        for (index, photo) in photos.enumerated() {
            var w: Int32 = 0, h: Int32 = 0
            sc_compositor_image_size(c, Int32(index), &w, &h)
            let data = pixels(photo, size: (Int(w), Int(h)), scene: scene)
            let status = data.withUnsafeBufferPointer {
                sc_compositor_add_image(c, Int32(index), $0.baseAddress, w, h, w * 8, 1, &message, message.count)
            }
            if status != 0 { return (nil, errorText(message)) }
        }
        var panorama = sc_panorama()
        if sc_compositor_finish(c, &panorama, &message, message.count) != 0 { return (nil, errorText(message)) }
        defer { sc_panorama_free(&panorama) }
        let count = Int(panorama.width) * Int(panorama.height) * 4
        let rgba = Array(UnsafeBufferPointer(start: panorama.pixels, count: count))
        let crop = PixelRect(x: Int(panorama.crop.0), y: Int(panorama.crop.1), width: Int(panorama.crop.2),
                             height: Int(panorama.crop.3))
        return (Result(width: Int(panorama.width), height: Int(panorama.height), crop: crop, opaque: panorama.opaque != 0,
                       rgba: rgba), "")
    }

    @Test("Original values keep alpha at 0 or 65535 when photos sit at fractional offsets")
    func binaryAlpha() throws {
        let photos = (0..<4).map { (k: Int) -> Photo in
            let column = Double(k % 2), row = Double(k / 2), step = Double(k)
            return Photo(width: 1201, height: 901, dx: column * 800 + 0.37 * step, dy: row * 620 + 0.185 * step)
        }
        let (result, message) = Self.run(photos, options: Self.options(blend: SC_BLEND_NONE, seam: SC_SEAM_GRAPHCUT, seamScale: 0.0905),
                                         scene: Self.smooth)
        let panorama = try #require(result, "\(message)")
        var partial = 0
        for i in stride(from: 3, to: panorama.rgba.count, by: 4) where panorama.rgba[i] != 0 && panorama.rgba[i] != 65535 {
            partial += 1
        }
        #expect(partial == 0, "\(partial) pixels with partial alpha")
        // The four tiles cover about 2001 x 1521 pixels; the rectangle without empty corners is nearly all of it.
        let crop = panorama.crop
        #expect(crop.width > 1990 && crop.height > 1510, "crop \(crop.width) x \(crop.height)")
    }

    @Test("Multi-band blending does not wrap around where four photos meet", arguments: [SC_SEAM_VORONOI, SC_SEAM_GRAPHCUT])
    func noWrapAround(seam: sc_seam) throws {
        let photos = (0..<4).map { (k: Int) -> Photo in
            Photo(width: 1200, height: 900, dx: Double(k % 2) * 800, dy: Double(k / 2) * 620)
        }
        let (result, message) = Self.run(photos, options: Self.options(blend: SC_BLEND_MULTIBAND, seam: seam), scene: Self.stars)
        let panorama = try #require(result, "\(message)")
        #expect(panorama.width == 2000 && panorama.height == 1520)
        var wrapped = 0
        for y in 0..<panorama.height {
            for x in 0..<panorama.width {
                let i = (y * panorama.width + x) * 4
                guard panorama.rgba[i + 3] == 65535 else { continue }
                let expected = Self.stars(Double(x), Double(y)).0
                if abs(Int(panorama.rgba[i]) - Int(expected)) > 32768 { wrapped += 1 }
            }
        }
        #expect(wrapped == 0, "\(wrapped) pixels off by more than half the range")
    }

    /// Counts sc_compositor_prepare's progress reports and asks for a stop at report `stopAt`.
    private final class Reports {
        var fractions: [Double] = []
        let stopAt: Int
        init(stopAt: Int) { self.stopAt = stopAt }
    }

    @Test("Preparing reports the exposure and then the seams, and stops when asked")
    func prepareProgress() throws {
        let photos = (0..<4).map { (k: Int) -> Photo in
            Photo(width: 1200, height: 900, dx: Double(k % 2) * 800, dy: Double(k / 2) * 620)
        }
        var options = Self.options(blend: SC_BLEND_MULTIBAND, seam: SC_SEAM_GRAPHCUT)
        options.exposure = SC_EXPOSURE_CHANNELS
        func prepare(stopAt: Int) throws -> (status: Int32, fractions: [Double]) {
            var message = [CChar](repeating: 0, count: 512)
            let composeImages = Self.images(photos)
            let c = try #require(composeImages.withUnsafeBufferPointer {
                sc_compositor_create($0.baseAddress, Int32($0.count), &options, &message, message.count)
            })
            defer { sc_compositor_free(c) }
            for (index, photo) in photos.enumerated() {
                var w: Int32 = 0, h: Int32 = 0
                sc_compositor_seam_size(c, Int32(index), &w, &h)
                let data = Self.pixels(photo, size: (Int(w), Int(h)), scene: Self.stars)
                _ = data.withUnsafeBufferPointer {
                    sc_compositor_add_seam_image(c, Int32(index), $0.baseAddress, w, h, w * 8, 1, &message, message.count)
                }
            }
            let reports = Reports(stopAt: stopAt)
            let control = sc_progress(report: { context, fraction in
                let reports = Unmanaged<Reports>.fromOpaque(context!).takeUnretainedValue()
                reports.fractions.append(fraction)
                return reports.fractions.count >= reports.stopAt ? 1 : 0
            }, context: Unmanaged.passUnretained(reports).toOpaque())
            let status = withExtendedLifetime(reports) {
                withUnsafePointer(to: control) { sc_compositor_prepare(c, $0, &message, message.count) }
            }
            return (status, reports.fractions)
        }
        // The exposure, then one report before each overlapping pair of the four tiles (all six overlap), then 1.
        let whole = try prepare(stopAt: .max)
        #expect(whole.status == 0)
        #expect(whole.fractions.first == 0 && whole.fractions.last == 1 && whole.fractions.count >= 8, "\(whole.fractions)")
        #expect(zip(whole.fractions, whole.fractions.dropFirst()).allSatisfy { $0 <= $1 }, "\(whole.fractions)")
        let stopped = try prepare(stopAt: 3)
        #expect(stopped.status == 2 && stopped.fractions.count == 3, "\(stopped)")
    }

    @Test("A second finish is refused instead of reusing the released blender")
    func finishOnce() throws {
        var message = [CChar](repeating: 0, count: 512)
        let photos = [Photo(width: 400, height: 300, dx: 0, dy: 0), Photo(width: 400, height: 300, dx: 250, dy: 10)]
        var options = Self.options(blend: SC_BLEND_MULTIBAND, seam: SC_SEAM_VORONOI)
        let composeImages = Self.images(photos)
        let c = try #require(composeImages.withUnsafeBufferPointer {
            sc_compositor_create($0.baseAddress, Int32($0.count), &options, &message, message.count)
        })
        defer { sc_compositor_free(c) }
        for pass in 0..<2 {
            if pass == 1 { #expect(sc_compositor_prepare(c, nil, &message, message.count) == 0) }
            for (index, photo) in photos.enumerated() {
                var w: Int32 = 0, h: Int32 = 0
                if pass == 0 { sc_compositor_seam_size(c, Int32(index), &w, &h) } else { sc_compositor_image_size(c, Int32(index), &w, &h) }
                let data = Self.pixels(photo, size: (Int(w), Int(h)), scene: Self.smooth)
                let status = data.withUnsafeBufferPointer {
                    pass == 0 ? sc_compositor_add_seam_image(c, Int32(index), $0.baseAddress, w, h, w * 8, 1, &message, message.count)
                        : sc_compositor_add_image(c, Int32(index), $0.baseAddress, w, h, w * 8, 1, &message, message.count)
                }
                #expect(status == 0)
            }
        }
        var first = sc_panorama(), second = sc_panorama()
        #expect(sc_compositor_finish(c, &first, &message, message.count) == 0)
        sc_panorama_free(&first)
        #expect(sc_compositor_finish(c, &second, &message, message.count) == 1)
        #expect(second.pixels == nil)
    }

    @Test("A homography that sends a corner towards infinity is refused with a message")
    func farCorner() {
        var message = [CChar](repeating: 0, count: 512)
        var options = Self.options(blend: SC_BLEND_MULTIBAND, seam: SC_SEAM_VORONOI)
        // w = 1 - (1 - 1e-7) x / 999 is 1e-7 at the right edge of the second photo.
        let images = [
            sc_compose_image(width: 1000, height: 800, transform: (1, 0, 0, 0, 1, 0, 0, 0, 1), focal: 0),
            sc_compose_image(width: 1000, height: 800, transform: (1, 0, 0, 0, 1, 0, -(1 - 1e-7) / 999, 0, 1), focal: 0),
        ]
        let c = images.withUnsafeBufferPointer {
            sc_compositor_create($0.baseAddress, Int32($0.count), &options, &message, message.count)
        }
        #expect(c == nil)
        if let c { sc_compositor_free(c) }
        #expect(errorText(message).contains("too far"), "\(errorText(message))")
    }

    @Test("A canvas over the size limit can be created, so that its scale can be lowered, and is refused when prepared")
    func sizeLimit() throws {
        var message = [CChar](repeating: 0, count: 512)
        var options = Self.options(blend: SC_BLEND_MULTIBAND, seam: SC_SEAM_VORONOI)
        let side = Int(SC_MAX_PANORAMA_SIDE)
        let images = [
            sc_compose_image(width: 1000, height: 800, transform: (1, 0, 0, 0, 1, 0, 0, 0, 1), focal: 0),
            sc_compose_image(width: 1000, height: 800, transform: (1, 0, Double(side), 0, 1, 0, 0, 0, 1), focal: 0),
        ]
        let c = try #require(images.withUnsafeBufferPointer {
            sc_compositor_create($0.baseAddress, Int32($0.count), &options, &message, message.count)
        }, "\(errorText(message))")
        defer { sc_compositor_free(c) }
        var width: Int32 = 0, height: Int32 = 0
        sc_compositor_canvas_size(c, &width, &height)
        #expect(Int(width) == side + 1000 && height == 800)
        let scale = StitchEngine.outputScale(canvas: PixelSize(width: Int(width), height: Int(height)), requested: 1,
                                             bytesPerPixel: 53, memory: 32 << 30)
        #expect(scale.scale < 1 && Double(width) * scale.scale <= Double(side))
        #expect(scale.limitedBySize)
        // At the requested scale the canvas is refused once the seam copies are in.
        for (index, dx) in [0.0, Double(side)].enumerated() {
            var w: Int32 = 0, h: Int32 = 0
            sc_compositor_seam_size(c, Int32(index), &w, &h)
            let data = Self.pixels(Photo(width: 1000, height: 800, dx: dx, dy: 0), size: (Int(w), Int(h)), scene: Self.smooth)
            let status = data.withUnsafeBufferPointer {
                sc_compositor_add_seam_image(c, Int32(index), $0.baseAddress, w, h, w * 8, 1, &message, message.count)
            }
            #expect(status == 0)
        }
        #expect(sc_compositor_prepare(c, nil, &message, message.count) == 1)
        #expect(errorText(message).contains("per side"), "\(errorText(message))")
    }

    @Test("Copies of a photo are resampled with pixel centres at (x + 0.5) * r - 0.5, as the compositor places them")
    func areaMap() throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // A horizontal ramp, 16 x the column, in 16 bits and linear light so that drawing does not convert it.
        let width = 1000, height = 8
        var ramp = [UInt16](repeating: 65535, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width { for c in 0..<3 { ramp[(y * width + x) * 4 + c] = UInt16(16 * x) } }
        }
        let space = try #require(CGColorSpace(name: CGColorSpace.linearSRGB))
        let info = CGImageAlphaInfo.noneSkipLast.rawValue | CGImageByteOrderInfo.order16Little.rawValue
        let provider = try #require(CGDataProvider(data: ramp.withUnsafeBytes { Data($0) } as CFData))
        let image = try #require(CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 64, bytesPerRow: width * 8,
                                         space: space, bitmapInfo: CGBitmapInfo(rawValue: info), provider: provider, decode: nil,
                                         shouldInterpolate: true, intent: .defaultIntent))
        let url = directory.appendingPathComponent("ramp.png")
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        let photo = try ImageLoader.describe(url: url, id: 0)
        for target in [735, 333, 100] {
            let stored = try ImageLoader.stored(photo, target: PixelSize(width: target, height: 4), space: space)
            #expect(stored.width == target && stored.height == 4 && stored.pixels.count == target * 4 * 4)
            let r = Double(width) / Double(target)
            for i in [target / 4, target / 2, 3 * target / 4] {
                let sampled = Double(stored.pixels[(2 * target + i) * 4]) / 16
                #expect(abs(sampled - ((Double(i) + 0.5) * r - 0.5)) < 0.05, "target \(target), pixel \(i): \(sampled)")
            }
        }
    }

    @Test("Photos scaled down are placed by their rounded sizes, so their edges keep their full-resolution distance")
    func roundedSizes() throws {
        var message = [CChar](repeating: 0, count: 512)
        let scale = 0.7349
        var options = Self.options(blend: SC_BLEND_MULTIBAND, seam: SC_SEAM_VORONOI, scale: scale)
        // 1000 x 0.7349 rounds to 735 and 1001 x 0.7349 to 736: the second photo is drawn slightly larger.
        let photos = [Photo(width: 1000, height: 700, dx: 0, dy: 0), Photo(width: 1001, height: 700, dx: 600.25, dy: 3.5)]
        let composeImages = Self.images(photos)
        let c = try #require(composeImages.withUnsafeBufferPointer {
            sc_compositor_create($0.baseAddress, Int32($0.count), &options, &message, message.count)
        })
        defer { sc_compositor_free(c) }
        func edges(_ index: Int) -> (left: Double, right: Double) {
            var points = [Float](repeating: 0, count: 2 * 160)
            let count = Int(sc_compositor_outline(c, Int32(index), &points, 160))
            let xs = (0..<count).map { Double(points[2 * $0]) }
            return (xs.min() ?? .nan, xs.max() ?? .nan)
        }
        let a = edges(0), b = edges(1)
        // Full resolution: left edges 600.25 apart, right edges (1001 + 600.25) - 1000 apart; times the scale.
        #expect(abs((b.left - a.left) - 600.25 * scale) < 1e-3, "\(b.left - a.left)")
        #expect(abs((b.right - a.right) - 601.25 * scale) < 1e-3, "\(b.right - a.right)")
    }
}
