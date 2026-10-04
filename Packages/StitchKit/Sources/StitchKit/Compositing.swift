import Accelerate
import CoreGraphics
import CStitchCore
import Foundation
import ImageIO
import simd

/// A photo decoded as stored in its file, before its EXIF orientation is applied.
struct StoredPixels: Sendable {
    var width: Int
    var height: Int
    var orientation: Int32
    /// RGBX, 16 bits per channel, little-endian.
    var pixels: [UInt16]
    /// The file stores more than 8 bits per component.
    var highBitDepth = false
    var bytesPerRow: Int { width * 8 }
}

extension ImageLoader {
    static func orientation(of source: CGImageSource) -> Int32 {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let value = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.int32Value ?? 1
        return (1...8).contains(value) ? value : 1
    }

    /// The full image decoded by ImageIO (not the thumbnail path, which decodes JPEG slightly differently),
    /// at a size that becomes `target` once oriented. 8-bit sources come out as 257 x v. The photo is drawn
    /// at its own size and resampled with vImage, whose pixel centres map as (x + 0.5) * r - 0.5, the map the
    /// compositor places copies by; CoreGraphics would map corner pixels to corner pixels instead.
    static func stored(_ image: SourceImage, target: PixelSize, space: CGColorSpace) throws -> StoredPixels {
        guard let source = CGImageSourceCreateWithURL(image.url as CFURL, nil) else {
            throw StitchError.cannotOpen(image.url)
        }
        let orientation = orientation(of: source)
        let swapped = (5...8).contains(orientation)
        guard let decoded = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw StitchError.cannotDecode(image.url)
        }
        let oriented = swapped ? PixelSize(width: decoded.height, height: decoded.width)
            : PixelSize(width: decoded.width, height: decoded.height)
        guard oriented == image.pixelSize else { throw StitchError.photoChanged(image.url) }
        let width = swapped ? target.height : target.width, height = swapped ? target.width : target.height
        let full = try draw(decoded, space: space, url: image.url)
        if width == decoded.width && height == decoded.height {
            return StoredPixels(width: width, height: height, orientation: orientation, pixels: full,
                                highBitDepth: decoded.bitsPerComponent > 8)
        }
        var pixels = [UInt16](unsafeUninitializedCapacity: width * height * 4) { _, count in count = width * height * 4 }
        let status = full.withUnsafeBufferPointer { input in
            pixels.withUnsafeMutableBufferPointer { output in
                var from = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: input.baseAddress), height: vImagePixelCount(decoded.height),
                                         width: vImagePixelCount(decoded.width), rowBytes: decoded.width * 8)
                var to = vImage_Buffer(data: output.baseAddress, height: vImagePixelCount(height), width: vImagePixelCount(width),
                                       rowBytes: width * 8)
                return vImageScale_ARGB16U(&from, &to, nil, vImage_Flags(kvImageHighQualityResampling))
            }
        }
        guard status == kvImageNoError else { throw StitchError.cannotDecode(image.url) }
        return StoredPixels(width: width, height: height, orientation: orientation, pixels: pixels,
                            highBitDepth: decoded.bitsPerComponent > 8)
    }

    /// RGBX, 16 bits per channel, of `image` at its own size in `space`, transparent areas white.
    private static func draw(_ image: CGImage, space: CGColorSpace, url: URL) throws -> [UInt16] {
        let width = image.width, height = image.height
        var pixels = [UInt16](unsafeUninitializedCapacity: width * height * 4) { _, count in count = width * height * 4 }
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 16, bytesPerRow: width * 8,
                space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGImageByteOrderInfo.order16Little.rawValue
            ) else { return false }
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw StitchError.cannotDecode(url) }
        return pixels
    }

    /// The photos' own colour space when they all share one RGB space (an iPhone's Display P3), else
    /// Display P3, which holds sRGB and most camera spaces.
    static func workingColorSpace(_ images: [SourceImage]) -> CGColorSpace {
        let spaces = images.compactMap { image -> CGColorSpace? in
            guard let source = CGImageSourceCreateWithURL(image.url as CFURL, nil) else { return nil }
            return CGImageSourceCreateImageAtIndex(source, 0, nil)?.colorSpace
        }
        if let first = spaces.first, spaces.count == images.count, first.model == .rgb,
           spaces.allSatisfy({ $0.copyICCData() as Data? == first.copyICCData() as Data? }) {
            return first
        }
        return CGColorSpace(name: CGColorSpace.displayP3)!
    }
}

/// Owns a native compositor; cancel() is safe from any thread.
final class Compositor: @unchecked Sendable {
    let pointer: OpaquePointer

    init(images: [sc_compose_image], options: sc_compose_options) throws {
        var message = [CChar](repeating: 0, count: 512)
        var options = options
        guard let pointer = images.withUnsafeBufferPointer({
            sc_compositor_create($0.baseAddress, Int32($0.count), &options, &message, message.count)
        }) else { throw StitchError.engine(errorText(message)) }
        self.pointer = pointer
    }

    deinit { sc_compositor_free(pointer) }

    func cancel() { sc_compositor_cancel(pointer) }

    var canvas: PixelSize {
        var width: Int32 = 0, height: Int32 = 0
        sc_compositor_canvas_size(pointer, &width, &height)
        return PixelSize(width: Int(width), height: Int(height))
    }

    func seamSize(_ index: Int) -> PixelSize {
        var width: Int32 = 0, height: Int32 = 0
        sc_compositor_seam_size(pointer, Int32(index), &width, &height)
        return PixelSize(width: Int(width), height: Int(height))
    }

    func imageSize(_ index: Int) -> PixelSize {
        var width: Int32 = 0, height: Int32 = 0
        sc_compositor_image_size(pointer, Int32(index), &width, &height)
        return PixelSize(width: Int(width), height: Int(height))
    }

    func outline(_ index: Int) -> [Point2] {
        var points = [Float](repeating: 0, count: 2 * 160)
        let count = Int(sc_compositor_outline(pointer, Int32(index), &points, 160))
        return (0..<count).map { Point2(x: points[2 * $0], y: points[2 * $0 + 1]) }
    }

    /// Turns a native status into Swift: 2 means the work was cancelled.
    private func check(_ status: Int32, _ message: [CChar]) throws {
        if status == 2 { throw CancellationError() }
        if status != 0 { throw StitchError.engine(errorText(message)) }
    }

    func addSeamImage(_ index: Int, _ stored: StoredPixels) throws {
        var message = [CChar](repeating: 0, count: 512)
        let status = stored.pixels.withUnsafeBufferPointer {
            sc_compositor_add_seam_image(pointer, Int32(index), $0.baseAddress, Int32(stored.width), Int32(stored.height),
                                         Int32(stored.bytesPerRow), stored.orientation, &message, message.count)
        }
        try check(status, message)
    }

    /// Exposure gains and seams, advancing `progress`.
    func prepare(progress: WorkLeaf = .none) throws {
        var message = [CChar](repeating: 0, count: 512)
        let observer = NativeProgress(progress)
        let control = observer.control
        let status = withExtendedLifetime(observer) {
            withUnsafePointer(to: control) { sc_compositor_prepare(pointer, $0, &message, message.count) }
        }
        try check(status, message)
    }

    func addImage(_ index: Int, _ stored: StoredPixels) throws {
        var message = [CChar](repeating: 0, count: 512)
        let status = stored.pixels.withUnsafeBufferPointer {
            sc_compositor_add_image(pointer, Int32(index), $0.baseAddress, Int32(stored.width), Int32(stored.height),
                                    Int32(stored.bytesPerRow), stored.orientation, &message, message.count)
        }
        try check(status, message)
    }

    /// The whole panorama, the native buffer handed over to `Data` without a copy.
    func finish(space: CGColorSpace) throws -> (PanoramaPixels, PixelRect) {
        var message = [CChar](repeating: 0, count: 512)
        var result = sc_panorama()
        try check(sc_compositor_finish(pointer, &result, &message, message.count), message)
        guard let pixels = result.pixels else { throw StitchError.engine(String(localized: "The panorama is empty")) }
        let width = Int(result.width), height = Int(result.height)
        let data = Data(bytesNoCopy: pixels, count: width * height * 8, deallocator: .free)
        let crop = PixelRect(x: Int(result.crop.0), y: Int(result.crop.1), width: Int(result.crop.2),
                             height: Int(result.crop.3))
        return (PanoramaPixels(width: width, height: height, bitsPerComponent: 16, pixels: data,
                               iccProfile: space.copyICCData() as Data?, opaque: result.opaque != 0), crop)
    }
}

/// Field of view of a rotation panorama, after centring it on the middle of its sweep.
struct RotationLayout {
    var rotations: [[Double]]
    /// Degrees.
    var yawExtent: Double
    var pitchExtent: Double
    /// Largest angle between a photo corner and the projection axis.
    var maxAngle: Double

    /// Rays through the edges of every photo, yaw = atan2(x, z) and pitch from the horizontal plane.
    init(_ alignment: Alignment, images: [SourceImage]) {
        func rays(_ rotation: simd_double3x3, _ focal: Double, _ size: PixelSize) -> [SIMD3<Double>] {
            let w = Double(size.width), h = Double(size.height)
            var points: [SIMD2<Double>] = []
            for k in 0...40 {
                let t = Double(k) / 40
                points += [SIMD2(t * w, 0), SIMD2(t * w, h), SIMD2(0, t * h), SIMD2(w, t * h)]
            }
            points.append(SIMD2(w / 2, h / 2))
            return points.map { normalize(rotation * SIMD3($0.x - w / 2, $0.y - h / 2, focal)) }
        }
        let matrices = alignment.transforms.map { PlaneGeometry.matrix($0) }
        let all = zip(matrices, zip(alignment.focals, images)).flatMap { rays($0, $1.0, $1.1.pixelSize) }
        let yaws = all.map { atan2($0.x, $0.z) }.sorted()
        // The sweep is the circle minus its largest empty gap; centre the panorama on its middle.
        var gap = yaws.first! + 2 * .pi - yaws.last!, start = yaws.first!
        for (previous, next) in zip(yaws, yaws.dropFirst()) where next - previous > gap {
            gap = next - previous
            start = next
        }
        let extent = 2 * .pi - gap
        let middle = start + extent / 2
        // Rotation about the vertical axis that brings the middle of the sweep to +z.
        let c = cos(-middle), s = sin(-middle)
        let recentre = simd_double3x3(rows: [SIMD3(c, 0, s), SIMD3(0, 1, 0), SIMD3(-s, 0, c)])
        let centred = matrices.map { recentre * $0 }
        rotations = centred.map { m in [m[0, 0], m[1, 0], m[2, 0], m[0, 1], m[1, 1], m[2, 1], m[0, 2], m[1, 2], m[2, 2]] }
        let centredRays = all.map { recentre * $0 }
        let pitches = centredRays.map { atan2($0.y, ($0.x * $0.x + $0.z * $0.z).squareRoot()) }
        yawExtent = extent * 180 / .pi
        pitchExtent = ((pitches.max() ?? 0) - (pitches.min() ?? 0)) * 180 / .pi
        maxAngle = centredRays.map { acos(min(1, max(-1, $0.z))) }.max().map { $0 * 180 / .pi } ?? 0
    }

    /// Rectilinear while straight lines can stay straight without extreme stretch, cylindrical while the
    /// vertical extent allows, spherical otherwise.
    func projection(for requested: Projection) throws -> Projection {
        if yawExtent >= 330 {
            throw StitchError.engine(String(localized: "Full 360-degree panoramas are not supported yet"))
        }
        switch requested {
        case .rectilinear:
            guard maxAngle < 80 else {
                throw StitchError.engine(String(format: String(localized: "The photos span %.0f degrees: too wide for a rectilinear projection"), locale: .current, yawExtent))
            }
            return .rectilinear
        case .cylindrical, .spherical:
            return requested
        case .automatic, .flat:
            if yawExtent <= 100 && pitchExtent <= 100 && maxAngle <= 60 { return .rectilinear }
            return pitchExtent <= 100 ? .cylindrical : .spherical
        }
    }
}

extension StitchEngine {
    /// Bytes per panorama pixel while compositing: multi-band holds a 16-bit pyramid and float weights
    /// (about 45 B/px measured), the direct composite an image and a mask; plus the result kept afterwards.
    static func bytesPerPixel(_ blending: Blending) -> Double { blending == .multiBand ? 53 : 23 }

    /// The output scale for a panorama whose canvas is `canvas` at scale `requested`: lowered so that it
    /// takes at most half of `memory` and at most SC_MAX_PANORAMA_SIDE pixels per side.
    static func outputScale(canvas: PixelSize, requested: Double, bytesPerPixel: Double, memory: UInt64)
        -> (scale: Double, limitedBySize: Bool) {
        let budget = 0.5 * Double(memory)
        let needed = Double(canvas.width) * Double(canvas.height) * bytesPerPixel
        let byMemory = needed > budget ? (budget / needed).squareRoot() * 0.98 : 1
        // A few pixels under the limit: the canvas is rounded outwards to whole pixels at the new scale.
        let side = Double(max(canvas.width, canvas.height)), limit = Double(SC_MAX_PANORAMA_SIDE)
        let bySize = side > limit ? (limit - 4) / side : 1
        return (requested * min(byMemory, bySize), bySize < byMemory)
    }

    /// Joins the main group of `report` into one image. The photos are decoded again from their files, so
    /// the report must have its local paths. Cancelling the calling task stops the work at the next photo.
    /// Progress stages: "align" (a fraction), "seams" (k of n), "exposure" (a fraction), "compose" (k of n), "blend".
    public nonisolated func stitch(
        _ report: MatchReport, request: StitchRequest = StitchRequest(),
        progress: (@Sendable (ProgressEvent) -> Void)? = nil
    ) async throws -> Panorama {
        let start = ContinuousClock.now
        var timings = PanoramaTimings()
        // The alignment, then about 2 ms a photo for the layout and the compositor.
        let meter = WorkMeter(stage: "align", emit: progress)
        let outcome = try Self.alignment(for: report, straighten: request.straighten,
                                         progress: meter.begin(after: 2e-3 * Double(report.images.count)))
        let problem = outcome.problem, alignment = outcome.alignment
        var notes = outcome.notes
        let images = problem.images
        try Task.checkCancellation()

        // Geometry for the compositor.
        let projection: Projection
        var transforms = alignment.transforms
        if alignment.model == .rotation {
            let layout = RotationLayout(alignment, images: images)
            projection = try layout.projection(for: request.projection)
            transforms = layout.rotations
        } else {
            projection = .flat
            // The compositor refuses a photo whose corner falls behind the plane or towards infinity.
            for (image, t) in zip(images, transforms) {
                let w = Double(image.pixelSize.width - 1), h = Double(image.pixelSize.height - 1)
                for (x, y) in [(0.0, 0.0), (w, 0), (w, h), (0, h)] {
                    let z = t[6] * x + t[7] * y + t[8]
                    let u = (t[0] * x + t[1] * y + t[2]) / z, v = (t[3] * x + t[4] * y + t[5]) / z
                    guard z > 1e-9, abs(u) < 1e7, abs(v) < 1e7 else {
                        throw StitchError.engine(String(localized: "The photos cannot be laid on one plane: try Rotation mode"))
                    }
                }
            }
        }
        let native: sc_projection = switch projection {
        case .rectilinear: SC_PROJECTION_RECTILINEAR
        case .cylindrical: SC_PROJECTION_CYLINDRICAL
        case .spherical: SC_PROJECTION_SPHERICAL
        default: SC_PROJECTION_PLANE
        }
        let composeImages = images.indices.map { i -> sc_compose_image in
            let t = transforms[i]
            return sc_compose_image(width: Int32(images[i].pixelSize.width), height: Int32(images[i].pixelSize.height),
                                    transform: (t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]),
                                    focal: alignment.model == .rotation ? alignment.focals[i] : 0)
        }
        let areas = images.map { Double($0.pixelSize.width * $0.pixelSize.height) }.sorted()
        // Seams and exposure on copies of about 0.1 MP per photo, as OpenCV's stitcher: graph cut grows fast
        // with the overlap area (8 s at 0.3 MP on four heavily overlapping scans, 0.8 s at 0.1 MP).
        let seamMegapixels = 0.1
        var options = sc_compose_options()
        options.projection = native
        options.scale = request.size.scale
        // Never above the output scale: the memory budget below covers the output canvas only.
        options.seam_scale = min(options.scale, (seamMegapixels * 1_000_000 / areas[areas.count / 2]).squareRoot())
        options.seam = request.seams == .voronoi ? SC_SEAM_VORONOI : images.count > 30 ? SC_SEAM_DP : SC_SEAM_GRAPHCUT
        options.exposure = switch request.exposure {
        case .channels: SC_EXPOSURE_CHANNELS
        case .blocks: SC_EXPOSURE_BLOCKS
        case .none: SC_EXPOSURE_NONE
        }
        options.similarity = 0.15
        options.blend = request.blending == .none ? SC_BLEND_NONE : SC_BLEND_MULTIBAND
        options.interpolation = 2

        // Lower the resolution when the panorama would not fit in half the memory or exceed the size limit.
        var compositor = try Compositor(images: composeImages, options: options)
        // Four times the size limit at the requested scale is not a mosaic but a broken alignment (a corner
        // sent far away): refuse it rather than make a tiny panorama.
        let planned = compositor.canvas
        guard max(planned.width, planned.height) <= 4 * Int(SC_MAX_PANORAMA_SIDE) else {
            throw StitchError.engine(String(format: String(localized: "The panorama would be %lld × %lld pixels: the alignment is probably wrong, try another mode"),
                                            locale: .current, planned.width, planned.height))
        }
        let fitted = Self.outputScale(canvas: compositor.canvas, requested: options.scale,
                                      bytesPerPixel: Self.bytesPerPixel(request.blending),
                                      memory: ProcessInfo.processInfo.physicalMemory)
        if fitted.scale < options.scale {
            options.scale = fitted.scale
            options.seam_scale = min(options.seam_scale, options.scale)
            compositor = try Compositor(images: composeImages, options: options)
            let percent = 100 * options.scale / request.size.scale
            notes.append(fitted.limitedBySize
                ? String(format: String(localized: "Made at %.0f%% of the requested size: a panorama can be at most %lld pixels wide or tall"),
                         locale: .current, percent, Int(SC_MAX_PANORAMA_SIDE))
                : String(format: String(localized: "Made at %.0f%% of the requested size to fit in memory"), locale: .current, percent))
        }
        let canvas = compositor.canvas
        guard max(canvas.width, canvas.height) <= Int(SC_MAX_PANORAMA_SIDE) else {
            throw StitchError.engine(String(format: String(localized: "The panorama would be %lld × %lld pixels: it can be at most %lld per side"),
                                            locale: .current, canvas.width, canvas.height, Int(SC_MAX_PANORAMA_SIDE)))
        }
        timings.alignment = (ContinuousClock.now - start).seconds

        let space = ImageLoader.workingColorSpace(images)
        meter.finish()
        let worker = compositor
        let (pixels, crop) = try await withTaskCancellationHandler {
            try Self.composite(worker, images: images, space: space, progress: progress, timings: &timings)
        } onCancel: {
            worker.cancel()
        }
        let preview = try pixels.preview()
        timings.total = (ContinuousClock.now - start).seconds

        let ids = Set(images.map(\.id))
        var leftOut: [Int: ExclusionReason] = [:]
        let inconsistent = Set(outcome.leftOutImages), misplaced = Set(outcome.misplacedImages)
        for node in report.graph.nodes where !ids.contains(node.id) {
            leftOut[node.id] = inconsistent.contains(node.id) ? .inconsistentPairs
                : misplaced.contains(node.id) ? .misplaced : node.exclusion ?? .separateGroup
        }
        return Panorama(
            request: request, model: alignment.model, projection: projection, pixels: pixels, crop: crop,
            preview: preview,
            outlines: Dictionary(uniqueKeysWithValues: images.indices.map { (images[$0].id, compositor.outline($0)) }),
            imageIDs: images.map(\.id), leftOut: leftOut, leftOutPairs: outcome.leftOutPairs, reportCreatedAt: report.createdAt,
            alignmentError: alignment.rms, notes: notes, timings: timings
        )
    }

    private static func composite(
        _ compositor: Compositor, images: [SourceImage], space: CGColorSpace,
        progress: (@Sendable (ProgressEvent) -> Void)?, timings: inout PanoramaTimings
    ) throws -> (PanoramaPixels, PixelRect) {
        var clock = ContinuousClock.now
        progress?(ProgressEvent(stage: "seams", completed: 0, total: images.count))
        for (index, image) in images.enumerated() {
            try Task.checkCancellation()
            try compositor.addSeamImage(index, ImageLoader.stored(image, target: compositor.seamSize(index), space: space))
            progress?(ProgressEvent(stage: "seams", completed: index + 1, total: images.count))
        }
        let meter = WorkMeter(stage: "exposure", emit: progress)
        try compositor.prepare(progress: meter.begin().leaf(1))
        meter.finish()
        timings.seams = (ContinuousClock.now - clock).seconds
        clock = ContinuousClock.now
        progress?(ProgressEvent(stage: "compose", completed: 0, total: images.count))
        var highBitDepth = false
        for (index, image) in images.enumerated() {
            try Task.checkCancellation()
            let stored = try ImageLoader.stored(image, target: compositor.imageSize(index), space: space)
            highBitDepth = highBitDepth || stored.highBitDepth
            try compositor.addImage(index, stored)
            progress?(ProgressEvent(stage: "compose", completed: index + 1, total: images.count))
        }
        timings.compositing = (ContinuousClock.now - clock).seconds
        clock = ContinuousClock.now
        progress?(ProgressEvent(stage: "blend", completed: 0, total: 0))
        var result = try compositor.finish(space: space)
        result.0.highBitDepth = highBitDepth
        timings.blending = (ContinuousClock.now - clock).seconds
        return result
    }
}
