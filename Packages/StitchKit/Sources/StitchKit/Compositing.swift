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
    var bytesPerRow: Int { width * 8 }
}

extension ImageLoader {
    static func orientation(of source: CGImageSource) -> Int32 {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let value = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.int32Value ?? 1
        return (1...8).contains(value) ? value : 1
    }

    /// The full image decoded by ImageIO (not the thumbnail path, which decodes JPEG slightly differently),
    /// drawn into `space` at a size that becomes `target` once oriented. 8-bit sources come out as 257 x v.
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
        var pixels = [UInt16](unsafeUninitializedCapacity: width * height * 4) { _, count in count = width * height * 4 }
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 16, bytesPerRow: width * 8,
                space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGImageByteOrderInfo.order16Little.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.draw(decoded, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw StitchError.cannotDecode(image.url) }
        return StoredPixels(width: width, height: height, orientation: orientation, pixels: pixels)
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

    func prepare() throws {
        var message = [CChar](repeating: 0, count: 512)
        try check(sc_compositor_prepare(pointer, &message, message.count), message)
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
        guard let pixels = result.pixels else { throw StitchError.engine("The panorama is empty") }
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
            throw StitchError.engine("Full 360-degree panoramas are not supported yet")
        }
        switch requested {
        case .rectilinear:
            guard maxAngle < 80 else {
                throw StitchError.engine(String(format: "The photos span %.0f degrees: too wide for a rectilinear projection", yawExtent))
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

    /// Joins the main group of `report` into one image. The photos are decoded again from their files, so
    /// the report must have its local paths. Cancelling the calling task stops the work at the next photo.
    /// Progress stages: "align", "seams" (k of n), "exposure", "compose" (k of n), "blend".
    public nonisolated func stitch(
        _ report: MatchReport, request: StitchRequest = StitchRequest(),
        progress: (@Sendable (ProgressEvent) -> Void)? = nil
    ) async throws -> Panorama {
        let start = ContinuousClock.now
        var timings = PanoramaTimings()
        progress?(ProgressEvent(stage: "align", completed: 0, total: 0))
        let (problem, alignment, alignmentNotes) = try Self.alignment(for: report, straighten: request.straighten)
        var notes = alignmentNotes
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
        options.seam_scale = min(1, (seamMegapixels * 1_000_000 / areas[areas.count / 2]).squareRoot())
        options.seam = request.seams == .voronoi ? SC_SEAM_VORONOI : images.count > 30 ? SC_SEAM_DP : SC_SEAM_GRAPHCUT
        options.exposure = switch request.exposure {
        case .channels: SC_EXPOSURE_CHANNELS
        case .blocks: SC_EXPOSURE_BLOCKS
        case .none: SC_EXPOSURE_NONE
        }
        options.similarity = 0.15
        options.blend = request.blending == .none ? SC_BLEND_NONE : SC_BLEND_MULTIBAND
        options.interpolation = 2

        // Lower the resolution when the panorama would not fit in half the memory.
        var compositor = try Compositor(images: composeImages, options: options)
        let budget = 0.5 * Double(ProcessInfo.processInfo.physicalMemory)
        let needed = Double(compositor.canvas.width * compositor.canvas.height) * Self.bytesPerPixel(request.blending)
        if needed > budget {
            options.scale *= (budget / needed).squareRoot() * 0.98
            compositor = try Compositor(images: composeImages, options: options)
            notes.append(String(format: "Made at %.0f%% of the requested size to fit in memory", 100 * options.scale / request.size.scale))
        }
        timings.alignment = (ContinuousClock.now - start).seconds

        let space = ImageLoader.workingColorSpace(images)
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
        for node in report.graph.nodes where !ids.contains(node.id) {
            leftOut[node.id] = node.exclusion ?? .separateGroup
        }
        return Panorama(
            request: request, model: alignment.model, projection: projection, pixels: pixels, crop: crop,
            preview: preview,
            outlines: Dictionary(uniqueKeysWithValues: images.indices.map { (images[$0].id, compositor.outline($0)) }),
            imageIDs: images.map(\.id), leftOut: leftOut, reportCreatedAt: report.createdAt,
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
        progress?(ProgressEvent(stage: "exposure", completed: 0, total: 0))
        try compositor.prepare()
        timings.seams = (ContinuousClock.now - clock).seconds
        clock = ContinuousClock.now
        progress?(ProgressEvent(stage: "compose", completed: 0, total: images.count))
        for (index, image) in images.enumerated() {
            try Task.checkCancellation()
            try compositor.addImage(index, ImageLoader.stored(image, target: compositor.imageSize(index), space: space))
            progress?(ProgressEvent(stage: "compose", completed: index + 1, total: images.count))
        }
        timings.compositing = (ContinuousClock.now - clock).seconds
        clock = ContinuousClock.now
        progress?(ProgressEvent(stage: "blend", completed: 0, total: 0))
        let result = try compositor.finish(space: space)
        timings.blending = (ContinuousClock.now - clock).seconds
        return result
    }
}
