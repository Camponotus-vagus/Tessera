import CoreGraphics
import CStitchCore
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum StitchError: Error, LocalizedError, Sendable {
    case cannotOpen(URL)
    case cannotDecode(URL)
    case engine(String)
    case modelMissing
    case noMatcher
    case nothingToStitch
    case photoChanged(URL)

    public var errorDescription: String? {
        switch self {
        case .cannotOpen(let url): String(localized: "Cannot open \(url.lastPathComponent)")
        case .cannotDecode(let url): String(localized: "Cannot decode \(url.lastPathComponent)")
        case .engine(let message): message
        case .modelMissing: String(localized: "No LightGlue model configured")
        case .noMatcher: String(localized: "No matcher selected")
        case .nothingToStitch: String(localized: "No two photos were joined")
        case .photoChanged(let url): String(localized: "\(url.lastPathComponent) changed since the analysis")
        }
    }

    /// The photo itself is the problem, not the engine: it is reported and the others go on.
    var isUnreadableImage: Bool {
        switch self {
        case .cannotOpen, .cannotDecode: true
        default: false
        }
    }
}

/// Pixel buffer at a reduced working resolution.
struct WorkingImage: Sendable {
    var size: PixelSize
    /// Working pixels per original pixel, per axis (rounding makes them differ slightly).
    var scale: SIMD2<Double>
    var gray: [UInt8] = []
    var planar: [Float] = []
}

enum ImageLoader {
    static func describe(url: URL, id: Int) throws -> SourceImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw StitchError.cannotOpen(url)
        }
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int, width > 0, height > 0
        else {
            throw StitchError.cannotDecode(url)
        }
        let orientation = properties[kCGImagePropertyOrientation] as? UInt32 ?? 1
        let swapped = (5...8).contains(orientation)
        let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any]
        let focal = exif?[kCGImagePropertyExifFocalLenIn35mmFilm] as? Double
        var date: Date?
        if let text = exif?[kCGImagePropertyExifDateTimeOriginal] as? String {
            let formatter = DateFormatter()
            formatter.locale = Locale(identifier: "en_US_POSIX")
            formatter.dateFormat = "yyyy:MM:dd HH:mm:ss"
            date = formatter.date(from: text)
        }
        return SourceImage(
            id: id,
            url: url,
            name: url.lastPathComponent,
            pixelSize: swapped ? PixelSize(width: height, height: width) : PixelSize(width: width, height: height),
            captureDate: date,
            focalLength35mm: focal
        )
    }

    /// Decodes an oriented thumbnail whose long side is `longSide` pixels, for display only: its pixels do not
    /// follow a known map (see analysisImage).
    static func thumbnail(url: URL, longSide: Int) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw StitchError.cannotOpen(url)
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: longSide,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw StitchError.cannotDecode(url)
        }
        return image
    }

    /// The oriented photo with its long side at `longSide` pixels (never enlarged), resampled so that pixel
    /// centres map as (x + 0.5) * r - 0.5 on each axis, the map the keypoint conversions assume: decoded at a
    /// power-of-two reduction, which ImageIO does exactly on that map for JPEG, HEIC, PNG and TIFF, then reduced by
    /// area averaging, as LightGlue's own image loading does (cv2.INTER_AREA). ImageIO thumbnails do not follow
    /// that map: JPEG ones are reduced corner to corner after the power-of-two step, which shrinks them by up to
    /// 5e-4 at 1024 pixels, and HEIC ones are also shifted unevenly.
    static func analysisImage(for image: SourceImage, longSide: Int) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(image.url as CFURL, nil) else {
            throw StitchError.cannotOpen(image.url)
        }
        let orientation = orientation(of: source)
        let swapped = (5...8).contains(orientation)
        let full = image.pixelSize
        let factor = min(1, Double(longSide) / Double(max(full.width, full.height)))
        let target = PixelSize(width: max(1, Int((Double(full.width) * factor).rounded())),
                               height: max(1, Int((Double(full.height) * factor).rounded())))
        // Sizes as stored in the file, before the EXIF orientation.
        let stored = swapped ? PixelSize(width: full.height, height: full.width) : full
        let goal = swapped ? PixelSize(width: target.height, height: target.width) : target
        // Only codecs measured to reduce exactly on the map: JPEG 2000, for one, is off by a third of a pixel.
        let exact: [UTType] = [.jpeg, .heic, .heif, .png, .tiff]
        let type = (CGImageSourceGetType(source) as String?).flatMap { UTType($0) }
        let subsample = type.map { t in exact.contains { t.conforms(to: $0) } } == true ? [8, 4, 2].first {
            stored.width % $0 == 0 && stored.height % $0 == 0 && stored.width / $0 >= goal.width && stored.height / $0 >= goal.height
        } ?? 1 : 1
        let options = subsample > 1 ? [kCGImageSourceSubsampleFactor: subsample] as CFDictionary : nil
        guard var decoded = CGImageSourceCreateImageAtIndex(source, 0, options) else {
            throw StitchError.cannotDecode(image.url)
        }
        // A codec without reduced decoding returns the full image, which is just as exact.
        let reduced = PixelSize(width: stored.width / subsample, height: stored.height / subsample)
        if PixelSize(width: decoded.width, height: decoded.height) != reduced {
            guard let whole = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw StitchError.cannotDecode(image.url) }
            decoded = whole
        }
        // A photo that changed since it was described is left out like an unreadable one.
        guard PixelSize(width: decoded.width, height: decoded.height) == reduced
            || PixelSize(width: decoded.width, height: decoded.height) == stored
        else { throw StitchError.cannotDecode(image.url) }

        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGImageAlphaInfo.noneSkipLast.rawValue
        var width = decoded.width, height = decoded.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: space, bitmapInfo: info) else { return false }
            // Transparent areas become white, as in a viewer.
            context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.draw(decoded, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw StitchError.cannotDecode(image.url) }
        if width != goal.width || height != goal.height {
            var reduced = [UInt8](repeating: 0, count: goal.width * goal.height * 4)
            let done = pixels.withUnsafeBufferPointer { input in
                reduced.withUnsafeMutableBufferPointer { output in
                    sc_resize_area_rgba8(input.baseAddress, Int32(width), Int32(height), Int32(width * 4), output.baseAddress,
                                         Int32(goal.width), Int32(goal.height))
                }
            }
            guard done == 1 else { throw StitchError.cannotDecode(image.url) }
            pixels = reduced
            width = goal.width
            height = goal.height
        }
        let (upright, uprightWidth, uprightHeight) = oriented(pixels, width: width, height: height, orientation: orientation)
        guard let provider = CGDataProvider(data: Data(upright) as CFData),
              let result = CGImage(width: uprightWidth, height: uprightHeight, bitsPerComponent: 8, bitsPerPixel: 32,
                                   bytesPerRow: uprightWidth * 4, space: space, bitmapInfo: CGBitmapInfo(rawValue: info),
                                   provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw StitchError.cannotDecode(image.url) }
        return result
    }

    /// Four-byte pixels as stored, turned the way a viewer shows them for EXIF orientation 1-8 (the same
    /// rule as the compositor's).
    static func oriented(_ pixels: [UInt8], width: Int, height: Int, orientation: Int32) -> ([UInt8], Int, Int) {
        guard (2...8).contains(orientation) else { return (pixels, width, height) }
        let swapped = orientation >= 5
        let outWidth = swapped ? height : width, outHeight = swapped ? width : height
        var out = [UInt8](repeating: 0, count: pixels.count)
        pixels.withUnsafeBytes { source in
            out.withUnsafeMutableBytes { target in
                let s = source.bindMemory(to: UInt32.self), t = target.bindMemory(to: UInt32.self)
                for y in 0..<outHeight {
                    for x in 0..<outWidth {
                        let (sx, sy): (Int, Int) = switch orientation {
                        case 2: (width - 1 - x, y)
                        case 3: (width - 1 - x, height - 1 - y)
                        case 4: (x, height - 1 - y)
                        case 5: (y, x)
                        case 6: (y, height - 1 - x)
                        case 7: (width - 1 - y, height - 1 - x)
                        default: (width - 1 - y, x)  // 8
                        }
                        t[y * outWidth + x] = s[sy * width + sx]
                    }
                }
            }
        }
        return (out, outWidth, outHeight)
    }

    static func longSide(for image: SourceImage, megapixels: Double) -> Int {
        let original = Double(image.pixelSize.width * image.pixelSize.height)
        let factor = min(1.0, (megapixels * 1_000_000 / original).squareRoot())
        return Int((Double(max(image.pixelSize.width, image.pixelSize.height)) * factor).rounded())
    }

    static func gray(for image: SourceImage, megapixels: Double) throws -> WorkingImage {
        let cgImage = try analysisImage(for: image, longSide: longSide(for: image, megapixels: megapixels))
        let width = cgImage.width
        let height = cgImage.height
        var pixels = [UInt8](repeating: 0, count: width * height)
        let drawn = pixels.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            // Transparent areas become white, as in a viewer, instead of black.
            context.setFillColor(gray: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw StitchError.cannotDecode(image.url) }
        return WorkingImage(
            size: PixelSize(width: width, height: height),
            scale: SIMD2(Double(width) / Double(image.pixelSize.width), Double(height) / Double(image.pixelSize.height)),
            gray: pixels
        )
    }

    /// RGB planar float image fitted inside a `canvas`, top-left aligned and zero padded.
    static func planarRGB(for image: SourceImage, longSide: Int, canvas: PixelSize) throws -> WorkingImage {
        let cgImage = try analysisImage(for: image, longSide: longSide)
        let width = min(cgImage.width, canvas.width)
        let height = min(cgImage.height, canvas.height)
        var rgba = [UInt8](repeating: 0, count: canvas.width * canvas.height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: canvas.width, height: canvas.height, bitsPerComponent: 8,
                bytesPerRow: canvas.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { return false }
            context.interpolationQuality = .high
            // CoreGraphics has its origin at the bottom left; draw into the top band, over white so that
            // transparent areas look as they do in a viewer. The padding stays black, as in training.
            let content = CGRect(x: 0, y: canvas.height - height, width: width, height: height)
            context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
            context.fill(content)
            context.draw(cgImage, in: content)
            return true
        }
        guard drawn else { throw StitchError.cannotDecode(image.url) }

        let plane = canvas.width * canvas.height
        var planar = [Float](repeating: 0, count: plane * 3)
        for index in 0..<plane {
            planar[index] = Float(rgba[index * 4]) / 255
            planar[plane + index] = Float(rgba[index * 4 + 1]) / 255
            planar[2 * plane + index] = Float(rgba[index * 4 + 2]) / 255
        }
        return WorkingImage(
            size: PixelSize(width: width, height: height),
            scale: SIMD2(Double(width) / Double(image.pixelSize.width), Double(height) / Double(image.pixelSize.height)),
            planar: planar
        )
    }
}
