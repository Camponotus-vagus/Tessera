import CoreGraphics
import Foundation
import ImageIO

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
        case .cannotOpen(let url): "Cannot open \(url.lastPathComponent)"
        case .cannotDecode(let url): "Cannot decode \(url.lastPathComponent)"
        case .engine(let message): message
        case .modelMissing: "No LightGlue model configured"
        case .noMatcher: "No matcher selected"
        case .nothingToStitch: "No two photos were joined"
        case .photoChanged(let url): "\(url.lastPathComponent) changed since the analysis"
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

    /// Decodes an oriented thumbnail whose long side is `longSide` pixels.
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

    static func longSide(for image: SourceImage, megapixels: Double) -> Int {
        let original = Double(image.pixelSize.width * image.pixelSize.height)
        let factor = min(1.0, (megapixels * 1_000_000 / original).squareRoot())
        return Int((Double(max(image.pixelSize.width, image.pixelSize.height)) * factor).rounded())
    }

    static func gray(for image: SourceImage, megapixels: Double) throws -> WorkingImage {
        let cgImage = try thumbnail(url: image.url, longSide: longSide(for: image, megapixels: megapixels))
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
        let cgImage = try thumbnail(url: image.url, longSide: longSide)
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
