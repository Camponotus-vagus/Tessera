import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// File formats a panorama can be saved in.
public enum PanoramaFormat: String, Sendable, Codable, CaseIterable {
    case png
    case jpeg
    case tiff
    case heic

    public var fileExtension: String {
        switch self {
        case .png: "png"
        case .jpeg: "jpg"
        case .tiff: "tif"
        case .heic: "heic"
        }
    }

    var type: UTType {
        switch self {
        case .png: .png
        case .jpeg: .jpeg
        case .tiff: .tiff
        case .heic: .heic
        }
    }

    /// Whether the format can keep 16 bits per channel.
    public var supportsSixteenBit: Bool { self == .png || self == .tiff }

    /// Whether the format can keep the transparent area around an uncropped panorama.
    public var supportsTransparency: Bool { self == .png || self == .tiff }
}

public struct PanoramaExportOptions: Sendable, Codable, Hashable {
    public var format: PanoramaFormat = .jpeg
    /// 0...1, for JPEG and HEIC.
    public var quality = 0.92
    /// 16 bits per channel when the panorama has them and the format allows it.
    public var sixteenBit = false

    public init(format: PanoramaFormat = .jpeg, quality: Double = 0.92, sixteenBit: Bool = false) {
        self.format = format
        self.quality = quality
        self.sixteenBit = sixteenBit
    }
}

/// An RGBA pixel buffer, unpremultiplied, rows top to bottom.
public struct PanoramaPixels: Sendable {
    public var width: Int
    public var height: Int
    /// 8 or 16.
    public var bitsPerComponent: Int
    public var pixels: Data
    /// ICC profile of the colour space the pixels are in.
    public var iccProfile: Data?
    /// Every pixel is covered: the alpha channel can be dropped.
    public var opaque: Bool

    public init(width: Int, height: Int, bitsPerComponent: Int, pixels: Data, iccProfile: Data?, opaque: Bool = false) {
        self.width = width
        self.height = height
        self.bitsPerComponent = bitsPerComponent
        self.pixels = pixels
        self.iccProfile = iccProfile
        self.opaque = opaque
    }

    public var bytesPerRow: Int { width * 4 * bitsPerComponent / 8 }

    var colorSpace: CGColorSpace {
        if let iccProfile, let space = CGColorSpace(iccData: iccProfile as CFData) { return space }
        return CGColorSpace(name: CGColorSpace.sRGB)!
    }

    /// The pixels as a CGImage over the same memory, without alpha when `opaque` (or when `crop`, which must
    /// hold no transparent pixel, is given).
    func sourceImage(crop: PixelRect? = nil) throws -> CGImage {
        guard pixels.count >= bytesPerRow * height, width > 0, height > 0 else {
            throw StitchError.engine("Panorama buffer is smaller than its size")
        }
        let alpha: CGImageAlphaInfo = opaque || crop != nil ? .noneSkipLast : .last
        let info: CGBitmapInfo = bitsPerComponent == 16
            ? [CGBitmapInfo(rawValue: alpha.rawValue), .byteOrder16Little]
            : CGBitmapInfo(rawValue: alpha.rawValue)
        guard let provider = CGDataProvider(data: pixels as CFData),
              let source = CGImage(width: width, height: height, bitsPerComponent: bitsPerComponent,
                                   bitsPerPixel: 4 * bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace,
                                   bitmapInfo: info, provider: provider, decode: nil, shouldInterpolate: false,
                                   intent: .defaultIntent)
        else { throw StitchError.engine("Cannot describe the panorama as an image") }
        guard let crop else { return source }
        guard let cropped = source.cropping(to: crop.cgRect) else { throw StitchError.engine("Invalid crop") }
        return cropped
    }

    /// The pixels as a CGImage, optionally cropped, converted to 8 bits and flattened onto `background`.
    func cgImage(bitsPerComponent target: Int, flattenOnto background: CGColor? = nil, crop: PixelRect? = nil) throws
        -> CGImage {
        let source = try sourceImage(crop: crop)
        let opaqueSource = source.alphaInfo == .noneSkipLast
        guard target != bitsPerComponent || (background != nil && !opaqueSource) else { return source }
        let width = source.width, height = source.height

        // Redraw: CoreGraphics premultiplies while drawing, so 8-bit output from 16-bit input is rounded once.
        let alpha: CGImageAlphaInfo = background == nil && !opaqueSource ? .premultipliedLast : .noneSkipLast
        let outInfo: CGBitmapInfo = target == 16
            ? [CGBitmapInfo(rawValue: alpha.rawValue), .byteOrder16Little]
            : CGBitmapInfo(rawValue: alpha.rawValue)
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: target,
                                      bytesPerRow: 0, space: colorSpace, bitmapInfo: outInfo.rawValue)
        else { throw StitchError.engine(String(localized: "Not enough memory to convert the panorama")) }
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        if let background {
            context.setFillColor(background)
            context.fill(rect)
        }
        context.draw(source, in: rect)
        guard let image = context.makeImage() else { throw StitchError.engine("Cannot convert the panorama") }
        return image
    }

    /// An 8-bit copy with the long side at most `longSide`, for display.
    public func preview(longSide: Int = 4096, crop: PixelRect? = nil) throws -> CGImage {
        let source = try sourceImage(crop: crop)
        let factor = min(1, Double(longSide) / Double(max(source.width, source.height)))
        let width = max(1, Int((Double(source.width) * factor).rounded()))
        let height = max(1, Int((Double(source.height) * factor).rounded()))
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw StitchError.engine(String(localized: "Not enough memory for the preview")) }
        context.interpolationQuality = .high
        context.draw(source, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = context.makeImage() else { throw StitchError.engine("Cannot make the preview") }
        return image
    }
}

public enum PanoramaWriter {
    /// Writes `panorama` to `url` in the chosen format, replacing an existing file only once the new one is
    /// complete. JPEG and HEIC have no transparency, so an uncropped panorama is flattened onto white.
    /// `crop` (the panorama's inscribed rectangle) keeps only that area, without alpha.
    public static func write(_ panorama: PanoramaPixels, crop: PixelRect? = nil, to url: URL,
                             options: PanoramaExportOptions) throws {
        let format = options.format
        let side = max(crop?.width ?? panorama.width, crop?.height ?? panorama.height)
        if format == .jpeg && side > 65535 {
            throw StitchError.engine(String(localized: "JPEG files can be at most 65535 pixels wide or tall: choose PNG, TIFF or HEIC"))
        }
        let depth = options.sixteenBit && format.supportsSixteenBit && panorama.bitsPerComponent == 16 ? 16 : 8
        let background = format.supportsTransparency ? nil : CGColor(gray: 1, alpha: 1)
        let image = try panorama.cgImage(bitsPerComponent: depth, flattenOnto: background, crop: crop)

        let directory = url.deletingLastPathComponent()
        let staging = directory.appendingPathComponent(".\(UUID().uuidString)-\(url.lastPathComponent)")
        guard let destination = CGImageDestinationCreateWithURL(staging as CFURL, format.type.identifier as CFString,
                                                                1, nil)
        else { throw StitchError.engine(String(localized: "This Mac cannot write \(format.rawValue.uppercased()) files")) }
        var properties: [CFString: Any] = [kCGImagePropertyOrientation: 1]
        switch format {
        case .jpeg, .heic:
            properties[kCGImageDestinationLossyCompressionQuality] = min(1, max(0, options.quality))
        case .tiff:
            // LZW for 8 bits; on 16-bit data ImageIO's LZW grows the file, Deflate does not.
            properties[kCGImagePropertyTIFFDictionary] = [kCGImagePropertyTIFFCompression: depth == 16 ? 8 : 5]
        case .png:
            break
        }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            try? FileManager.default.removeItem(at: staging)
            throw StitchError.engine(String(localized: "Could not write \(url.lastPathComponent)"))
        }
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                _ = try FileManager.default.replaceItemAt(url, withItemAt: staging)
            } else {
                try FileManager.default.moveItem(at: staging, to: url)
            }
        } catch {
            try? FileManager.default.removeItem(at: staging)
            throw error
        }
    }
}
