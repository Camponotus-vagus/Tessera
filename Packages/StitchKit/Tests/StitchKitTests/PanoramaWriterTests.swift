import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import StitchKit

@Suite("Panorama files")
struct PanoramaWriterTests {
    /// A 16-bit gradient in Display P3 whose left quarter is transparent, as around an uncropped panorama.
    private func panorama(width: Int = 64, height: Int = 48) -> PanoramaPixels {
        var values = [UInt16](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                values[i] = UInt16(x * 65535 / (width - 1))
                values[i + 1] = UInt16(y * 65535 / (height - 1))
                values[i + 2] = 30000
                values[i + 3] = x < width / 4 ? 0 : 65535
            }
        }
        let data = values.withUnsafeBytes { Data($0) }
        let profile = CGColorSpace(name: CGColorSpace.displayP3)!.copyICCData()! as Data
        return PanoramaPixels(width: width, height: height, bitsPerComponent: 16, pixels: data, iccProfile: profile)
    }

    private func read(_ url: URL) throws -> (image: CGImage, properties: [CFString: Any]) {
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]
        return (image, properties)
    }

    @Test("Every format is written with the right depth, transparency and colour profile")
    func formats() throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let pixels = panorama()
        for format in PanoramaFormat.allCases {
            let url = directory.appendingPathComponent("pano.\(format.fileExtension)")
            try PanoramaWriter.write(pixels, to: url, options: PanoramaExportOptions(format: format, sixteenBit: true))
            let (image, properties) = try read(url)
            #expect(image.width == 64 && image.height == 48)
            #expect(image.bitsPerComponent == (format.supportsSixteenBit ? 16 : 8), "\(format)")
            let hasAlpha = ![.none, .noneSkipLast, .noneSkipFirst].contains(image.alphaInfo)
            #expect(hasAlpha == format.supportsTransparency, "\(format)")
            #expect((properties[kCGImagePropertyProfileName] as? String)?.contains("P3") == true, "\(format)")
        }
    }

    @Test("Writing over an existing file replaces it, and 8-bit output is available for 16-bit formats")
    func replaceAndDepth() throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("pano.png")
        try Data("old".utf8).write(to: url)
        try PanoramaWriter.write(panorama(), to: url, options: PanoramaExportOptions(format: .png, sixteenBit: false))
        let (image, _) = try read(url)
        #expect(image.bitsPerComponent == 8)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["pano.png"])
    }

    @Test("JPEG flattens the transparent border onto white")
    func flattening() throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("pano.jpg")
        try PanoramaWriter.write(panorama(), to: url, options: PanoramaExportOptions(format: .jpeg, quality: 1))
        let (image, _) = try read(url)
        let context = try #require(CGContext(data: nil, width: 64, height: 48, bitsPerComponent: 8, bytesPerRow: 64 * 4,
                                             space: CGColorSpace(name: CGColorSpace.displayP3)!,
                                             bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: 64, height: 48))
        let bytes = try #require(context.data).assumingMemoryBound(to: UInt8.self)
        // Row 24, column 4 lies in the transparent quarter.
        let i = (24 * 64 + 4) * 4
        #expect(bytes[i] > 245 && bytes[i + 1] > 245 && bytes[i + 2] > 245)
    }

    @Test("A buffer smaller than its size is rejected")
    func shortBuffer() {
        let pixels = PanoramaPixels(width: 10, height: 10, bitsPerComponent: 8, pixels: Data(count: 10), iccProfile: nil)
        #expect(throws: StitchError.self) {
            try PanoramaWriter.write(pixels, to: URL(fileURLWithPath: "/tmp/never.png"),
                                     options: PanoramaExportOptions(format: .png))
        }
    }
}
