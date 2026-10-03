import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import StitchKit

@Suite("Photos with more than 8 bits")
struct HighBitDepthTests {
    /// A 16-bit grey TIFF of `image` with its 0-255 range squeezed into `range`, as a phase-contrast camera
    /// leaves its tiles (the NIST stitching tiles span about 3600-5500).
    private func writeNarrowTIFF(_ image: CGImage, range: ClosedRange<Int>, to url: URL) throws {
        let width = image.width, height = image.height
        var grey = [UInt8](repeating: 0, count: width * height)
        let context = try #require(CGContext(data: &grey, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                                             space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        let span = Double(range.upperBound - range.lowerBound)
        var wide = grey.map { UInt16(Double(range.lowerBound) + Double($0) / 255 * span).littleEndian }
        let data = Data(bytes: &wide, count: wide.count * 2)
        let provider = try #require(CGDataProvider(data: data as CFData))
        let narrow = try #require(CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: width * 2,
                                          space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue),
                                          provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.tiff.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, narrow, nil)
        #expect(CGImageDestinationFinalize(destination))
    }

    /// The darkest and brightest of the middle 98% of an image's red values, drawn at 8 bits.
    private func spread(_ image: CGImage) -> (Int, Int) {
        var pixels = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let context = CGContext(data: &pixels, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        let red = stride(from: 0, to: pixels.count, by: 4).map { Int(pixels[$0]) }.sorted()
        return (red[red.count / 100], red[red.count * 99 / 100])
    }

    @Test("Narrow 16-bit tiles are stretched for analysis and display, and they match")
    func narrowTiles() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2200, height: 1100, seed: 51)
        let size = CGSize(width: 1392, height: 1040)
        var urls: [URL] = []
        for (index, origin) in [CGPoint(x: 0, y: 30), CGPoint(x: 760, y: 0)].enumerated() {
            let url = directory.appendingPathComponent("tile\(index).tif")
            try writeNarrowTIFF(Synthetic.tile(of: scene, origin: origin, size: size), range: 3600...5500, to: url)
            urls.append(url)
        }
        let photo = try ImageLoader.describe(url: urls[0], id: 0)
        // Drawn as it is, the tile keeps about seven grey levels; stretched, it fills the scale.
        let analysis = try ImageLoader.analysisImage(for: photo, longSide: 1392)
        let (dark, bright) = spread(analysis)
        #expect(dark < 40 && bright > 215, "analysis copy spans \(dark)-\(bright)")
        let (thumbDark, thumbBright) = spread(try ImageLoader.thumbnail(url: urls[0], longSide: 360))
        #expect(thumbDark < 40 && thumbBright > 215, "thumbnail spans \(thumbDark)-\(thumbBright)")

        var configuration = PipelineConfiguration()
        configuration.mode = .plane
        configuration.sources = [.rootSIFT]
        let report = try await StitchEngine().analyze(urls: urls, configuration: configuration)
        let fit = try #require(report.evidence(0, 1, source: .rootSIFT)?.chosenFit)
        #expect(report.evidence(0, 1, source: .rootSIFT)?.verdict == .verified)
        #expect(abs(fit.transform[2] + 760) < 0.5 && abs(fit.transform[5] - 30) < 0.5, "\(fit.transform)")
    }

    @Test("Only panoramas of narrow high-bit-depth photos are stretched, and only in 8-bit copies")
    func panoramaLevels() throws {
        // 64 x 64 pixels from 4000 to 5023 in every channel, fully covered.
        func panorama(highBitDepth: Bool, scale: Int = 1) -> PanoramaPixels {
            var values: [UInt16] = []
            for i in 0..<(64 * 64) {
                let v = UInt16(4000 + (i % 1024) * scale)
                values += [v, v, v, 65535].map(\.littleEndian)
            }
            var pixels = PanoramaPixels(width: 64, height: 64, bitsPerComponent: 16,
                                        pixels: values.withUnsafeBytes { Data($0) }, iccProfile: nil, opaque: true)
            pixels.highBitDepth = highBitDepth
            return pixels
        }
        let narrow = panorama(highBitDepth: true)
        let levels = try #require(narrow.levels())
        #expect(levels.low <= 4001 && levels.high >= 5020, "\(levels)")
        #expect(panorama(highBitDepth: false).levels() == nil)
        #expect(panorama(highBitDepth: true, scale: 60).levels() == nil)
        let (dark, bright) = spread(try narrow.preview())
        #expect(dark < 10 && bright > 245, "preview spans \(dark)-\(bright)")
        let (exportDark, exportBright) = spread(try narrow.cgImage(bitsPerComponent: 8))
        #expect(exportDark < 10 && exportBright > 245, "8-bit export spans \(exportDark)-\(exportBright)")
        // The 16-bit copy keeps the values.
        let wide = try narrow.cgImage(bitsPerComponent: 16)
        #expect(wide.decode == nil)
    }
}
