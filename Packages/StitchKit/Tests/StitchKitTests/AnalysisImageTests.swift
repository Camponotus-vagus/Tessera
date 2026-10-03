import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import StitchKit

@Suite("Analysis images")
struct AnalysisImageTests {
    /// RGBA bytes of `image` drawn at its own size in sRGB.
    private func bytes(_ image: CGImage) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let context = CGContext(data: &out, width: image.width, height: image.height, bitsPerComponent: 8,
                                bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return out
    }

    @Test("Every EXIF orientation is turned upright as ImageIO turns it, at full and at reduced size",
          arguments: Array(1...8), [360, 180])
    func orientations(orientation: Int, longSide: Int) throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // An asymmetric picture, stored as is and tagged with the orientation a viewer applies.
        let stored = Synthetic.scene(width: 360, height: 240, seed: UInt64(orientation))
        let url = directory.appendingPathComponent("o\(orientation).jpg")
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, stored, [
            kCGImagePropertyTIFFDictionary: [kCGImagePropertyTIFFOrientation: orientation],
            kCGImageDestinationLossyCompressionQuality: 1.0,
        ] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))

        let photo = try ImageLoader.describe(url: url, id: 0)
        let ours = try ImageLoader.analysisImage(for: photo, longSide: longSide)
        let reference = try ImageLoader.thumbnail(url: url, longSide: longSide)
        #expect(ours.width == reference.width && ours.height == reference.height)
        #expect(ours.width * 360 == photo.pixelSize.width * longSide && ours.height * 360 == photo.pixelSize.height * longSide)
        let a = bytes(ours), b = bytes(reference)
        var difference = 0
        for i in a.indices where i % 4 != 3 { difference += abs(Int(a[i]) - Int(b[i])) }
        let mean = Double(difference) / Double(a.count / 4 * 3)
        #expect(mean < 2, "mean difference \(mean) for orientation \(orientation)")
    }

    @Test("Analysis copies follow the pixel-centre map, through the reduced decoding and the area averaging",
          arguments: [UTType.png, UTType.jpeg])
    func pixelCentreMap(type: UTType) throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // Vertical edges between columns e - 1 and e, that is at pixel-centre coordinate e - 0.5.
        let width = 4032, height = 3024, edges = [500, 2016, 3500]
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for x in 0..<width {
            let bright = edges.filter { x >= $0 }.count % 2 == 1
            for y in 0..<height { for c in 0..<3 { pixels[(y * width + x) * 4 + c] = bright ? 230 : 25 } }
        }
        let space = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let image = try #require(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                                         space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                         provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let url = directory.appendingPathComponent("edges").appendingPathExtension(for: type)
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 1.0] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))
        let photo = try ImageLoader.describe(url: url, id: 0)
        for longSide in [1414, 1024] {
            let reduced = try ImageLoader.analysisImage(for: photo, longSide: longSide)
            let w = reduced.width, row = bytes(reduced)[(reduced.height / 2 * w * 4)..<((reduced.height / 2 + 1) * w * 4)]
            let values = stride(from: row.startIndex, to: row.endIndex, by: 4).map { Double(row[$0]) }
            var found: [Double] = []
            for x in 0..<(w - 1) where (values[x] - 127.5) * (values[x + 1] - 127.5) < 0 {
                found.append(Double(x) + (127.5 - values[x]) / (values[x + 1] - values[x]))
            }
            try #require(found.count == edges.count, "\(found)")
            let r = Double(w) / Double(width)
            for (edge, position) in zip(edges, found) {
                // ImageIO thumbnails are off by 0.11 to 0.25 pixels here; JPEG ringing alone moves an edge by 0.08.
                #expect(abs(position - (Double(edge) * r - 0.5)) < 0.1, "\(type) \(longSide): edge \(edge) at \(position)")
            }
        }
    }

    @Test("Photos reduced for analysis keep their distances: no bias from resampling or from RaCo's upsampling")
    func noScaleBias() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        // Two 12-megapixel tiles exactly 3000 px apart, the size of an iPhone photo.
        let scene = Synthetic.scene(width: 7032, height: 3064, seed: 21)
        let size = CGSize(width: 4032, height: 3024)
        let origins = [CGPoint(x: 0, y: 0), CGPoint(x: 3000, y: 40)]
        var urls: [URL] = []
        for (index, origin) in origins.enumerated() {
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: origin, size: size), to: url)
            urls.append(url)
        }
        // A third tile shifted mostly downwards, for the vertical axis.
        let tall = Synthetic.scene(width: 4100, height: 5300, seed: 22)
        var vertical: [URL] = []
        for (index, origin) in [CGPoint(x: 0, y: 0), CGPoint(x: 40, y: 2200)].enumerated() {
            let url = directory.appendingPathComponent("vertical\(index).png")
            try Synthetic.write(Synthetic.tile(of: tall, origin: origin, size: size), to: url)
            vertical.append(url)
        }
        var configuration = PipelineConfiguration()
        configuration.mode = .plane
        // The learned matcher too when its models are installed.
        configuration.learnedModels = LearnedModelSet.standard()
        configuration.sources = configuration.learnedModels != nil ? [.rootSIFT, .racoLightGlue] : [.rootSIFT]
        let report = try await StitchEngine().analyze(urls: urls, configuration: configuration)
        for source in configuration.sources {
            let fit = try #require(report.evidence(0, 1, source: source)?.chosenFit, "\(source)")
            // A point at p in tile 0 sits at p - (3000, 40) in tile 1. With ImageIO thumbnails RootSIFT gave
            // 2999.40; without the correction of RaCo's stretch the learned matcher gives 3001.2.
            let tolerance = source == .rootSIFT ? 0.1 : 0.5
            #expect(abs(fit.transform[2] + 3000) < tolerance, "\(source) tx \(fit.transform[2])")
            #expect(abs(fit.transform[5] + 40) < tolerance, "\(source) ty \(fit.transform[5])")
        }
        let downwards = try await StitchEngine().analyze(urls: vertical, configuration: configuration)
        for source in configuration.sources {
            let fit = try #require(downwards.evidence(0, 1, source: source)?.chosenFit, "\(source)")
            let tolerance = source == .rootSIFT ? 0.1 : 0.5
            #expect(abs(fit.transform[2] + 40) < tolerance, "\(source) tx \(fit.transform[2])")
            #expect(abs(fit.transform[5] + 2200) < tolerance, "\(source) ty \(fit.transform[5])")
        }
    }
}
