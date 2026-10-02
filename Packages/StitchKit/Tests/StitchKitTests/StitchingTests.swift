import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import StitchKit

@Suite("Stitching")
struct StitchingTests {
    /// 8-bit RGB values of `image` drawn into sRGB, top row first.
    private func rgb(_ image: CGImage) -> [UInt8] {
        let width = image.width, height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &pixels, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixels
    }

    /// PSNR of `panorama` against `scene` cropped at `origin`, over pixels the panorama covers.
    private func psnr(_ panorama: Panorama, scene: CGImage, origin: CGPoint) throws -> Double {
        let image = try panorama.pixels.cgImage(bitsPerComponent: 8)
        let region = try #require(scene.cropping(to: CGRect(origin: origin, size: CGSize(width: image.width, height: image.height))))
        let a = rgb(image), b = rgb(region)
        let alpha = panorama.pixels.pixels.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        var squared = 0.0, count = 0
        for i in 0..<(image.width * image.height) where alpha[4 * i + 3] == 65535 {
            for c in 0..<3 {
                let d = Double(a[4 * i + c]) - Double(b[4 * i + c])
                squared += d * d
            }
            count += 3
        }
        return 10 * log10(255 * 255 / max(squared / Double(max(count, 1)), 1e-9))
    }

    @Test("Four plane tiles give back the scene they were cut from")
    func planeMosaic() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2400, height: 1700, seed: 5)
        let size = CGSize(width: 1200, height: 900)
        let origins = [CGPoint(x: 100, y: 80), CGPoint(x: 900, y: 80), CGPoint(x: 100, y: 700), CGPoint(x: 900, y: 700)]
        var urls: [URL] = []
        for (index, origin) in origins.enumerated() {
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: origin, size: size), to: url)
            urls.append(url)
        }
        var configuration = PipelineConfiguration()
        configuration.mode = .plane
        configuration.sources = [.rootSIFT]
        configuration.pairSelection = .all
        let engine = StitchEngine()
        let report = try await engine.analyze(urls: urls, configuration: configuration)
        for request in [StitchRequest(), StitchRequest.originalPixels] {
            let panorama = try await engine.stitch(report, request: request)
            #expect(panorama.model == .translation)
            #expect(panorama.alignmentError < 0.5)
            // The tiles cover 100...2100 x 80...1600 of the scene exactly.
            #expect(abs(panorama.size.width - 2000) <= 1 && abs(panorama.size.height - 1520) <= 1, "\(panorama.size)")
            #expect(panorama.pixels.opaque)
            let quality = try psnr(panorama, scene: scene, origin: CGPoint(x: 100, y: 80))
            #expect(quality > 32, "PSNR \(quality) dB with \(request.blending)")
        }
    }

    @Test("A photo stored rotated with an EXIF orientation is placed like its upright copy")
    func exifOrientation() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2200, height: 1100, seed: 9)
        let left = Synthetic.tile(of: scene, origin: CGPoint(x: 50, y: 60), size: CGSize(width: 1200, height: 900))
        let right = Synthetic.tile(of: scene, origin: CGPoint(x: 850, y: 120), size: CGSize(width: 1200, height: 900))
        let a = directory.appendingPathComponent("a.png"), upright = directory.appendingPathComponent("b.png")
        try Synthetic.write(left, to: a)
        try Synthetic.write(right, to: upright)
        // Stored turned a quarter turn counter-clockwise, tagged 6: viewers rotate it back clockwise.
        let stored = directory.appendingPathComponent("b-rotated.jpg")
        let context = try #require(CGContext(data: nil, width: 900, height: 1200, bitsPerComponent: 8, bytesPerRow: 0,
                                             space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                             bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.translateBy(x: 900, y: 0)
        context.rotate(by: .pi / 2)
        context.draw(right, in: CGRect(x: 0, y: 0, width: 1200, height: 900))
        let rotated = try #require(context.makeImage())
        let destination = try #require(CGImageDestinationCreateWithURL(stored as CFURL, UTType.jpeg.identifier as CFString, 1, nil))
        // ImageIO writes the orientation of a JPEG only from the TIFF dictionary.
        CGImageDestinationAddImage(destination, rotated, [
            kCGImagePropertyTIFFDictionary: [kCGImagePropertyTIFFOrientation: 6],
            kCGImageDestinationLossyCompressionQuality: 1.0,
        ] as CFDictionary)
        #expect(CGImageDestinationFinalize(destination))

        var configuration = PipelineConfiguration()
        configuration.mode = .plane
        configuration.sources = [.rootSIFT]
        let engine = StitchEngine()
        var sizes: [PixelSize] = []
        for second in [upright, stored] {
            let report = try await engine.analyze(urls: [a, second], configuration: configuration)
            #expect(report.images[1].pixelSize == PixelSize(width: 1200, height: 900))
            let panorama = try await engine.stitch(report)
            #expect(panorama.model == .translation)
            #expect(panorama.alignmentError < 1)
            sizes.append(panorama.size)
            let quality = try psnr(panorama, scene: scene, origin: CGPoint(x: 50, y: 60))
            #expect(quality > 28, "PSNR \(quality) dB for \(second.lastPathComponent)")
        }
        #expect(sizes[0] == sizes[1])
    }

    @Test("Stitching needs two joined photos")
    func nothingToStitch() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let urls = try (0..<2).map { index -> URL in
            let url = directory.appendingPathComponent("unrelated\(index).png")
            try Synthetic.write(Synthetic.scene(width: 800, height: 600, seed: UInt64(30 + index)), to: url)
            return url
        }
        let engine = StitchEngine()
        let report = try await engine.analyze(urls: urls, configuration: PipelineConfiguration())
        await #expect(throws: StitchError.self) { _ = try await engine.stitch(report) }
    }

    @Test("Suggested names keep the first photo and the distinct end of the last")
    func names() {
        #expect(Panorama.suggestedName(first: "IMG_5355.jpeg", last: "IMG_5360.jpeg") == "IMG_5355-5360")
        #expect(Panorama.suggestedName(first: "tile.png", last: "tile.png") == "tile")
        #expect(Panorama.suggestedName(first: "a1.jpg", last: "b2.jpg") == "a1-b2")
    }
}
