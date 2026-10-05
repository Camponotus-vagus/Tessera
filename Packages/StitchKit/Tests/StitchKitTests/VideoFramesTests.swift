import AVFoundation
import CoreGraphics
import Foundation
import Testing

@testable import StitchKit

@Suite("Frames from a video")
struct VideoFramesTests {
    @Test("Frames are chosen a half to a whole step apart, the sharpest of each step")
    func frameSelection() {
        var selector = FrameSelector(step: 1.0 / 3, start: SIMD2(0, 0))
        var kept: [Int] = [0], held: Int?
        // 1/100 of a frame per frame along x; frame 47 is the sharpest of the second step.
        for i in 1...150 {
            let sharpness = i == 47 ? 100.0 : Double(i % 7)
            let decision = selector.feed(i, position: SIMD2(Double(i) / 100, 0), sharpness: sharpness)
            if decision.commitHeld, let h = held { kept.append(h); held = nil }
            if decision.commitCurrent { kept.append(i) } else if decision.keep { held = i }
        }
        if let held { kept.append(held) }
        for (a, b) in zip(kept, kept.dropFirst()) {
            #expect(b - a >= 17 && b - a <= 33, "frames \(a) and \(b)")
        }
        #expect(kept.contains(47))
        // A jump past a whole step takes the frame after it at once.
        var jumpy = FrameSelector(step: 1.0 / 3, start: SIMD2(0, 0))
        #expect(jumpy.feed(1, position: SIMD2(0, 0.5), sharpness: 1).commitCurrent)
    }

    @Test("Phase correlation finds the shift of a textured image")
    func phaseCorrelation() {
        let width = 960, height = 540
        var rng = SplitMix64(state: 7)
        // Smooth random texture: a coarse noise grid upsampled.
        let coarse = (0..<(121 * 69)).map { _ in Double.random(in: 0...255, using: &rng) }
        func texture(_ x: Double, _ y: Double) -> UInt8 {
            let gx = x / 8, gy = y / 8
            let x0 = Int(gx.rounded(.down)), y0 = Int(gy.rounded(.down)), fx = gx - Double(x0), fy = gy - Double(y0)
            func c(_ i: Int, _ j: Int) -> Double { coarse[max(0, min(68, j)) * 121 + max(0, min(120, i))] }
            let v = (1 - fy) * ((1 - fx) * c(x0, y0) + fx * c(x0 + 1, y0)) + fy * ((1 - fx) * c(x0, y0 + 1) + fx * c(x0 + 1, y0 + 1))
            return UInt8(max(0, min(255, v)))
        }
        let shift = SIMD2<Double>(19, -11)
        let a = (0..<(width * height)).map { texture(Double($0 % width), Double($0 / width)) }
        let b = (0..<(width * height)).map { texture(Double($0 % width) - shift.x, Double($0 / width) - shift.y) }
        let correlator = PhaseCorrelator(width: 512, height: 256)
        let (found, peak) = correlator.shift(from: correlator.spectrum(of: a, width: width, height: height),
                                             to: correlator.spectrum(of: b, width: width, height: height))
        let scaled = found * Double(width) / 512
        #expect(abs(scaled.x - shift.x) < 1.5 && abs(scaled.y - shift.y) < 1.5, "found \(scaled)")
        #expect(peak > 0.1)
    }

    @Test("The preferred transform of a track gives the orientation of its frames")
    func orientation() {
        #expect(VideoFrames.orientation(of: .identity) == .up)
        #expect(VideoFrames.orientation(of: CGAffineTransform(rotationAngle: .pi / 2)) == .right)
        #expect(VideoFrames.orientation(of: CGAffineTransform(rotationAngle: .pi)) == .down)
        #expect(VideoFrames.orientation(of: CGAffineTransform(rotationAngle: -.pi / 2)) == .left)
    }

    @Test("A video panning over a texture gives frames a third of a frame apart, named in time order")
    func extraction() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("tessera-frames-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let video = folder.appendingPathComponent("pan.mov")
        // 320 x 180 frames moving 4 px per frame over a 1000 px wide texture: 1/80 of a frame each.
        let width = 320, height = 180, count = 120, speed = 4
        var rng = SplitMix64(state: 3)
        let coarse = (0..<(260 * 50)).map { _ in UInt8.random(in: 0...255, using: &rng) }
        func texture(_ x: Int, _ y: Int) -> UInt8 { coarse[min(49, y / 4) * 260 + min(259, x / 4)] }

        let writer = try AVAssetWriter(outputURL: video, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        #expect(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        for frame in 0..<count {
            while !input.isReadyForMoreMediaData { try await Task.sleep(for: .milliseconds(2)) }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &buffer)
            let pixels = try #require(buffer)
            CVPixelBufferLockBaseAddress(pixels, [])
            let base = CVPixelBufferGetBaseAddress(pixels)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(pixels)
            for y in 0..<height {
                for x in 0..<width {
                    let v = texture(x + frame * speed, y)
                    for c in 0..<3 { base[y * stride + 4 * x + c] = v }
                    base[y * stride + 4 * x + 3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(pixels, [])
            adaptor.append(pixels, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 30))
        }
        input.markAsFinished()
        await writer.finishWriting()
        #expect(writer.status == .completed)

        let frames = try await VideoFrames.extract(video: video, into: folder.appendingPathComponent("frames"))
        let numbers = frames.map { name in
            Int(name.lastPathComponent.firstMatch(of: /_f(\d+)_/)!.1)!
        }
        #expect(numbers.first == 0)
        #expect(numbers == numbers.sorted())
        // A third of a frame is about 27 frames at 1/80 of a frame each; never more apart than that.
        for (a, b) in zip(numbers, numbers.dropFirst()) { #expect(b - a >= 13 && b - a <= 28, "frames \(a) and \(b)") }
        #expect(frames.allSatisfy { $0.pathExtension == "jpg" && FileManager.default.fileExists(atPath: $0.path) })
        // The second call reuses the frames already written.
        #expect(try await VideoFrames.extract(video: video, into: folder.appendingPathComponent("frames")) == frames)
    }
}
