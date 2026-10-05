import Accelerate
import AVFoundation
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Chooses sharp frames from a video swept over a subject, a step of a third of the frame apart, and
/// writes them as JPEG files named after their frame number and time.
public enum VideoFrames {
    /// Distance between chosen frames, as a share of the frame (the larger of the two axes).
    public static let defaultStep = 1.0 / 3

    /// Whether `url` is a video file the app can read.
    public static func isVideo(_ url: URL) -> Bool {
        UTType(filenameExtension: url.pathExtension)?.conforms(to: .movie) == true
    }

    /// The folder in the caches where the frames of `video` go, named after the video and its size and date
    /// so that a changed file gets new frames.
    public static func cacheFolder(for video: URL) -> URL {
        let values = try? video.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        let stamp = "\(values?.fileSize ?? 0)-\(Int(values?.contentModificationDate?.timeIntervalSince1970 ?? 0))"
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return caches.appendingPathComponent("Tessera/Frames", isDirectory: true)
            .appendingPathComponent("\(video.deletingPathExtension().lastPathComponent)-\(stamp)", isDirectory: true)
    }

    /// Reads every frame of `video` once, and writes the chosen ones into `folder` (by default the cache
    /// folder, reused when an earlier extraction finished). `progress` gets "frames" events with the frames
    /// read. Returns the files in time order. Cancelling the calling task stops the extraction.
    public static func extract(
        video: URL, into folder: URL? = nil, step: Double = defaultStep,
        progress: (@Sendable (ProgressEvent) -> Void)? = nil
    ) async throws -> [URL] {
        let folder = folder ?? cacheFolder(for: video)
        let manifest = folder.appendingPathComponent("frames.txt")
        if let listed = try? String(contentsOf: manifest, encoding: .utf8) {
            let files = listed.split(separator: "\n").map { folder.appendingPathComponent(String($0)) }
            if !files.isEmpty, files.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) { return files }
        }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        let asset = AVURLAsset(url: video)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw StitchError.engine(String(format: String(localized: "%@ has no video track"), locale: .current,
                                            video.lastPathComponent))
        }
        let (naturalSize, transform, rate) = try await track.load(.naturalSize, .preferredTransform, .nominalFrameRate)
        let duration = try await asset.load(.duration).seconds
        let expected = max(1, Int((Double(rate) * duration).rounded()))
        let orientation = orientation(of: transform)

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw StitchError.engine(reader.error?.localizedDescription ?? "cannot read \(video.lastPathComponent)")
        }
        defer { if reader.status == .reading { reader.cancelReading() } }

        let stem = video.deletingPathExtension().lastPathComponent
        let digits = max(4, String(expected).count)
        let context = CIContext(options: [.cacheIntermediates: false])
        let analyser = FrameAnalyser(width: Int(naturalSize.width), height: Int(naturalSize.height))
        var selector: FrameSelector?
        var held: (index: Int, time: Double, image: CGImage)?
        var written: [URL] = []

        func render(_ buffer: CVPixelBuffer) throws -> CGImage {
            let image = CIImage(cvPixelBuffer: buffer).oriented(orientation)
            guard let cg = context.createCGImage(image, from: image.extent, format: .RGBA8,
                                                 colorSpace: CGColorSpace(name: CGColorSpace.sRGB)) else {
                throw StitchError.engine("cannot render frame")
            }
            return cg
        }
        func write(_ frame: (index: Int, time: Double, image: CGImage)) throws {
            let number = String(repeating: "0", count: max(0, digits - String(frame.index).count)) + String(frame.index)
            let name = "\(stem)_f\(number)_\(String(format: "%06.2f", locale: Locale(identifier: "en_US_POSIX"), frame.time))s.jpg"
            let url = folder.appendingPathComponent(name)
            guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
            else { throw StitchError.engine("cannot write \(name)") }
            CGImageDestinationAddImage(destination, frame.image, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary)
            guard CGImageDestinationFinalize(destination) else { throw StitchError.engine("cannot write \(name)") }
            written.append(url)
        }

        var index = 0
        // A pool per frame: the decoded frames and the rendered copies are autoreleased.
        func next() throws -> Bool {
            try autoreleasepool {
                guard let sample = output.copyNextSampleBuffer() else { return false }
                try Task.checkCancellation()
                guard let buffer = CMSampleBufferGetImageBuffer(sample) else { return true }
                let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
                let measure = analyser.measure(buffer)
                if var current = selector {
                    let decision = current.feed(index, position: measure.position, sharpness: measure.sharpness)
                    if decision.commitHeld, let frame = held {
                        try write(frame)
                        held = nil
                    }
                    if decision.commitCurrent {
                        try write((index, time, try render(buffer)))
                    } else if decision.keep {
                        held = (index, time, try render(buffer))
                    }
                    selector = current
                } else {
                    // The first frame is always taken.
                    try write((index, time, try render(buffer)))
                    selector = FrameSelector(step: step, start: measure.position)
                }
                index += 1
                if index % 10 == 0 { progress?(ProgressEvent(stage: "frames", completed: min(index, expected), total: expected)) }
                return true
            }
        }
        while try next() {}
        if reader.status == .failed {
            throw StitchError.engine(reader.error?.localizedDescription ?? "cannot read \(video.lastPathComponent)")
        }
        if let frame = held { try write(frame) }
        progress?(ProgressEvent(stage: "frames", completed: expected, total: expected))
        try written.map(\.lastPathComponent).joined(separator: "\n").write(to: manifest, atomically: true, encoding: .utf8)
        return written
    }

    /// The image orientation a track's preferred transform stands for (rotations by quarter turns).
    static func orientation(of transform: CGAffineTransform) -> CGImagePropertyOrientation {
        let degrees = (atan2(Double(transform.b), Double(transform.a)) * 180 / .pi).rounded()
        switch degrees {
        case 90: return .right
        case 180, -180: return .down
        case -90: return .left
        default: return .up
        }
    }
}

/// Decides, frame by frame, which frames of a sweep to keep: after each kept frame (the anchor), the
/// sharpest of the frames between half a step and a step away from it. A frame a step away commits the
/// sharpest found so far, which becomes the next anchor.
struct FrameSelector {
    struct Decision: Equatable {
        /// Write the frame held from earlier before anything else.
        var commitHeld = false
        /// Write this frame now (the motion skipped a whole window).
        var commitCurrent = false
        /// Hold this frame's pixels as the sharpest of the window.
        var keep = false
    }

    let step: Double
    private(set) var anchor: SIMD2<Double>
    private var best: (index: Int, sharpness: Double, position: SIMD2<Double>)?

    init(step: Double, start: SIMD2<Double>) {
        self.step = step
        anchor = start
    }

    /// Distance as a share of the frame: positions are already in frame widths and heights.
    static func distance(_ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        max(abs(a.x - b.x), abs(a.y - b.y))
    }

    mutating func feed(_ index: Int, position: SIMD2<Double>, sharpness: Double) -> Decision {
        var decision = Decision()
        if Self.distance(position, anchor) > step {
            if let best {
                decision.commitHeld = true
                anchor = best.position
                self.best = nil
            }
            if Self.distance(position, anchor) > step {
                decision.commitCurrent = true
                anchor = position
                return decision
            }
        }
        if Self.distance(position, anchor) >= step / 2, sharpness > best?.sharpness ?? -.infinity {
            best = (index, sharpness, position)
            decision.keep = true
        }
        return decision
    }
}

/// Sharpness and position of each frame of a video, from its luma plane.
final class FrameAnalyser {
    /// Size of the image the sharpness is measured on (long side), smaller videos at their own size.
    static let sharpnessSide = 960
    /// Phase correlation window.
    static let correlationWidth = 512, correlationHeight = 256

    let width: Int, height: Int
    private let sharpSize: (width: Int, height: Int)
    private let correlator = PhaseCorrelator(width: correlationWidth, height: correlationHeight)
    private var previous: PhaseCorrelator.Spectrum?
    private var velocity = SIMD2<Double>(0, 0)
    private(set) var position = SIMD2<Double>(0, 0)

    init(width: Int, height: Int) {
        self.width = max(1, width)
        self.height = max(1, height)
        let scale = min(1, Double(Self.sharpnessSide) / Double(max(self.width, self.height)))
        sharpSize = (max(8, Int(Double(self.width) * scale)), max(8, Int(Double(self.height) * scale)))
    }

    /// Sharpness (variance of the Laplacian) and position, in frame widths and heights from the first frame.
    func measure(_ buffer: CVPixelBuffer) -> (sharpness: Double, position: SIMD2<Double>) {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let planar = CVPixelBufferIsPlanar(buffer)
        guard let base = planar ? CVPixelBufferGetBaseAddressOfPlane(buffer, 0) : CVPixelBufferGetBaseAddress(buffer)
        else { return (0, position) }
        var luma = vImage_Buffer(
            data: base, height: vImagePixelCount(planar ? CVPixelBufferGetHeightOfPlane(buffer, 0) : CVPixelBufferGetHeight(buffer)),
            width: vImagePixelCount(planar ? CVPixelBufferGetWidthOfPlane(buffer, 0) : CVPixelBufferGetWidth(buffer)),
            rowBytes: planar ? CVPixelBufferGetBytesPerRowOfPlane(buffer, 0) : CVPixelBufferGetBytesPerRow(buffer))
        var small = [UInt8](repeating: 0, count: sharpSize.width * sharpSize.height)
        small.withUnsafeMutableBytes { raw in
            var target = vImage_Buffer(data: raw.baseAddress, height: vImagePixelCount(sharpSize.height),
                                       width: vImagePixelCount(sharpSize.width), rowBytes: sharpSize.width)
            _ = vImageScale_Planar8(&luma, &target, nil, vImage_Flags(kvImageNoFlags))
        }
        let sharpness = Self.laplacianVariance(small, width: sharpSize.width, height: sharpSize.height)

        let spectrum = correlator.spectrum(of: small, width: sharpSize.width, height: sharpSize.height)
        if let previous {
            // A weak peak (blur, a featureless stretch) keeps the motion of the frame before.
            let (shift, peak) = correlator.shift(from: previous, to: spectrum)
            if peak >= 0.02 {
                // The window spans the frame's width and as many rows as the same scale gives it.
                let window = Double(Self.correlationWidth)
                velocity = SIMD2(shift.x / window, shift.y / (window * Double(sharpSize.height) / Double(sharpSize.width)))
            }
            position += velocity
        }
        previous = spectrum
        return (sharpness, position)
    }

    static func laplacianVariance(_ pixels: [UInt8], width: Int, height: Int) -> Double {
        guard width > 2, height > 2 else { return 0 }
        var sum = 0.0, squares = 0.0
        pixels.withUnsafeBufferPointer { p in
            for y in 1..<(height - 1) {
                var rowSum = 0, rowSquares = 0
                let row = y * width
                for x in 1..<(width - 1) {
                    let centre = Int(p[row + x])
                    let value = Int(p[row + x - 1]) + Int(p[row + x + 1]) + Int(p[row + x - width]) + Int(p[row + x + width]) - 4 * centre
                    rowSum += value
                    rowSquares += value * value
                }
                sum += Double(rowSum)
                squares += Double(rowSquares)
            }
        }
        let count = Double((width - 2) * (height - 2))
        let mean = sum / count
        return squares / count - mean * mean
    }
}

/// Translation between two images by phase correlation, on a fixed power-of-two window.
final class PhaseCorrelator {
    struct Spectrum {
        var real: [Float]
        var imaginary: [Float]
    }

    let width: Int, height: Int
    private let log2Width: vDSP_Length, log2Height: vDSP_Length
    private let setup: FFTSetup
    private let window: [Float]

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        log2Width = vDSP_Length(log2(Double(width)).rounded())
        log2Height = vDSP_Length(log2(Double(height)).rounded())
        setup = vDSP_create_fftsetup(max(log2Width, log2Height), FFTRadix(kFFTRadix2))!
        // Hann window, so that the frame edges do not correlate with each other.
        let hx = (0..<width).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(width - 1)) }
        let hy = (0..<height).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(height - 1)) }
        window = (0..<(width * height)).map { hx[$0 % width] * hy[$0 / width] }
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    /// Spectrum of `pixels` (`width` x `height` grey), resampled to the window width and cut to its height
    /// about the centre (padded with the mean when shorter).
    func spectrum(of pixels: [UInt8], width sourceWidth: Int, height sourceHeight: Int) -> Spectrum {
        let scale = Double(sourceWidth) / Double(width)
        let rows = Int(Double(sourceHeight) / scale)
        let top = (rows - height) / 2
        var real = [Float](repeating: 0, count: width * height)
        var mean: Float = 0
        for y in 0..<height {
            let row = y + top
            guard row >= 0, row < rows else { continue }
            let sy = min(sourceHeight - 1, Int((Double(row) + 0.5) * scale))
            for x in 0..<width {
                let sx = min(sourceWidth - 1, Int((Double(x) + 0.5) * scale))
                let value = Float(pixels[sy * sourceWidth + sx])
                real[y * width + x] = value
                mean += value
            }
        }
        mean /= Float(width * height)
        for i in real.indices { real[i] = (real[i] - mean) * window[i] }
        var imaginary = [Float](repeating: 0, count: width * height)
        transform(&real, &imaginary, direction: FFTDirection(kFFTDirection_Forward))
        return Spectrum(real: real, imaginary: imaginary)
    }

    /// How far the content of `b` lies from where it is in `a`, in window pixels (x right, y down), and the
    /// height of the correlation peak (1 for identical images).
    func shift(from a: Spectrum, to b: Spectrum) -> (SIMD2<Double>, Double) {
        let n = width * height
        var real = [Float](repeating: 0, count: n), imaginary = [Float](repeating: 0, count: n)
        for i in 0..<n {
            // conj(A) * B, normalised to unit magnitude.
            let r = a.real[i] * b.real[i] + a.imaginary[i] * b.imaginary[i]
            let m = a.real[i] * b.imaginary[i] - a.imaginary[i] * b.real[i]
            let magnitude = (r * r + m * m).squareRoot()
            if magnitude > 1e-12 {
                real[i] = r / magnitude
                imaginary[i] = m / magnitude
            }
        }
        transform(&real, &imaginary, direction: FFTDirection(kFFTDirection_Inverse))
        var peak = 0
        for i in 1..<n where real[i] > real[peak] { peak = i }
        let px = peak % width, py = peak / width
        func value(_ x: Int, _ y: Int) -> Double {
            Double(real[((y + height) % height) * width + (x + width) % width])
        }
        // Parabola through the peak and its neighbours on each axis.
        func refine(_ left: Double, _ centre: Double, _ right: Double) -> Double {
            let denominator = left - 2 * centre + right
            return abs(denominator) > 1e-12 ? max(-0.5, min(0.5, 0.5 * (left - right) / denominator)) : 0
        }
        var dx = Double(px) + refine(value(px - 1, py), value(px, py), value(px + 1, py))
        var dy = Double(py) + refine(value(px, py - 1), value(px, py), value(px, py + 1))
        if dx > Double(width) / 2 { dx -= Double(width) }
        if dy > Double(height) / 2 { dy -= Double(height) }
        return (SIMD2(dx, dy), Double(real[peak]) / Double(n))
    }

    private func transform(_ real: inout [Float], _ imaginary: inout [Float], direction: FFTDirection) {
        real.withUnsafeMutableBufferPointer { r in
            imaginary.withUnsafeMutableBufferPointer { i in
                var split = DSPSplitComplex(realp: r.baseAddress!, imagp: i.baseAddress!)
                vDSP_fft2d_zip(setup, &split, 1, 0, log2Width, log2Height, direction)
            }
        }
    }
}
