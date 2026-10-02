import CoreGraphics
import CoreML
import CStitchCore
import Foundation
import Testing

@testable import StitchKit

@Suite("Native keypoint selection")
struct NativeSelectTests {
    // MARK: Helpers

    /// Runs sc_select_keypoints_native; nil and the message on failure.
    static func select(_ logits: [Float], _ ranker: [Float], width: Int, height: Int, keypoints: Int) -> [Float]? {
        var out = [Float](repeating: .nan, count: keypoints * 2)
        var message = [CChar](repeating: 0, count: 256)
        let status = sc_select_keypoints_native(logits, ranker, Int32(width), Int32(height), Int32(keypoints), &out,
                                                &message, message.count)
        return status == 0 ? out : nil
    }

    /// The selection written out directly (full sort, no bands, no partial selection), with the same
    /// rules for ties and non-finite values as select.cpp.
    static func reference(_ logits: [Float], _ ranker: [Float], width: Int, height: Int, keypoints: Int) -> [Float] {
        func inside(_ x: Int, _ y: Int) -> Bool { x >= 0 && x < width && y >= 0 && y < height }
        var suppressed = [Float](repeating: -.infinity, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let value = logits[y * width + x]
                guard !value.isNaN else { continue }
                var isMaximum = true
                for dy in -1...1 {
                    for dx in -1...1 where inside(x + dx, y + dy) {
                        let other = logits[(y + dy) * width + x + dx]
                        if !other.isNaN && other > value { isMaximum = false }
                    }
                }
                if isMaximum { suppressed[y * width + x] = value }
            }
        }
        let pool = keypoints + 256, start = keypoints - 256
        let order = suppressed.indices.sorted {
            suppressed[$0] > suppressed[$1] || (suppressed[$0] == suppressed[$1] && $0 < $1)
        }.prefix(pool)
        let points: [SIMD2<Float>] = order.map { index in
            let cx = index % width, cy = index / width
            var values: [Float] = []
            for dy in -1...1 {
                for dx in -1...1 {
                    let value = inside(cx + dx, cy + dy) ? logits[(cy + dy) * width + cx + dx] : 0
                    values.append(value.isNaN ? -.infinity : value)
                }
            }
            let maximum = values.max()!
            let weights = values.indices.map { k -> Float in
                if maximum == .infinity { return values[k] == .infinity ? 1 : 0 }
                if maximum == -.infinity { return k == 4 ? 1 : 0 }
                return exp((values[k] - maximum) * 2)
            }
            let sum = weights.reduce(0, +)
            var offset = SIMD2<Float>(0, 0)
            for k in 0..<9 { offset += weights[k] / sum * SIMD2(Float(k % 3 - 1), Float(k / 3 - 1)) }
            return SIMD2(Float(cx), Float(cy)) + offset
        }
        func sample(_ point: SIMD2<Float>) -> Float {
            let x = min(max(point.x, 0), Float(width - 1)), y = min(max(point.y, 0), Float(height - 1))
            let x0 = Int(x.rounded(.down)), y0 = Int(y.rounded(.down))
            let x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1)
            let ax = x - Float(x0), ay = y - Float(y0)
            let top = (1 - ax) * ranker[y0 * width + x0] + ax * ranker[y0 * width + x1]
            let bottom = (1 - ax) * ranker[y1 * width + x0] + ax * ranker[y1 * width + x1]
            return (1 - ay) * top + ay * bottom
        }
        let scores = (0..<512).map { sample(points[start + $0]) }
        let reranked = scores.indices.sorted { a, b in
            if scores[a].isNaN != scores[b].isNaN { return scores[b].isNaN }
            if !scores[a].isNaN && scores[a] != scores[b] { return scores[a] > scores[b] }
            return a < b
        }.prefix(256)
        let selected = Array(points[0..<start]) + reranked.map { points[start + $0] }
        return selected.flatMap { [$0.x + 0.5, $0.y + 0.5] }
    }

    /// Checks a selection against `reference`: the first K - 256 points in the same order, and the same
    /// points overall. Two points that nearly coincide may swap places in the re-ranked block, since
    /// their ranker scores then differ only in the last bits.
    static func expectReference(_ out: [Float], _ logits: [Float], _ ranker: [Float], width: Int, height: Int,
                                keypoints: Int, sourceLocation: SourceLocation = #_sourceLocation) {
        let expected = reference(logits, ranker, width: width, height: height, keypoints: keypoints)
        let prefix = 2 * (keypoints - 256)
        #expect(maximumDifference(Array(out[..<prefix]), Array(expected[..<prefix])) < 1e-5,
                sourceLocation: sourceLocation)
        #expect(multisetDifference(out, expected, tolerance: 1e-5).unmatched == 0, sourceLocation: sourceLocation)
        #expect(maximumDifference(out, expected) < 1e-3, sourceLocation: sourceLocation)
    }

    static func maximumDifference(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return .infinity }
        return zip(a, b).map { $0.isNaN || $1.isNaN ? .infinity : abs($0 - $1) }.max() ?? 0
    }

    /// Pairs every keypoint of `a` with an unused keypoint of `b` within `tolerance` pixels per axis:
    /// the number left without a partner and the largest difference among the pairs.
    static func multisetDifference(_ a: [Float], _ b: [Float], tolerance: Float) -> (unmatched: Int, delta: Float) {
        let (partners, delta) = multisetPairing(a, b, tolerance: tolerance)
        return (partners.filter { $0 == nil }.count + max(0, b.count / 2 - a.count / 2), delta)
    }

    /// The pairing of `multisetDifference`: for each keypoint of `a`, the index of its partner in `b`.
    static func multisetPairing(_ a: [Float], _ b: [Float], tolerance: Float) -> (partners: [Int?], delta: Float) {
        let countB = b.count / 2
        let byX = (0..<countB).sorted { b[2 * $0] < b[2 * $1] }
        var used = [Bool](repeating: false, count: countB)
        var partners: [Int?] = []
        var delta: Float = 0
        for i in 0..<(a.count / 2) {
            let x = a[2 * i], y = a[2 * i + 1]
            var low = 0, high = byX.count
            while low < high {
                let middle = (low + high) / 2
                if b[2 * byX[middle]] < x - tolerance { low = middle + 1 } else { high = middle }
            }
            var best: (index: Int, distance: Float)?
            var cursor = low
            while cursor < byX.count, b[2 * byX[cursor]] <= x + tolerance {
                let j = byX[cursor]
                let distance = max(abs(b[2 * j] - x), abs(b[2 * j + 1] - y))
                if !used[j], distance <= tolerance, distance < (best?.distance ?? .infinity) { best = (j, distance) }
                cursor += 1
            }
            if let best {
                used[best.index] = true
                delta = max(delta, best.distance)
            }
            partners.append(best?.index)
        }
        return (partners, delta)
    }

    /// Indices of the keypoints inside the photo, not on the padding of the canvas (the rule of the
    /// engine, which drops the others).
    static func inside(_ keypoints: [Float], content: PixelSize) -> [Int] {
        (0..<(keypoints.count / 2)).filter {
            keypoints[2 * $0] < Float(content.width) && keypoints[2 * $0 + 1] < Float(content.height)
        }
    }

    static func gather(_ keypoints: [Float], _ indices: [Int]) -> [Float] {
        indices.flatMap { [keypoints[2 * $0], keypoints[2 * $0 + 1]] }
    }

    /// A ramp, so ranker scores follow the position and never tie.
    static func ramp(width: Int, height: Int) -> [Float] {
        (0..<(width * height)).map { Float($0 % width) + Float($0 / width) * Float(width) }
    }

    // MARK: Synthetic maps

    @Test("Known maxima, a plateau and the border neighbours outside the image")
    func knownMaxima() throws {
        let width = 40, height = 30, k = 300
        var logits = [Float](repeating: -10, count: width * height)
        let peaks: [(x: Int, y: Int, value: Float)] = [(20, 15, 9), (5, 5, 8), (33, 22, 7), (10, 25, 6)]
        for peak in peaks { logits[peak.y * width + peak.x] = peak.value }
        // A two-pixel plateau: both pixels are kept, the lower index first.
        logits[10 * width + 30] = 5
        logits[10 * width + 31] = 5
        let ranker = [Float](repeating: 0, count: width * height)
        let out = try #require(Self.select(logits, ranker, width: width, height: height, keypoints: k))

        for (n, peak) in peaks.enumerated() {
            #expect(abs(out[2 * n] - (Float(peak.x) + 0.5)) < 1e-5 && abs(out[2 * n + 1] - (Float(peak.y) + 0.5)) < 1e-5)
        }
        // Each plateau pixel shares its softmax with the other one: half a pixel towards it.
        #expect(abs(out[8] - 30.5 - 0.5) < 1e-5 && abs(out[9] - 10.5) < 1e-5)
        #expect(abs(out[10] - 31.5 + 0.5) < 1e-5 && abs(out[11] - 10.5) < 1e-5)
        // Then the flat background in index order, starting at the corner, where the five neighbours
        // outside the image count as 0 and pull the point outwards by 2/5 of a pixel on each axis.
        #expect(abs(out[12] - 0.1) < 1e-5 && abs(out[13] - 0.1) < 1e-5)
        // Along the top edge three neighbours are outside: the offset is (0, -1).
        #expect(abs(out[14] - 1.5) < 1e-5 && abs(out[15] + 0.5) < 1e-5)
        // A constant ranker keeps the detector order (ties by window index).
        #expect(Self.maximumDifference(out, Self.reference(logits, ranker, width: width, height: height, keypoints: k)) < 1e-5)
        // The neighbours of a peak are suppressed; selected, they would land on the peak.
        #expect((0..<k).filter { abs(out[2 * $0] - 5.5) < 1e-3 && abs(out[2 * $0 + 1] - 5.5) < 1e-3 }.count == 1)
    }

    @Test("The ranker re-orders only the window after the first K - 256 keypoints")
    func boundaryWindow() throws {
        let width = 128, height = 96, k = 512
        var rng = SplitMix64(state: 3)
        let logits = (0..<(width * height)).map { _ in Float.random(in: -4...4, using: &rng) }
        let constant = try #require(Self.select(logits, [Float](repeating: 1, count: width * height), width: width,
                                                height: height, keypoints: k))
        let ramp = try #require(Self.select(logits, Self.ramp(width: width, height: height), width: width,
                                            height: height, keypoints: k))
        #expect(Array(constant[0..<(2 * (k - 256))]) == Array(ramp[0..<(2 * (k - 256))]))
        #expect(Array(constant[(2 * (k - 256))...]) != Array(ramp[(2 * (k - 256))...]))
        // With the ramp the re-ranked block is sorted by y * width + x (clamped to the map), best first.
        let scores = stride(from: 2 * (k - 256), to: 2 * k, by: 2).map { index -> Float in
            let x = min(max(ramp[index] - 0.5, 0), Float(width - 1))
            let y = min(max(ramp[index + 1] - 0.5, 0), Float(height - 1))
            return y * Float(width) + x
        }
        #expect(zip(scores, scores.dropFirst()).allSatisfy { $0 >= $1 - 0.01 })
    }

    /// Map size, K, quantisation step of the logits (exact ties and plateaus), NaN and infinities among
    /// the logits, NaN in the left third of the ranker.
    static let randomCases: [(width: Int, height: Int, k: Int, step: Float?, special: Bool, nanRanker: Bool)] = [
        (64, 48, 1024, nil, false, false),
        (64, 48, 1024, 0.5, false, false),
        (37, 29, 817, nil, true, false),  // the pool is the whole map
        (128, 9, 896, 0.5, true, true),
        (256, 192, 256, nil, false, false),
        (256, 192, 256, 6, false, false),  // plateaus: most bands hold more maxima than the pool
        (256, 192, 1024, 0.5, true, true),
        (256, 192, 256, 1000, false, false),  // constant map: every band is cut to the pool
    ]

    @Test("Random maps, with ties and non-finite values, match the direct implementation",
          arguments: randomCases.indices)
    func randomMaps(index: Int) throws {
        let (width, height, k, step, special, nanRanker) = Self.randomCases[index]
        var rng = SplitMix64(state: UInt64(index) &+ 100)
        var logits = (0..<(width * height)).map { _ in Float.random(in: -4...4, using: &rng) }
        if let step { logits = logits.map { ($0 / step).rounded() * step } }
        if special {
            let values: [Float] = [.nan, .infinity, -.infinity]
            for _ in 0..<(width * height / 20) {
                logits[Int.random(in: 0..<logits.count, using: &rng)] = values[Int.random(in: 0..<3, using: &rng)]
            }
        }
        var ranker = Self.ramp(width: width, height: height)
        if nanRanker {
            for index in ranker.indices where index % width < width / 3 { ranker[index] = .nan }
        }
        let out = try #require(Self.select(logits, ranker, width: width, height: height, keypoints: k))
        #expect(out.allSatisfy { $0.isFinite })
        Self.expectReference(out, logits, ranker, width: width, height: height, keypoints: k)
        // Deterministic: the same bits on a second run.
        let again = try #require(Self.select(logits, ranker, width: width, height: height, keypoints: k))
        #expect(out.map(\.bitPattern) == again.map(\.bitPattern))
    }

    @Test("NaN never wins, +inf shares its weight, and a map without finite maxima still gives K points")
    func nonFinite() throws {
        // A flat background is a plateau of local maxima, so the pool needs no filling.
        let width = 32, height = 24, k = 300
        var logits = [Float](repeating: -10, count: width * height)
        // NaN next to a peak does not suppress it, and is never selected.
        logits[10 * width + 10] = 50
        logits[10 * width + 11] = .nan
        // Two adjacent +inf pixels: both first, each half a pixel towards the other.
        logits[5 * width + 20] = .infinity
        logits[5 * width + 21] = .infinity
        let ranker = Self.ramp(width: width, height: height)
        let out = try #require(Self.select(logits, ranker, width: width, height: height, keypoints: k))
        #expect(out.allSatisfy { $0.isFinite })
        #expect(abs(out[0] - 21) < 1e-5 && abs(out[1] - 5.5) < 1e-5)
        #expect(abs(out[2] - 21) < 1e-5 && abs(out[3] - 5.5) < 1e-5)
        #expect(abs(out[4] - 10.5) < 1e-4 && abs(out[5] - 10.5) < 1e-4)
        // Selected, the NaN pixel would be pulled onto the peak.
        #expect((0..<k).filter { abs(out[2 * $0] - 10.5) < 1e-3 && abs(out[2 * $0 + 1] - 10.5) < 1e-3 }.count == 1)
        Self.expectReference(out, logits, ranker, width: width, height: height, keypoints: k)

        for fill in [Float.nan, -.infinity] {
            let empty = [Float](repeating: fill, count: width * height)
            let points = try #require(Self.select(empty, ranker, width: width, height: height, keypoints: k))
            #expect(points.allSatisfy { $0.isFinite })
            Self.expectReference(points, empty, ranker, width: width, height: height, keypoints: k)
        }
        // A NaN ranker ranks last: with NaN everywhere the window keeps its order.
        let nanRanker = [Float](repeating: .nan, count: width * height)
        let flat = try #require(Self.select(logits, nanRanker, width: width, height: height, keypoints: k))
        #expect(Self.maximumDifference(flat, Self.reference(logits, nanRanker, width: width, height: height,
                                                            keypoints: k)) < 1e-5)
    }

    @Test("Invalid arguments fail with a message")
    func invalidArguments() {
        let width = 32, height = 24
        let map = [Float](repeating: 0, count: width * height)
        var out = [Float](repeating: 0, count: 2 * width * height)
        map.withUnsafeBufferPointer { mapBuffer in
            out.withUnsafeMutableBufferPointer { outBuffer in
                let m = mapBuffer.baseAddress, o = outBuffer.baseAddress
                let cases: [(UnsafePointer<Float>?, UnsafePointer<Float>?, UnsafeMutablePointer<Float>?, Int, Int, Int)] = [
                    (nil, m, o, width, height, 256), (m, nil, o, width, height, 256), (m, m, nil, width, height, 256),
                    (m, m, o, 0, height, 256), (m, m, o, width, -1, 256), (m, m, o, width, height, 255),
                    (m, m, o, width, height, width * height - 255),
                ]
                for (logits, ranker, target, w, h, k) in cases {
                    var message = [CChar](repeating: 0, count: 256)
                    #expect(sc_select_keypoints_native(logits, ranker, Int32(w), Int32(h), Int32(k), target, &message,
                                                       message.count) == 1)
                    #expect(!errorText(message).isEmpty)
                }
                #expect(sc_select_keypoints_native(m, m, Int32(width), Int32(height), Int32(width * height - 256), o,
                                                   nil, 0) == 0)
                // Without an error buffer a failure is still reported.
                #expect(sc_select_keypoints_native(m, m, Int32(width), Int32(height), 10, o, nil, 0) == 1)
            }
        }
    }

    // MARK: Real photos against the ONNX select model

    /// A set with the feature levels and the ONNX select model.
    static var models: LearnedModelSet? {
        guard let models = LearnedModelSet.standard(), models.hasLevelsExtractor(nativeSelection: false) else {
            return nil
        }
        return models
    }

    /// Photo sets in `TESSERA_TESTDATA`, or in TestData/real of the source checkout (not in git): one
    /// folder per set, consecutive photos overlap.
    static var photoSets: [(name: String, urls: [URL])] {
        let root: URL
        if let custom = ProcessInfo.processInfo.environment["TESSERA_TESTDATA"], !custom.isEmpty {
            root = URL(fileURLWithPath: custom, isDirectory: true)
        } else {
            var checkout = URL(fileURLWithPath: #filePath)
            for _ in 0..<5 { checkout.deleteLastPathComponent() }
            root = checkout.appendingPathComponent("TestData/real", isDirectory: true)
        }
        let manager = FileManager.default
        let folders = (try? manager.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return folders.sorted { $0.lastPathComponent < $1.lastPathComponent }.compactMap { folder in
            let files = (try? manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            let urls = files.filter { ["jpg", "jpeg", "heic", "png", "tif", "tiff"].contains($0.pathExtension.lowercased()) }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            return urls.isEmpty ? nil : (folder.lastPathComponent, urls)
        }
    }

    /// A 16:9 band across the middle of a photo, written to `directory`: in the 4:3 canvas a quarter of
    /// the rows (or columns) are padding.
    static func letterboxed(_ url: URL, in directory: URL) throws -> URL {
        let image = try ImageLoader.thumbnail(url: url, longSide: 2048)
        let band = image.width >= image.height
            ? CGRect(x: 0, y: (image.height - image.width * 9 / 16) / 2, width: image.width, height: image.width * 9 / 16)
            : CGRect(x: (image.width - image.height * 9 / 16) / 2, y: 0, width: image.height * 9 / 16, height: image.height)
        let target = directory.appendingPathComponent("letterboxed-\(url.deletingPathExtension().lastPathComponent).png")
        try Synthetic.write(try #require(image.cropping(to: band)), to: target)
        return target
    }

    struct Selected {
        var canvas: PixelSize
        var onnx: [Float]
        var native: [Float]
        var onnxDescriptors: [Float]
        var nativeDescriptors: [Float]
        /// The ONNX keypoints inside the photo.
        var onnxInside: Set<Int>
        /// For each native keypoint inside the photo, its partner among the ONNX keypoints.
        var nativeToONNX: [Int?]
    }

    static func seconds(_ body: () -> Void) -> Double {
        let start = ContinuousClock.now
        body()
        return (ContinuousClock.now - start).seconds
    }

    static func milliseconds(_ seconds: Double) -> String { String(format: "%.2f ms", seconds * 1000) }

    /// LightGlue's input: each image normalised by its own long edge around its centre.
    static func normalised(_ keypoints: [Float], canvas: PixelSize) -> [Float] {
        let half = Float(max(canvas.width, canvas.height)) / 2
        let centre = [Float(canvas.width) / 2, Float(canvas.height) / 2]
        return keypoints.indices.map { (keypoints[$0] - centre[$0 % 2]) / half }
    }

    /// Inside the photo the selections hold the same keypoints. Exactly equal logits may come in another
    /// order (so some places hold a permutation of each other's keypoints), and on the padding of a
    /// letterboxed photo, a plateau, other pixels of the same value may be taken.
    @Test("Same keypoints as the ONNX select model on the Core ML maps of real photos",
          .enabled(if: LearnedModelSet.onnxAvailable && models != nil && !photoSets.isEmpty))
    func matchesONNXOnRealPhotos() async throws {
        let models = try #require(Self.models)
        let k = models.keypoints
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = try #require(Self.photoSets.first?.urls.first)
        let sets = Self.photoSets + [(name: "letterboxed", urls: [try Self.letterboxed(first, in: directory)])]
        var dense: [PixelSize: CoreMLDense] = [:]
        var onnxSelect: [PixelSize: ONNXModel] = [:]
        for canvas in [LearnedModelSet.landscape, LearnedModelSet.portrait] {
            dense[canvas] = try await CoreMLDense(package: models.levels(for: canvas), computeUnits: .cpuAndGPU)
            onnxSelect[canvas] = try ONNXModel(models.select(for: canvas), execution: SC_EXECUTION_CPU, lowMemory: false)
        }
        let head = try DescriptorHead(models.descriptorHead)
        var matchers: [(precision: String, matcher: CoreMLMatcher)] = []
        for precision in ["fp32", "fp16"] where models.hasCoreMLMatcher(precision: precision) {
            matchers.append((precision, try await CoreMLMatcher(package: models.coreMLMatcher(precision: precision),
                                                                computeUnits: .cpuAndGPU)))
        }

        var lines: [String] = []
        var nativeTimes: [Double] = []
        for set in sets {
            var previous: Selected?
            for url in set.urls {
                let name = "\(set.name)/\(url.lastPathComponent)"
                let image = try ImageLoader.describe(url: url, id: 0)
                let canvas = LearnedModelSet.canvas(for: image)
                let fit = min(Double(canvas.width) / Double(image.pixelSize.width),
                              Double(canvas.height) / Double(image.pixelSize.height))
                let longSide = Int((Double(max(image.pixelSize.width, image.pixelSize.height)) * fit).rounded(.down))
                let working = try ImageLoader.planarRGB(for: image, longSide: longSide, canvas: canvas)
                let maps = try #require(dense[canvas]).run(
                    working.planar, canvas: canvas, outputs: ["logits", "ranker", "level1", "level2", "level3", "level4"])
                var message = [CChar](repeating: 0, count: 1024)

                // Both selections on the same maps; the native one 20 times for its timing.
                var onnx: [Float] = []
                var onnxSeconds = 0.0
                var native = [Float](repeating: 0, count: k * 2)
                var times: [Double] = []
                let select = try #require(onnxSelect[canvas])
                try withContiguousFloats(Array(maps[0...1])) { pointers in
                    var result = sc_extraction()
                    defer { sc_extraction_free(&result) }
                    onnxSeconds = Self.seconds {
                        #expect(sc_select_keypoints(select.pointer, pointers[0], pointers[1], Int32(canvas.width),
                                                    Int32(canvas.height), &result, &message, message.count) == 0)
                    }
                    onnx = Array(UnsafeBufferPointer(start: result.keypoints, count: Int(result.keypoint_count) * 2))
                    for _ in 0..<20 {
                        times.append(Self.seconds {
                            #expect(sc_select_keypoints_native(pointers[0], pointers[1], Int32(canvas.width),
                                                               Int32(canvas.height), Int32(k), &native, &message,
                                                               message.count) == 0)
                        })
                    }
                }
                times.sort()
                nativeTimes.append(times[times.count / 2])

                func describe(_ keypoints: [Float]) throws -> [Float] {
                    let levels = Array(maps[2...5])
                    let widths = levels.map { Int32($0.shape[3].intValue) }
                    let heights = levels.map { Int32($0.shape[2].intValue) }
                    var descriptors = [Float](repeating: 0, count: keypoints.count / 2 * head.dimensions)
                    let status = try withContiguousFloats(levels) { pointers in
                        pointers.map { Optional($0) }.withUnsafeBufferPointer { levelPointers in
                            sc_describe(head.pointer, levelPointers.baseAddress, widths, heights, Int32(levels.count),
                                        Int32(levels[0].shape[1].intValue), Int32(canvas.width), Int32(canvas.height),
                                        keypoints, Int32(keypoints.count / 2), &descriptors, &message, message.count)
                        }
                    }
                    #expect(status == 0)
                    return descriptors
                }
                try #require(onnx.count == 2 * k)

                // The keypoints inside the photo, paired as multisets.
                let content = working.size
                let onnxInside = Self.inside(onnx, content: content), nativeInside = Self.inside(native, content: content)
                let (partners, setDelta) = Self.multisetPairing(Self.gather(onnx, onnxInside), Self.gather(native, nativeInside),
                                                                tolerance: 1e-4)
                let unmatched = partners.filter { $0 == nil }.count + max(0, nativeInside.count - onnxInside.count)
                var nativeToONNX = [Int?](repeating: nil, count: k)
                for (index, partner) in partners.enumerated() {
                    if let partner { nativeToONNX[nativeInside[partner]] = onnxInside[index] }
                }
                #expect(unmatched == 0, "\(name): \(unmatched) keypoints inside the photo without a partner within 1e-4 px")

                // Places that hold different keypoints: inside the photo they swap keypoints among themselves.
                let moved = (0..<k).filter {
                    max(abs(onnx[2 * $0] - native[2 * $0]), abs(onnx[2 * $0 + 1] - native[2 * $0 + 1])) > 1e-4
                }
                let onnxSet = Set(onnxInside), nativeSet = Set(nativeInside)
                let swapped = Self.multisetDifference(Self.gather(onnx, moved.filter(onnxSet.contains)),
                                                      Self.gather(native, moved.filter(nativeSet.contains)), tolerance: 1e-4)
                #expect(swapped.unmatched == 0,
                        "\(name): \(swapped.unmatched) of \(moved.count) keypoints in other places are not a permutation")

                // Descriptors of paired keypoints.
                let onnxDescriptors = try describe(onnx), nativeDescriptors = try describe(native)
                let size = head.dimensions
                var descriptorDelta: Float = 0
                for (nativeIndex, onnxIndex) in nativeToONNX.enumerated() {
                    guard let onnxIndex else { continue }
                    descriptorDelta = max(descriptorDelta, Self.maximumDifference(
                        Array(onnxDescriptors[(onnxIndex * size)..<((onnxIndex + 1) * size)]),
                        Array(nativeDescriptors[(nativeIndex * size)..<((nativeIndex + 1) * size)])))
                }
                #expect(descriptorDelta <= 1e-4, "\(name): descriptors differ by \(descriptorDelta)")
                let selected = Selected(canvas: canvas, onnx: onnx, native: native, onnxDescriptors: onnxDescriptors,
                                        nativeDescriptors: nativeDescriptors, onnxInside: onnxSet,
                                        nativeToONNX: nativeToONNX)

                let identical = zip(onnx, native).filter { $0.bitPattern == $1.bitPattern }.count
                var line = "\(name) \(canvas.width)x\(canvas.height), photo \(content.width)x\(content.height): " +
                    "\(onnxInside.count) ONNX and \(nativeInside.count) native keypoints inside, unmatched \(unmatched), " +
                    "set max delta \(setDelta) px, \(moved.count) in other places, bit-identical coordinates " +
                    "\(identical)/\(2 * k), descriptor max delta \(descriptorDelta); native median " +
                    "\(Self.milliseconds(times[times.count / 2])) (min \(Self.milliseconds(times[0]))), " +
                    "ONNX \(Self.milliseconds(onnxSeconds))"

                // LightGlue on the previous photo of the set and this one.
                if let previous {
                    for (precision, matcher) in matchers {
                        func matches(_ a: [Float], _ descriptorsA: [Float], _ b: [Float], _ descriptorsB: [Float]) throws
                            -> [(Int, Int)] {
                            let (partner, confidence) = try matcher.match(
                                keypoints: Self.normalised(a, canvas: previous.canvas) + Self.normalised(b, canvas: canvas),
                                descriptors: descriptorsA + descriptorsB, count: k, descriptorSize: head.dimensions)
                            return (0..<k).filter { confidence[$0] > 0.1 && partner[$0] >= 0 }.map { ($0, Int(partner[$0])) }
                        }
                        // Matches between keypoints inside the photos, the native ones under the ONNX indices.
                        let viaONNX = Set(try matches(previous.onnx, previous.onnxDescriptors, onnx, onnxDescriptors)
                            .filter { previous.onnxInside.contains($0.0) && onnxSet.contains($0.1) }.map { [$0.0, $0.1] })
                        let viaNative = Set(try matches(previous.native, previous.nativeDescriptors, native, nativeDescriptors)
                            .compactMap { pair in
                                previous.nativeToONNX[pair.0].flatMap { a in nativeToONNX[pair.1].map { [a, $0] } }
                            })
                        let differ = viaONNX.symmetricDifference(viaNative).count
                        line += "; LightGlue \(precision) with the previous photo: \(viaONNX.count) vs " +
                            "\(viaNative.count) matches, \(differ) not in both"
                        // fp16 turns the last-bit differences of a few coordinates into a few changed
                        // borderline matches, so only fp32 is held to (nearly) identical matches.
                        if precision == "fp32" {
                            #expect(differ * 200 <= viaONNX.count, "\(name): \(differ) LightGlue matches differ")
                        }
                    }
                }
                lines.append(line)
                previous = selected
            }
        }
        nativeTimes.sort()
        lines.append("native selection, median over \(nativeTimes.count) photos: " +
                     Self.milliseconds(nativeTimes[nativeTimes.count / 2]))
        print(lines.joined(separator: "\n"))
    }

    @Test("LearnedSession selects in C++ unless asked for the ONNX model",
          .enabled(if: LearnedModelSet.onnxAvailable && models != nil))
    func sessionSwitch() async throws {
        let models = try #require(Self.models)
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("scene.png")
        try Synthetic.write(Synthetic.scene(width: 2048, height: 1536, seed: 23), to: url)
        let image = try ImageLoader.describe(url: url, id: 0)
        let canvas = LearnedModelSet.canvas(for: image)
        let working = try ImageLoader.planarRGB(for: image, longSide: 1024, canvas: canvas)

        var onnx = PipelineConfiguration()
        onnx.learnedModels = models
        onnx.extractorBackend = .coreMLGPU
        onnx.nativeKeypointSelection = false
        var native = onnx
        native.nativeKeypointSelection = nil
        let a = try await LearnedSession(models: models, configuration: onnx)
        let b = try await LearnedSession(models: models, configuration: native)
        #expect(a.key != b.key)
        #expect(a.pipeline.contains("levels + select"))
        #expect(b.pipeline.contains("levels + C++ select"))
        let ea = try a.extract(working.planar, canvas: canvas), eb = try b.extract(working.planar, canvas: canvas)
        #expect(ea.keypoints.count == 2 * models.keypoints)
        #expect(Self.maximumDifference(ea.keypoints, eb.keypoints) <= 1e-4)
        #expect(Self.maximumDifference(ea.descriptors, eb.descriptors) <= 1e-4)
    }
}
