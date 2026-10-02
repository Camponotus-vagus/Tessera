import CStitchCore
import Foundation
import Testing

@Suite("Panorama crop")
struct CropTests {
    /// Area of the largest all-valid rectangle by trying every rectangle (small masks only).
    private func bruteForce(_ mask: [UInt8], width: Int, height: Int) -> Int {
        var best = 0
        for top in 0..<height {
            for left in 0..<width {
                for bottom in top..<height {
                    for right in left..<width {
                        var full = true
                        check: for y in top...bottom {
                            for x in left...right where mask[y * width + x] == 0 {
                                full = false
                                break check
                            }
                        }
                        if full { best = max(best, (bottom - top + 1) * (right - left + 1)) }
                    }
                }
            }
        }
        return best
    }

    @Test("The rectangle is valid and as large as an exhaustive search finds")
    func randomMasks() {
        var rng = SplitMix64(state: 11)
        for _ in 0..<60 {
            let width = Int.random(in: 1...14, using: &rng), height = Int.random(in: 1...10, using: &rng)
            let density = Double.random(in: 0.3...0.95, using: &rng)
            let mask = (0..<(width * height)).map { _ in Double.random(in: 0..<1, using: &rng) < density ? UInt8(255) : 0 }
            var rect = [Int32](repeating: -1, count: 4)
            let found = sc_largest_rectangle(mask, Int32(width), Int32(height), Int32(width), &rect)
            let expected = bruteForce(mask, width: width, height: height)
            #expect((found == 1) == (expected > 0))
            guard found == 1 else { continue }
            #expect(Int(rect[2] * rect[3]) == expected)
            for y in Int(rect[1])..<Int(rect[1] + rect[3]) {
                for x in Int(rect[0])..<Int(rect[0] + rect[2]) { #expect(mask[y * width + x] != 0) }
            }
        }
    }

    @Test("Padding bytes at the end of each row are ignored, and an empty mask has no rectangle")
    func strideAndEmpty() {
        let width = 5, height = 3, stride = 8
        var mask = [UInt8](repeating: 0, count: stride * height)
        for y in 0..<height { for x in 0..<width { mask[y * stride + x] = 1 } }
        var rect = [Int32](repeating: 0, count: 4)
        #expect(sc_largest_rectangle(mask, Int32(width), Int32(height), Int32(stride), &rect) == 1)
        #expect(rect == [0, 0, 5, 3])
        #expect(sc_largest_rectangle([UInt8](repeating: 0, count: 15), 5, 3, 5, &rect) == 0)
    }
}

extension CropTests {
    /// A panorama-like mask: a band with wavy top and bottom edges and slanted ends.
    private func panoramaMask(width: Int, height: Int) -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let top = 0.06 * Double(height) * (1 + sin(Double(x) / Double(width) * 9))
                let bottom = Double(height) - 0.05 * Double(height) * (1 + cos(Double(x) / Double(width) * 7))
                let left = 0.004 * Double(width) * (1 + Double(y) / Double(height))
                let right = Double(width) - 0.003 * Double(width) * (2 - Double(y) / Double(height))
                if Double(y) > top && Double(y) < bottom && Double(x) > left && Double(x) < right { mask[y * width + x] = 255 }
            }
        }
        return mask
    }

    @Test("The panorama crop is the exact largest rectangle on wide masks with ragged edges")
    func panoramaCrop() {
        for (width, height) in [(3000, 800), (7000, 1600)] {
            let mask = panoramaMask(width: width, height: height)
            var exact = [Int32](repeating: 0, count: 4), crop = [Int32](repeating: 0, count: 4)
            #expect(sc_largest_rectangle(mask, Int32(width), Int32(height), Int32(width), &exact) == 1)
            #expect(sc_inscribed_rectangle(mask, Int32(width), Int32(height), Int32(width), &crop) == 1)
            #expect(crop == exact)
        }
    }
}
