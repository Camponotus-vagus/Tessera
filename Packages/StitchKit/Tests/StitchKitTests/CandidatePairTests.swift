import CStitchCore
import CoreGraphics
import Foundation
import Testing

@testable import StitchKit

@Suite("Candidate pairs and fast matching")
struct CandidatePairTests {
    /// Brute-force reference for nearest neighbours with ratio test and mutual check.
    private func bruteForce(_ a: [[Float]], _ b: [[Float]], ratio: Float) -> Set<[Int]> {
        func distance(_ x: [Float], _ y: [Float]) -> Float { zip(x, y).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }.squareRoot() }
        let backward = b.map { y in a.indices.min { distance(a[$0], y) < distance(a[$1], y) }! }
        var result = Set<[Int]>()
        for (i, x) in a.enumerated() {
            let order = b.indices.sorted { distance(x, b[$0]) < distance(x, b[$1]) }
            let d1 = distance(x, b[order[0]]), d2 = distance(x, b[order[1]])
            if d1 < ratio * d2, backward[order[0]] == i { result.insert([i, order[0]]) }
        }
        return result
    }

    @Test("Matching through a matrix product equals brute force")
    func gemmMatchesBruteForce() {
        var rng = SplitMix64(state: 42)
        func descriptor() -> [Float] { (0..<32).map { _ in Float.random(in: -1...1, using: &rng) } }
        let a = (0..<160).map { _ in descriptor() }
        // B shares 60 noisy copies of A's descriptors and adds unrelated ones.
        var b = (0..<60).map { i in a[i * 2].map { $0 + Float.random(in: -0.05...0.05, using: &rng) } }
        b += (0..<100).map { _ in descriptor() }
        let flatA = a.flatMap { $0 }, flatB = b.flatMap { $0 }
        var raw: UnsafeMutablePointer<sc_match>?
        var message = [CChar](repeating: 0, count: 256)
        let count = sc_match_raw(flatA, Int32(a.count), flatB, Int32(b.count), 32, 0.8, 1, &raw, &message, message.count)
        defer { sc_free(raw) }
        let fast = Set((0..<Int(count)).map { [Int(raw![$0].index_a), Int(raw![$0].index_b)] })
        #expect(fast == bruteForce(a, b, ratio: 0.8))
        #expect(fast.count >= 55)
    }

    private func image(_ id: Int, minute: Int) -> SourceImage {
        SourceImage(id: id, url: URL(fileURLWithPath: "/tmp/\(id).jpg"), name: "\(id).jpg",
                    pixelSize: PixelSize(width: 4000, height: 3000),
                    captureDate: Date(timeIntervalSince1970: Double(minute * 60)), focalLength35mm: nil)
    }

    @Test("Candidates always connect every photo and keep consecutive shots")
    func proposalIsConnected() {
        let images = (0..<8).map { image($0, minute: $0) }
        // Affinity only between a few far-apart photos; most pairs have none.
        let affinity: [PairProposal.Key: Int] = [PairProposal.Key(0, 5): 40, PairProposal.Key(2, 7): 30]
        let candidates = PairProposal.propose(images: images, affinity: affinity, neighbours: 2)
        let keys = candidates.map { PairProposal.Key($0.a, $0.b) }
        #expect(PairProposal.components(images.map(\.id), edges: keys).count == 1)
        for id in 0..<7 { #expect(keys.contains(PairProposal.Key(id, id + 1))) }
        #expect(keys.contains(PairProposal.Key(0, 5)))
        #expect(candidates.count < 28)
    }

    @Test("Bridges are untried pairs between groups, best affinity first")
    func bridgesBetweenGroups() {
        let groups: [Set<Int>] = [[0, 1, 2], [3, 4]]
        let affinity: [PairProposal.Key: Int] = [PairProposal.Key(1, 3): 9, PairProposal.Key(2, 4): 20,
                                                 PairProposal.Key(0, 3): 15]
        let tried: Set<PairProposal.Key> = [PairProposal.Key(2, 4)]
        let bridges = PairProposal.bridges(groups: groups, affinity: affinity, tried: tried, perGroupPair: 2, limit: 10)
        #expect(bridges.map { [$0.a, $0.b] } == [[0, 3], [1, 3]])
        #expect(bridges.allSatisfy { $0.reasons == [.bridge] })
    }

    @Test("Six plane tiles analysed through candidate pairs end up in one group")
    func candidatesOnSyntheticTiles() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 4200, height: 1400, seed: 13)
        let size = CGSize(width: 1200, height: 900)
        var urls: [URL] = []
        for index in 0..<6 {
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: CGPoint(x: index * 580, y: (index % 2) * 120),
                                               size: size), to: url)
            urls.append(url)
        }
        var configuration = PipelineConfiguration()
        configuration.mode = .plane
        configuration.sources = [.rootSIFT]
        let report = try await StitchEngine().analyze(urls: urls, configuration: configuration)
        let candidates = try #require(report.candidates)
        #expect(candidates.count < 15)
        #expect(report.graph.components.first?.count == 6)
    }
}
