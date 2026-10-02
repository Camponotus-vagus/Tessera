import CoreGraphics
import Foundation
import Testing

@testable import StitchKit

@Suite("Learned extractor")
struct DescriptorHeadTests {
    static var models: LearnedModelSet? {
        guard let models = LearnedModelSet.standard(), models.hasLevelsExtractor,
              models.hasSplitExtractor(precision: "fp32") else { return nil }
        return models
    }

    @Test("C++ descriptor head on feature levels equals the ONNX head on the full map",
          .enabled(if: LearnedModelSet.onnxAvailable && models != nil))
    func fastHeadMatchesONNXHead() async throws {
        let models = try #require(Self.models)
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("scene.png")
        try Synthetic.write(Synthetic.scene(width: 2048, height: 1536, seed: 17), to: url)
        let image = try ImageLoader.describe(url: url, id: 0)
        let canvas = LearnedModelSet.canvas(for: image)
        let working = try ImageLoader.planarRGB(for: image, longSide: 1024, canvas: canvas)

        var fast = PipelineConfiguration()
        fast.learnedModels = models
        fast.extractorBackend = .coreMLGPU
        var reference = fast
        reference.fastDescriptorHead = false
        let a = try await LearnedSession(models: models, configuration: fast).extract(working.planar, canvas: canvas)
        let b = try await LearnedSession(models: models, configuration: reference).extract(working.planar, canvas: canvas)

        #expect(a.size == b.size)
        #expect(a.keypoints.count == b.keypoints.count)
        let keypointDelta = zip(a.keypoints, b.keypoints).map { abs($0 - $1) }.max() ?? .infinity
        #expect(keypointDelta < 1e-3, "keypoints differ by \(keypointDelta) px")
        let descriptorDelta = zip(a.descriptors, b.descriptors).map { abs($0 - $1) }.max() ?? .infinity
        #expect(descriptorDelta < 1e-3, "descriptors differ by \(descriptorDelta)")
    }
}
