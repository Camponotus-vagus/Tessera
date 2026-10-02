import CoreGraphics
import Foundation
import Testing

@testable import StitchKit

/// Which settings a model set can run, checked on folders of fake model files (only their names count).
@Suite("Learned configurations")
struct LearnedConfigurationTests {
    static let coreML: [LearnedBackend] = [.coreMLGPU, .coreMLNeuralEngine, .coreMLAll]

    /// The models a source checkout has besides the downloaded set.
    static let onnxEntries = [
        "raco_aliked_extractor_k2048_768x1024.onnx", "raco_aliked_extractor_k2048_1024x768.onnx",
        "lightglue_raco_aliked_k2048.onnx", "lightglue_raco_aliked_k2048_fp32.mlpackage",
    ]

    static func fakeSet(_ entries: [String], keypoints: Int = 2048) throws -> LearnedModelSet {
        let directory = try Synthetic.temporaryDirectory()
        try ModelInstallerTests.writeFakeSet(entries, in: directory, marker: "fake")
        return LearnedModelSet(directory: directory, keypoints: keypoints)
    }

    static func configuration(_ models: LearnedModelSet, _ change: (inout PipelineConfiguration) -> Void = { _ in })
        -> PipelineConfiguration {
        var configuration = PipelineConfiguration()
        configuration.learnedModels = models
        change(&configuration)
        return configuration
    }

    @Test("The downloaded set offers Core ML backends and the fp16 matcher only")
    func downloadedSet() throws {
        let models = try Self.fakeSet(ModelInstallerTests.entries)
        defer { try? FileManager.default.removeItem(at: models.directory) }
        let configuration = Self.configuration(models)
        #expect(models.isUsable && models.supports(configuration))
        #expect(models.extractorBackends(for: configuration) == Self.coreML)
        #expect(models.matcherBackends(for: configuration) == Self.coreML)
        #expect(models.matcherPrecisions(for: configuration) == ["fp16"])
        #expect(!models.supports(Self.configuration(models) { $0.extractorBackend = .onnxCPU }))
        #expect(!models.supports(Self.configuration(models) { $0.matcherBackend = .onnxCPU }))
        #expect(!models.supports(Self.configuration(models) { $0.matcherPrecision = "fp32" }))
        #expect(!models.supports(Self.configuration(models) { $0.matcherPrecision = "fp8" }))
        // The set has the ONNX select model too, but only a build with ONNX Runtime can run it.
        let onnxSelect = Self.configuration(models) { $0.nativeKeypointSelection = false }
        #expect(try models.extractorPlan(for: onnxSelect)
            == .levels(.cpuAndGPU, nativeSelection: !LearnedModelSet.onnxAvailable))
    }

    @Test("ONNX Runtime backends are offered only by builds that include it")
    func onnxBackends() throws {
        let models = try Self.fakeSet(ModelInstallerTests.entries + Self.onnxEntries)
        defer { try? FileManager.default.removeItem(at: models.directory) }
        let configuration = Self.configuration(models)
        let all = LearnedModelSet.onnxAvailable ? LearnedBackend.allCases : Self.coreML
        #expect(models.extractorBackends(for: configuration) == all)
        #expect(models.matcherBackends(for: configuration) == all)
        #expect(models.matcherPrecisions(for: configuration) == ["fp16", "fp32"])
        let cpu = Self.configuration(models) {
            $0.extractorBackend = .onnxCPU
            $0.matcherBackend = .onnxCPU
        }
        #expect(models.supports(cpu) == LearnedModelSet.onnxAvailable)
        if !LearnedModelSet.onnxAvailable {
            let error = #expect(throws: StitchError.self) { try models.matcherOnCoreML(for: cpu) }
            #expect(error?.localizedDescription.contains("needs ONNX Runtime") == true)
        }
    }

    @Test("Keypoints are selected in C++ only for 1024 to 2560 of them", arguments: [512, 1023, 1024, 2048, 2560, 2561])
    func nativeSelectionRange(keypoints: Int) throws {
        let entries = ModelInstallerTests.entries.filter { !$0.hasPrefix("lightglue") }
            + ["lightglue_raco_aliked_k\(keypoints)_fp16.mlpackage"]
        let models = try Self.fakeSet(entries, keypoints: keypoints)
        defer { try? FileManager.default.removeItem(at: models.directory) }
        let configuration = Self.configuration(models)
        if LearnedModelSet.nativeSelectionKeypoints.contains(keypoints) {
            #expect(try models.extractorPlan(for: configuration) == .levels(.cpuAndGPU, nativeSelection: true))
            #expect(models.supports(configuration))
        } else {
            let error = #expect(throws: StitchError.self) { try models.extractorPlan(for: configuration) }
            #expect(error?.localizedDescription.contains("only for 1024 to 2560 keypoints, not \(keypoints)") == true)
            #expect(!models.supports(configuration))
        }
    }

    @Test("A learned configuration that cannot load leaves RootSIFT to run alone and says why")
    func rootSIFTAlone() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let scene = Synthetic.scene(width: 2600, height: 1800, seed: 7)
        var urls: [URL] = []
        for (index, origin) in [CGPoint(x: 0, y: 0), CGPoint(x: 820, y: 30)].enumerated() {
            let url = directory.appendingPathComponent("tile\(index).png")
            try Synthetic.write(Synthetic.tile(of: scene, origin: origin, size: CGSize(width: 1200, height: 900)), to: url)
            urls.append(url)
        }
        let models = try Self.fakeSet(ModelInstallerTests.entries)
        defer { try? FileManager.default.removeItem(at: models.directory) }
        var configuration = Self.configuration(models) {
            $0.mode = .plane
            $0.sources = [.rootSIFT, .racoLightGlue]
            // The downloaded set has no ONNX matcher: the session fails before loading anything.
            $0.matcherBackend = .onnxCPU
        }

        let report = try await StitchEngine().analyze(urls: urls, configuration: configuration)
        #expect(report.learnedPipeline == nil)
        let problem = try #require(report.learnedProblem)
        #expect(problem.contains("LightGlue"))
        #expect(report.evidence(0, 1, source: .rootSIFT)?.verdict == .verified)
        #expect(report.pairs.allSatisfy { $0.source == .rootSIFT })
        #expect(report.withoutLocalPaths().learnedProblem?.contains(models.directory.path) == false)

        // With nothing else to run, the error stops the analysis.
        configuration.sources = [.racoLightGlue]
        await #expect(throws: StitchError.self) { try await StitchEngine().analyze(urls: urls, configuration: configuration) }
    }
}
