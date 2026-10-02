import Accelerate
import CoreML
import CryptoKit
import CStitchCore
import Foundation

/// Where a learned model runs: ONNX Runtime on the CPU, or Core ML on the chosen compute units.
public enum LearnedBackend: String, Sendable, Codable, CaseIterable {
    case onnxCPU
    case coreMLGPU
    case coreMLNeuralEngine
    case coreMLAll

    var computeUnits: MLComputeUnits? {
        switch self {
        case .onnxCPU: nil
        case .coreMLGPU: .cpuAndGPU
        case .coreMLNeuralEngine: .cpuAndNeuralEngine
        case .coreMLAll: .all
        }
    }
}

/// File layout of the split models produced by tools/export.
public struct LearnedModelSet: Sendable, Hashable, Codable {
    public var directory: URL
    public var keypoints: Int

    /// Fixed network input sizes, landscape and portrait.
    static let landscape = PixelSize(width: 1024, height: 768)
    static let portrait = PixelSize(width: 768, height: 1024)

    public init(directory: URL, keypoints: Int) {
        self.directory = directory
        self.keypoints = keypoints
    }

    func extractor(for canvas: PixelSize) -> URL {
        directory.appendingPathComponent("raco_aliked_extractor_k\(keypoints)_\(canvas.height)x\(canvas.width).onnx")
    }

    var onnxMatcher: URL { directory.appendingPathComponent("lightglue_raco_aliked_k\(keypoints).onnx") }

    /// Dense half of the split extractor (Core ML) and its sparse half (ONNX Runtime).
    func dense(for canvas: PixelSize, precision: String) -> URL {
        directory.appendingPathComponent("raco_aliked_dense_\(canvas.height)x\(canvas.width)_\(precision).mlpackage")
    }

    func sparse(for canvas: PixelSize) -> URL {
        directory.appendingPathComponent("raco_aliked_sparse_k\(keypoints)_\(canvas.height)x\(canvas.width).onnx")
    }

    /// Faster split: dense maps and ALIKED's feature levels on Core ML, keypoint selection on ONNX
    /// Runtime (or in C++), descriptor head in C++.
    func levels(for canvas: PixelSize) -> URL {
        directory.appendingPathComponent("raco_aliked_levels_\(canvas.height)x\(canvas.width)_fp32.mlpackage")
    }

    func select(for canvas: PixelSize) -> URL {
        directory.appendingPathComponent("raco_select_k\(keypoints)_\(canvas.height)x\(canvas.width).onnx")
    }

    var descriptorHead: URL { directory.appendingPathComponent("aliked_descriptor_head.bin") }

    var hasLevelsExtractor: Bool { hasLevelsExtractor(nativeSelection: true) }

    /// Whether this build links ONNX Runtime (the ONNXRuntime package trait).
    public static var onnxAvailable: Bool { sc_onnx_available() != 0 }

    /// Keypoint selection in C++ needs no select model.
    func hasLevelsExtractor(nativeSelection: Bool) -> Bool {
        FileManager.default.fileExists(atPath: descriptorHead.path) && [Self.landscape, Self.portrait].allSatisfy {
            FileManager.default.fileExists(atPath: levels(for: $0).path) &&
                (nativeSelection || FileManager.default.fileExists(atPath: select(for: $0).path))
        }
    }

    func hasSplitExtractor(precision: String) -> Bool {
        [Self.landscape, Self.portrait].allSatisfy {
            FileManager.default.fileExists(atPath: dense(for: $0, precision: precision).path) &&
                FileManager.default.fileExists(atPath: sparse(for: $0).path)
        }
    }

    func coreMLMatcher(precision: String) -> URL {
        directory.appendingPathComponent("lightglue_raco_aliked_k\(keypoints)_\(precision).mlpackage")
    }

    var hasONNXExtractor: Bool {
        [Self.landscape, Self.portrait].allSatisfy { FileManager.default.fileExists(atPath: extractor(for: $0).path) }
    }

    func hasCoreMLMatcher(precision: String) -> Bool {
        FileManager.default.fileExists(atPath: coreMLMatcher(precision: precision).path)
    }

    var hasONNXMatcher: Bool { FileManager.default.fileExists(atPath: onnxMatcher.path) }

    /// At least one complete extractor and one matcher that this build can run.
    public var isUsable: Bool {
        let onnx = Self.onnxAvailable
        let extractor = hasLevelsExtractor
            || onnx && (Self.precisions.contains { hasSplitExtractor(precision: $0) } || hasONNXExtractor)
        let matcher = Self.precisions.contains { hasCoreMLMatcher(precision: $0) } || onnx && hasONNXMatcher
        return extractor && matcher
    }

    static let precisions = ["fp16", "fp32"]

    static func canvas(for image: SourceImage) -> PixelSize {
        image.pixelSize.width >= image.pixelSize.height ? landscape : portrait
    }

    /// Folders searched for the models, in order: `TESSERA_MODELS`, the app bundle, Application Support,
    /// then `Models` in the working directory and its parents (a source checkout).
    public static var searchPaths: [URL] {
        var paths: [URL] = []
        if let custom = ProcessInfo.processInfo.environment["TESSERA_MODELS"], !custom.isEmpty {
            paths.append(URL(fileURLWithPath: custom, isDirectory: true))
        }
        if let resources = Bundle.main.resourceURL {
            paths.append(resources.appendingPathComponent("Models", isDirectory: true))
        }
        paths.append(ModelInstaller.standardDestination)
        var folder = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true).standardized
        while true {
            paths.append(folder.appendingPathComponent("Models", isDirectory: true))
            let parent = folder.deletingLastPathComponent().standardized
            if parent.path == folder.path { break }
            folder = parent
        }
        return paths
    }

    /// The first folder in `searchPaths` with a usable model set.
    public static func standard(keypoints: Int = 2048) -> LearnedModelSet? {
        searchPaths.map { LearnedModelSet(directory: $0, keypoints: keypoints) }.first { $0.isUsable }
    }

    /// Keypoint counts for which the C++ selection reproduces RaCo: those for which RaCo ranks the
    /// boundary window on these canvases.
    static let nativeSelectionKeypoints = 1024...2560

    /// The extractor that `LearnedSession` builds.
    enum ExtractorPlan: Equatable {
        /// Feature levels on Core ML, keypoint selection in C++ or with the ONNX select model, C++ head.
        case levels(MLComputeUnits, nativeSelection: Bool)
        /// Dense maps on Core ML, sparse half on ONNX Runtime.
        case split(MLComputeUnits)
        /// The whole extractor on ONNX Runtime.
        case onnx
    }

    /// What `LearnedSession` runs for `configuration` with this set in this build, or why it cannot.
    func extractorPlan(for configuration: PipelineConfiguration) throws -> ExtractorPlan {
        guard Self.precisions.contains(configuration.extractorPrecision) else {
            throw StitchError.engine("Unknown precision \(configuration.extractorPrecision): use fp16 or fp32")
        }
        let onnx = Self.onnxAvailable
        let units = configuration.extractorBackend.computeUnits
        let nativeSelection = configuration.nativeKeypointSelection ?? true || !onnx
        if let units, configuration.fastDescriptorHead, hasLevelsExtractor(nativeSelection: nativeSelection) {
            guard !nativeSelection || Self.nativeSelectionKeypoints.contains(keypoints) else {
                throw StitchError.engine(
                    "Keypoint selection in C++ reproduces RaCo only for \(Self.nativeSelectionKeypoints.lowerBound) to " +
                        "\(Self.nativeSelectionKeypoints.upperBound) keypoints, not \(keypoints)" +
                        (onnx ? "; select them with the ONNX model instead" : "")
                )
            }
            return .levels(units, nativeSelection: nativeSelection)
        }
        if onnx, let units, hasSplitExtractor(precision: configuration.extractorPrecision) { return .split(units) }
        if onnx && hasONNXExtractor { return .onnx }
        throw StitchError.engine(units == nil && !onnx
            ? String(localized: "The RaCo-ALIKED extractor on the CPU needs ONNX Runtime, which this build does not include")
            : String(localized: "No RaCo-ALIKED extractor for this configuration in \(directory.path)"))
    }

    /// Whether `LearnedSession` runs LightGlue on Core ML (true) or ONNX Runtime (false) for
    /// `configuration`, or why it cannot run it. A missing Core ML package falls back on ONNX Runtime.
    func matcherOnCoreML(for configuration: PipelineConfiguration) throws -> Bool {
        guard Self.precisions.contains(configuration.matcherPrecision) else {
            throw StitchError.engine("Unknown precision \(configuration.matcherPrecision): use fp16 or fp32")
        }
        let coreML = configuration.matcherBackend.computeUnits != nil
        if coreML && hasCoreMLMatcher(precision: configuration.matcherPrecision) { return true }
        guard Self.onnxAvailable && hasONNXMatcher else {
            throw StitchError.engine(!coreML && !Self.onnxAvailable
                ? String(localized: "LightGlue on the CPU needs ONNX Runtime, which this build does not include")
                : String(localized: "No LightGlue matcher for this configuration in \(directory.path)"))
        }
        return false
    }

    /// Whether the learned matcher can run `configuration` with this set in this build: an extractor and
    /// a matcher for its backends, precisions and number of keypoints.
    public func supports(_ configuration: PipelineConfiguration) -> Bool {
        (try? extractorPlan(for: configuration)) != nil && (try? matcherOnCoreML(for: configuration)) != nil
    }

    /// The extractor backends this set and build can run, with the rest of `configuration` unchanged.
    public func extractorBackends(for configuration: PipelineConfiguration) -> [LearnedBackend] {
        LearnedBackend.allCases.filter { backend in
            var candidate = configuration
            candidate.extractorBackend = backend
            return (try? extractorPlan(for: candidate)) != nil
        }
    }

    /// The matcher backends this set and build can run, with the rest of `configuration` unchanged.
    public func matcherBackends(for configuration: PipelineConfiguration) -> [LearnedBackend] {
        LearnedBackend.allCases.filter { backend in
            var candidate = configuration
            candidate.matcherBackend = backend
            return (try? matcherOnCoreML(for: candidate)) != nil
        }
    }

    /// The matcher precisions this set and build can run, with the rest of `configuration` unchanged.
    public func matcherPrecisions(for configuration: PipelineConfiguration) -> [String] {
        Self.precisions.filter { precision in
            var candidate = configuration
            candidate.matcherPrecision = precision
            return (try? matcherOnCoreML(for: candidate)) != nil
        }
    }
}

/// Compiles a Core ML package once and keeps the result in Caches, keyed by name and by a digest of the
/// package's files (paths, sizes, modification dates). Safe against concurrent compiles of the same
/// package, and a damaged cache entry is compiled again.
func compiledCoreMLModel(_ package: URL, computeUnits: MLComputeUnits) async throws -> MLModel {
    let package = package.resolvingSymlinksInPath()
    let manager = FileManager.default
    let name = package.deletingPathExtension().lastPathComponent
    let cache = cacheDirectory("CoreML")
    let target = cache.appendingPathComponent("\(name)-\(try packageDigest(package)).mlmodelc")
    let configuration = MLModelConfiguration()
    configuration.computeUnits = computeUnits
    if manager.fileExists(atPath: target.path) {
        if let model = try? MLModel(contentsOf: target, configuration: configuration) { return model }
        try? manager.removeItem(at: target)
    }
    try manager.createDirectory(at: cache, withIntermediateDirectories: true)
    let compiled = try await MLModel.compileModel(at: package)
    // Stage next to the target, then rename: the entry appears complete or not at all. If another
    // process got there first, its copy is used.
    let staging = cache.appendingPathComponent(".\(UUID().uuidString).mlmodelc")
    defer {
        try? manager.removeItem(at: compiled)
        try? manager.removeItem(at: staging)
    }
    try manager.moveItem(at: compiled, to: staging)
    do {
        try manager.moveItem(at: staging, to: target)
    } catch {
        guard manager.fileExists(atPath: target.path) else { throw error }
    }
    // Entries compiled from earlier versions of the same package are no longer needed.
    for entry in (try? manager.contentsOfDirectory(at: cache, includingPropertiesForKeys: nil)) ?? []
    where entry.lastPathComponent.hasPrefix("\(name)-") && entry.lastPathComponent != target.lastPathComponent {
        try? manager.removeItem(at: entry)
    }
    return try MLModel(contentsOf: target, configuration: configuration)
}

func cacheDirectory(_ name: String) -> URL {
    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        ?? FileManager.default.temporaryDirectory
    return caches.appendingPathComponent("Tessera/\(name)", isDirectory: true)
}

/// Short digest of the relative path, size and modification date of every file in a package.
func packageDigest(_ package: URL) throws -> String {
    let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey, .contentModificationDateKey]
    guard let enumerator = FileManager.default.enumerator(at: package, includingPropertiesForKeys: keys) else {
        throw StitchError.engine("Cannot read \(package.lastPathComponent)")
    }
    var lines: [String] = []
    let base = package.path
    for case let file as URL in enumerator {
        let values = try file.resourceValues(forKeys: Set(keys))
        guard values.isRegularFile == true else { continue }
        let stamp = values.contentModificationDate?.timeIntervalSince1970 ?? 0
        lines.append("\(file.path.dropFirst(base.count))|\(values.fileSize ?? 0)|\(stamp)")
    }
    guard !lines.isEmpty else { throw StitchError.engine("\(package.lastPathComponent) is empty") }
    let digest = SHA256.hash(data: Data(lines.sorted().joined(separator: "\n").utf8))
    return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
}

/// Throws unless `array` holds exactly `count` elements.
func expectElements(_ array: MLMultiArray, _ count: Int, _ name: String) throws {
    guard array.count == count else {
        throw StitchError.engine("Core ML output \(name) has \(array.count) values, expected \(count): wrong model file?")
    }
}

/// Fills a new multi-array of the model's input type.
func makeArray(_ values: UnsafeBufferPointer<Float>, shape: [Int], type: MLMultiArrayDataType) throws -> MLMultiArray {
    let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
    array.withUnsafeMutableBytes { buffer, _ in
        if type == .float16 {
            let target = buffer.bindMemory(to: Float16.self)
            for index in values.indices { target[index] = Float16(values[index]) }
        } else {
            _ = buffer.bindMemory(to: Float.self).initialize(from: values)
        }
    }
    return array
}

/// Calls `body` with a contiguous float32 view of a 4-D multi-array, copying only when needed.
func withContiguousFloats<R>(_ array: MLMultiArray, _ body: (UnsafePointer<Float>) throws -> R) throws -> R {
    guard [.float32, .float16, .double].contains(array.dataType) else {
        throw StitchError.engine("unexpected Core ML output type \(array.dataType.rawValue)")
    }
    guard array.count > 0 else { throw StitchError.engine("empty Core ML output") }
    let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
    var expected = 1
    var contiguous = true
    for axis in shape.indices.reversed() {
        if shape[axis] > 1, strides[axis] != expected { contiguous = false }
        expected *= shape[axis]
    }
    if contiguous, array.dataType == .float32 {
        return try array.withUnsafeBufferPointer(ofType: Float.self) { try body($0.baseAddress!) }
    }
    if contiguous, array.dataType == .float16 {
        // vImage converts half floats in bulk; a per-element loop would cost hundreds of ms here.
        var converted = [Float](repeating: 0, count: expected)
        array.withUnsafeBytes { raw in
            converted.withUnsafeMutableBytes { target in
                var source = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: raw.baseAddress), height: 1,
                                           width: vImagePixelCount(expected), rowBytes: expected * 2)
                var destination = vImage_Buffer(data: target.baseAddress, height: 1,
                                                width: vImagePixelCount(expected), rowBytes: expected * 4)
                vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags))
            }
        }
        return try converted.withUnsafeBufferPointer { try body($0.baseAddress!) }
    }
    guard shape.count == 4 else { throw StitchError.engine("unexpected Core ML output rank \(shape.count)") }
    var copy = [Float](repeating: 0, count: expected)
    array.withUnsafeBytes { raw in
        func value(_ offset: Int) -> Float {
            switch array.dataType {
            case .float16: Float(raw.load(fromByteOffset: offset * 2, as: Float16.self))
            case .double: Float(raw.load(fromByteOffset: offset * 8, as: Double.self))
            default: raw.load(fromByteOffset: offset * 4, as: Float.self)
            }
        }
        var index = 0
        for n in 0..<shape[0] {
            for c in 0..<shape[1] {
                for y in 0..<shape[2] {
                    let row = n * strides[0] + c * strides[1] + y * strides[2]
                    for x in 0..<shape[3] {
                        copy[index] = value(row + x * strides[3])
                        index += 1
                    }
                }
            }
        }
    }
    return try copy.withUnsafeBufferPointer { try body($0.baseAddress!) }
}

/// Contiguous float32 views of several multi-arrays at once.
func withContiguousFloats<R>(_ arrays: [MLMultiArray], _ body: ([UnsafePointer<Float>]) throws -> R) throws -> R {
    func step(_ index: Int, _ collected: [UnsafePointer<Float>]) throws -> R {
        guard index < arrays.count else { return try body(collected) }
        return try withContiguousFloats(arrays[index]) { try step(index + 1, collected + [$0]) }
    }
    return try step(0, [])
}

/// LightGlue compiled for Core ML. Predictions run synchronously on the calling thread.
final class CoreMLMatcher: @unchecked Sendable {
    private let model: MLModel
    private let inputType: MLMultiArrayDataType

    init(package: URL, computeUnits: MLComputeUnits) async throws {
        model = try await compiledCoreMLModel(package, computeUnits: computeUnits)
        inputType = model.modelDescription.inputDescriptionsByName["keypoints"]?.multiArrayConstraint?.dataType
            ?? .float32
    }

    func match(keypoints: [Float], descriptors: [Float], count: Int, descriptorSize: Int) throws
        -> (partner: [Int32], confidence: [Float]) {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "keypoints": try keypoints.withUnsafeBufferPointer { try makeArray($0, shape: [2, count, 2], type: inputType) },
            "descriptors": try descriptors.withUnsafeBufferPointer {
                try makeArray($0, shape: [2, count, descriptorSize], type: inputType)
            },
        ])
        let output = try model.prediction(from: input)
        guard let match0 = output.featureValue(for: "match0")?.multiArrayValue,
              let confidence0 = output.featureValue(for: "confidence0")?.multiArrayValue
        else { throw StitchError.engine("Core ML matcher returned no outputs") }
        try expectElements(match0, count, "match0")
        try expectElements(confidence0, count, "confidence0")
        let partner = read(match0, count: count).map { $0.isFinite && $0 >= 0 && $0 < Float(count) ? Int32($0) : -1 }
        return (partner, read(confidence0, count: count))
    }

    private func read(_ array: MLMultiArray, count: Int) -> [Float] {
        array.withUnsafeBytes { buffer in
            switch array.dataType {
            case .float16: Array(buffer.bindMemory(to: Float16.self).prefix(count)).map(Float.init)
            case .int32: Array(buffer.bindMemory(to: Int32.self).prefix(count)).map(Float.init)
            case .double: Array(buffer.bindMemory(to: Double.self).prefix(count)).map(Float.init)
            default: Array(buffer.bindMemory(to: Float.self).prefix(count))
            }
        }
    }
}

/// Dense part of the extractor on Core ML: image -> named feature maps.
final class CoreMLDense: @unchecked Sendable {
    private let model: MLModel
    private let inputType: MLMultiArrayDataType

    init(package: URL, computeUnits: MLComputeUnits) async throws {
        model = try await compiledCoreMLModel(package, computeUnits: computeUnits)
        inputType = model.modelDescription.inputDescriptionsByName["image"]?.multiArrayConstraint?.dataType ?? .float32
    }

    func run(_ planar: [Float], canvas: PixelSize, outputs names: [String]) throws -> [MLMultiArray] {
        let image = try planar.withUnsafeBufferPointer {
            try makeArray($0, shape: [1, 3, canvas.height, canvas.width], type: inputType)
        }
        let output = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: ["image": image]))
        return try names.map { name in
            guard let array = output.featureValue(for: name)?.multiArrayValue else {
                throw StitchError.engine("Core ML extractor returned no \(name)")
            }
            return array
        }
    }
}

/// ALIKED's descriptor head in C++ (weights from tools/export).
final class DescriptorHead: @unchecked Sendable {
    let pointer: OpaquePointer
    let dimensions: Int

    init(_ url: URL) throws {
        var message = [CChar](repeating: 0, count: 512)
        guard let pointer = sc_descriptor_head_load(url.path, &message, message.count) else {
            throw StitchError.engine(errorText(message))
        }
        self.pointer = pointer
        dimensions = Int(sc_descriptor_head_dimensions(pointer))
    }

    deinit { sc_descriptor_head_free(pointer) }
}

/// Owns an ONNX Runtime session.
final class ONNXModel: @unchecked Sendable {
    let pointer: OpaquePointer

    init(_ url: URL, execution: sc_execution, lowMemory: Bool) throws {
        let cache = cacheDirectory("ONNX")
        try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        var message = [CChar](repeating: 0, count: 1024)
        guard let pointer = sc_onnx_model_create(url.path, execution, cache.path, 0, lowMemory ? 1 : 0, &message,
                                                 message.count)
        else { throw StitchError.engine(errorText(message)) }
        self.pointer = pointer
    }

    deinit { sc_onnx_model_free(pointer) }
}

/// Extractors for both orientations plus a matcher, each on its configured backend.
final class LearnedSession: @unchecked Sendable {
    struct Extractor {
        var full: ONNXModel?
        var dense: CoreMLDense?
        var sparse: ONNXModel?
        var select: ONNXModel?
    }

    let key: String
    let models: LearnedModelSet
    /// What actually runs, for reports and logs (an absent Core ML file falls back to ONNX Runtime).
    let pipeline: String
    private var extractors: [PixelSize: Extractor] = [:]
    private let gpu = NSLock()
    private let onnxMatcher: ONNXModel?
    private let coreMLMatcher: CoreMLMatcher?
    private let head: DescriptorHead?

    init(models: LearnedModelSet, configuration: PipelineConfiguration) async throws {
        self.models = models
        key = Self.key(models, configuration)
        let lowMemory = configuration.lightGlueLowMemory
        let plan = try models.extractorPlan(for: configuration)
        let matcherOnCoreML = try models.matcherOnCoreML(for: configuration)
        if case .levels = plan {
            head = try DescriptorHead(models.descriptorHead)
        } else {
            head = nil
        }
        for canvas in [LearnedModelSet.landscape, LearnedModelSet.portrait] {
            switch plan {
            case .levels(let units, let nativeSelection):
                extractors[canvas] = Extractor(
                    dense: try await CoreMLDense(package: models.levels(for: canvas), computeUnits: units),
                    select: nativeSelection ? nil
                        : try ONNXModel(models.select(for: canvas), execution: SC_EXECUTION_CPU, lowMemory: lowMemory)
                )
            case .split(let units):
                extractors[canvas] = Extractor(
                    dense: try await CoreMLDense(package: models.dense(for: canvas, precision: configuration.extractorPrecision),
                                                 computeUnits: units),
                    sparse: try ONNXModel(models.sparse(for: canvas), execution: SC_EXECUTION_CPU, lowMemory: lowMemory)
                )
            case .onnx:
                extractors[canvas] = Extractor(
                    full: try ONNXModel(models.extractor(for: canvas), execution: SC_EXECUTION_CPU, lowMemory: lowMemory)
                )
            }
        }
        let extractorName = switch plan {
        case .levels(_, let nativeSelection):
            "levels + \(nativeSelection ? "C++ select" : "select") + C++ head (Core ML \(configuration.extractorBackend.rawValue))"
        case .split: "dense \(configuration.extractorPrecision) + sparse (Core ML \(configuration.extractorBackend.rawValue))"
        case .onnx: "ONNX Runtime CPU"
        }
        let matcherName: String
        if matcherOnCoreML, let units = configuration.matcherBackend.computeUnits {
            coreMLMatcher = try await CoreMLMatcher(package: models.coreMLMatcher(precision: configuration.matcherPrecision),
                                                    computeUnits: units)
            onnxMatcher = nil
            matcherName = "Core ML \(configuration.matcherPrecision) \(configuration.matcherBackend.rawValue)"
        } else {
            onnxMatcher = try ONNXModel(models.onnxMatcher, execution: SC_EXECUTION_CPU, lowMemory: lowMemory)
            coreMLMatcher = nil
            matcherName = "ONNX Runtime CPU"
        }
        pipeline = "extractor: \(extractorName); matcher: \(matcherName)"
    }

    static func key(_ models: LearnedModelSet, _ configuration: PipelineConfiguration) -> String {
        "\(models.directory.path)|\(models.keypoints)|\(configuration.extractorBackend.rawValue)|" +
            "\(configuration.extractorPrecision)|\(configuration.matcherBackend.rawValue)|" +
            "\(configuration.matcherPrecision)|\(configuration.lightGlueLowMemory)|\(configuration.fastDescriptorHead)|" +
            "\(configuration.nativeKeypointSelection ?? true)"
    }

    /// Keypoints (canvas pixels) and descriptors for one photo drawn into a fixed-size canvas.
    func extract(_ planar: [Float], canvas: PixelSize) throws -> (keypoints: [Float], descriptors: [Float], size: Int) {
        guard let extractor = extractors[canvas] else { throw StitchError.modelMissing }
        var result = sc_extraction()
        var message = [CChar](repeating: 0, count: 1024)
        defer { sc_extraction_free(&result) }
        let status: Int32
        if let dense = extractor.dense, let head {
            // One Core ML prediction at a time; other photos go through the CPU stages meanwhile.
            let maps = try gpu.withLock {
                try dense.run(planar, canvas: canvas, outputs: ["logits", "ranker", "level1", "level2", "level3", "level4"])
            }
            try expectElements(maps[0], canvas.width * canvas.height, "logits")
            try expectElements(maps[1], canvas.width * canvas.height, "ranker")
            guard maps[2...5].allSatisfy({ $0.shape.count == 4 }) else {
                throw StitchError.engine("Core ML feature levels have the wrong rank: wrong model file?")
            }
            var keypoints: [Float]
            if let select = extractor.select {
                status = try withContiguousFloats(Array(maps[0...1])) { pointers in
                    sc_select_keypoints(select.pointer, pointers[0], pointers[1], Int32(canvas.width), Int32(canvas.height),
                                        &result, &message, message.count)
                }
                guard status == 0 else { throw StitchError.engine(errorText(message)) }
                keypoints = Array(UnsafeBufferPointer(start: result.keypoints, count: Int(result.keypoint_count) * 2))
            } else {
                keypoints = [Float](repeating: 0, count: models.keypoints * 2)
                status = try withContiguousFloats(Array(maps[0...1])) { pointers in
                    sc_select_keypoints_native(pointers[0], pointers[1], Int32(canvas.width), Int32(canvas.height),
                                               Int32(models.keypoints), &keypoints, &message, message.count)
                }
                guard status == 0 else { throw StitchError.engine(errorText(message)) }
            }
            let count = keypoints.count / 2
            var descriptors = [Float](repeating: 0, count: count * head.dimensions)
            let levels = Array(maps[2...5])
            let widths = levels.map { Int32($0.shape[3].intValue) }, heights = levels.map { Int32($0.shape[2].intValue) }
            let described = try withContiguousFloats(levels) { pointers in
                pointers.map { Optional($0) }.withUnsafeBufferPointer { levelPointers in
                    sc_describe(head.pointer, levelPointers.baseAddress, widths, heights, Int32(levels.count),
                                Int32(levels[0].shape[1].intValue), Int32(canvas.width), Int32(canvas.height),
                                keypoints, Int32(count), &descriptors, &message, message.count)
                }
            }
            guard described == 0 else { throw StitchError.engine(errorText(message)) }
            return (keypoints, descriptors, head.dimensions)
        } else if let dense = extractor.dense, let sparse = extractor.sparse {
            let maps = try gpu.withLock { try dense.run(planar, canvas: canvas, outputs: ["logits", "ranker", "features"]) }
            try expectElements(maps[0], canvas.width * canvas.height, "logits")
            try expectElements(maps[1], canvas.width * canvas.height, "ranker")
            guard maps[2].shape.count == 4 else { throw StitchError.engine("Core ML features have the wrong rank") }
            let channels = maps[2].shape[1].intValue
            try expectElements(maps[2], channels * canvas.width * canvas.height, "features")
            status = try withContiguousFloats(maps) { pointers in
                sc_extract_sparse(sparse.pointer, pointers[0], pointers[1], pointers[2], Int32(canvas.width),
                                  Int32(canvas.height), Int32(channels), &result, &message, message.count)
            }
        } else if let full = extractor.full {
            status = planar.withUnsafeBufferPointer { buffer in
                sc_extract(full.pointer, buffer.baseAddress, Int32(canvas.width), Int32(canvas.height), &result,
                           &message, message.count)
            }
        } else {
            throw StitchError.modelMissing
        }
        guard status == 0 else { throw StitchError.engine(errorText(message)) }
        let count = Int(result.keypoint_count), size = Int(result.descriptor_size)
        return (Array(UnsafeBufferPointer(start: result.keypoints, count: count * 2)),
                Array(UnsafeBufferPointer(start: result.descriptors, count: count * size)), size)
    }

    func match(keypoints: [Float], descriptors: [Float], count: Int, descriptorSize: Int) throws
        -> (partner: [Int32], confidence: [Float]) {
        if let coreMLMatcher {
            return try coreMLMatcher.match(keypoints: keypoints, descriptors: descriptors, count: count,
                                           descriptorSize: descriptorSize)
        }
        guard let onnxMatcher else { throw StitchError.modelMissing }
        var partner = [Int32](repeating: -1, count: count)
        var confidence = [Float](repeating: 0, count: count)
        var message = [CChar](repeating: 0, count: 1024)
        let status = sc_match_learned(onnxMatcher.pointer, keypoints, descriptors, Int32(count), Int32(descriptorSize),
                                      &partner, &confidence, &message, message.count)
        guard status == 0 else { throw StitchError.engine(errorText(message)) }
        return (partner, confidence)
    }
}
