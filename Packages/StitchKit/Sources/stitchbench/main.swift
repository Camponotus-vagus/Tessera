import Darwin
import Foundation
import StitchKit

// Command-line driver for the diagnostic pipeline: timings, a text summary,
// and optionally the JSON report plus one PNG per pair.

struct Options {
    var configuration = PipelineConfiguration()
    var models: URL?
    var keypoints = 2048
    var downloadModels = false
    var output: URL?
    var repeats = 1
    var files: [URL] = []
}

func usage(_ problem: String? = nil) -> Never {
    if let problem { FileHandle.standardError.write(Data("stitchbench: \(problem)\n".utf8)) }
    print("""
    usage: stitchbench [--mode auto|rotation|plane|document] [--source sift|lightglue|both]
                       [--models dir] [--keypoints 2048] [--extractor onnx|gpu|ane|all] [--extractor-precision fp32|fp16]
                       [--matcher onnx|gpu|ane|all] [--precision fp16|fp32] [--sift-mp 1.5] [--all-pairs]
                       [--low-memory] [--native-select] [--repeat N] [--out dir] images...
           stitchbench --download-models [images...]
    """)
    exit(2)
}

func warn(_ message: String) {
    FileHandle.standardError.write(Data("stitchbench: \(message)\n".utf8))
}

func backend(_ name: String) -> LearnedBackend {
    switch name {
    case "onnx": .onnxCPU
    case "gpu": .coreMLGPU
    case "ane": .coreMLNeuralEngine
    case "all": .coreMLAll
    default: usage("unknown backend \(name)")
    }
}

func precision(_ name: String) -> String {
    guard ["fp16", "fp32"].contains(name) else { usage("unknown precision \(name)") }
    return name
}

func parse() -> Options {
    var options = Options()
    var arguments = CommandLine.arguments.dropFirst()
    func value(_ option: String) -> String {
        guard let next = arguments.popFirst() else { usage("\(option) needs a value") }
        return next
    }
    func number<T: LosslessStringConvertible & Comparable>(_ option: String, minimum: T) -> T {
        let text = value(option)
        guard let number = T(text), number >= minimum else { usage("invalid value for \(option): \(text)") }
        return number
    }
    while let argument = arguments.popFirst() {
        switch argument {
        case "--mode":
            let text = value(argument)
            guard let mode = StitchMode(rawValue: text) else { usage("unknown mode \(text)") }
            options.configuration.mode = mode
        case "--source":
            switch value(argument) {
            case "sift": options.configuration.sources = [.rootSIFT]
            case "lightglue": options.configuration.sources = [.racoLightGlue]
            case "both": options.configuration.sources = [.rootSIFT, .racoLightGlue]
            case let other: usage("unknown source \(other)")
            }
        case "--models": options.models = URL(fileURLWithPath: value(argument), isDirectory: true)
        case "--keypoints": options.keypoints = number(argument, minimum: 1)
        case "--download-models": options.downloadModels = true
        case "--matcher": options.configuration.matcherBackend = backend(value(argument))
        case "--precision": options.configuration.matcherPrecision = precision(value(argument))
        case "--extractor": options.configuration.extractorBackend = backend(value(argument))
        case "--extractor-precision": options.configuration.extractorPrecision = precision(value(argument))
        case "--sift-mp": options.configuration.siftMegapixels = number(argument, minimum: 0.01)
        case "--all-pairs": options.configuration.pairSelection = .all
        case "--low-memory": options.configuration.lightGlueLowMemory = true
        case "--native-select": options.configuration.nativeKeypointSelection = true
        case "--repeat": options.repeats = number(argument, minimum: 1)
        case "--out": options.output = URL(fileURLWithPath: value(argument), isDirectory: true)
        case "-h", "--help": usage()
        case let option where option.hasPrefix("--"): usage("unknown option \(option)")
        default: options.files.append(URL(fileURLWithPath: argument))
        }
    }
    // --download-models alone only installs the models.
    if options.files.count < 2, !(options.downloadModels && options.files.isEmpty) {
        usage("at least two images are needed")
    }
    return options
}

/// Picks the learned models, after any download: --models, or the first usable set in the search paths.
func resolveModels(_ options: inout Options) {
    options.configuration.learnedModels = options.models.map { LearnedModelSet(directory: $0, keypoints: options.keypoints) }
        ?? LearnedModelSet.standard(keypoints: options.keypoints)
    if let models = options.configuration.learnedModels, !models.isUsable {
        usage("no usable models in \(models.directory.path)")
    }
    // As in the app: without models the learned matcher is dropped, with a warning, rather than failing.
    if options.configuration.learnedModels == nil, options.configuration.sources.contains(.racoLightGlue) {
        options.configuration.sources.removeAll { $0 == .racoLightGlue }
        guard !options.configuration.sources.isEmpty else {
            usage("no RaCo-ALIKED + LightGlue models found; run stitchbench --download-models or pass --models")
        }
        warn("no RaCo-ALIKED + LightGlue models found, running RootSIFT only")
    }
}

/// Installs the default models in Application Support, as the app does, with progress on standard error.
func downloadModels() async throws {
    let installer = ModelInstaller()
    guard let manifest = installer.manifest else { throw ModelInstallError.noManifest }
    FileHandle.standardError.write(Data("downloading \(manifest.url.absoluteString)\n".utf8))
    let terminal = isatty(STDERR_FILENO) == 1
    // On a terminal each report overwrites the previous one, and the line is ended before anything else.
    func endLine() { if terminal { FileHandle.standardError.write(Data("\n".utf8)) } }
    let models: LearnedModelSet
    do {
        models = try await installer.install { progress in
            let line: String? = switch progress.stage {
            case .downloading:
                if progress.received > 0, terminal || progress.received == progress.expected {
                    "\(progress.received.formatted(.byteCount(style: .file)))" +
                        (progress.expected.map { " of \($0.formatted(.byteCount(style: .file)))" } ?? "")
                } else {
                    nil
                }
            case .verifying: "checking SHA-256"
            case .extracting: "extracting"
            }
            if let line { FileHandle.standardError.write(Data((terminal ? "\r\u{1B}[K\(line)" : "\(line)\n").utf8)) }
        }
    } catch {
        endLine()
        throw error
    }
    endLine()
    print("models ready in \(models.directory.path)")
}

func peakResidentMegabytes() -> Double {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    return Double(usage.ru_maxrss) / 1_048_576
}

func format(_ value: Double, _ digits: Int = 2) -> String { String(format: "%.\(digits)f", value) }

func run() async throws {
    var options = parse()
    if options.downloadModels {
        try await downloadModels()
        if options.files.isEmpty { return }
    }
    resolveModels(&options)
    let engine = StitchEngine()
    var report: MatchReport?
    for run in 1...options.repeats {
        let start = ContinuousClock.now
        report = try await engine.analyze(urls: options.files, configuration: options.configuration)
        let elapsed = ContinuousClock.now - start
        print("run \(run): \(elapsed.formatted(.units(allowed: [.seconds, .milliseconds], width: .narrow))), " +
              "peak RSS \(format(peakResidentMegabytes(), 0)) MB")
    }
    guard let report else { exit(1) }
    if let pipeline = report.learnedPipeline { print("learned: \(pipeline)") }

    print("\nimages")
    for image in report.images {
        let counts = report.features.filter { $0.imageID == image.id }
            .map { "\($0.source.rawValue)=\($0.keypoints.count) (\(format($0.extractionSeconds))s)" }
            .joined(separator: " ")
        print("  [\(image.id)] \(image.name) \(image.pixelSize.width)x\(image.pixelSize.height) \(counts)")
    }

    if let candidates = report.candidates {
        print("\ncandidates (\(candidates.count) of \(report.images.count * (report.images.count - 1) / 2) pairs)")
        for pair in candidates {
            print("  \(pair.a)-\(pair.b) affinity=\(pair.affinity.map(String.init) ?? "-") " +
                  pair.reasons.map(\.rawValue).joined(separator: ","))
        }
    }

    print("\npairs")
    for pair in report.pairs {
        let fits = pair.fits.map { "\($0.model.rawValue.prefix(5))=\($0.inliers.count)" }.joined(separator: " ")
        print("  \(pair.a)-\(pair.b) \(pair.source.rawValue.padding(toLength: 13, withPad: " ", startingAt: 0)) " +
              "matches=\(pair.matches.count) [\(fits)] chosen=\(pair.chosenModel?.rawValue ?? "-") " +
              "inliers=\(pair.inlierCount)/\(pair.matchesInOverlap) nfa=\(pair.log10NFA.isFinite ? format(pair.log10NFA, 0) : "inf") " +
              "bl=\(format(pair.confidence)) cover=\(format(pair.inlierCoverage)) " +
              "rmse=\(format(pair.chosenFit?.rmse ?? 0, 1))px \(pair.verdict.rawValue) " +
              "t=\(format(pair.matchingSeconds))+\(format(pair.verificationSeconds, 3))s")
    }

    print("\ngraph")
    for (index, members) in report.graph.components.enumerated() {
        print("  component \(index): \(members)")
    }
    for node in report.graph.nodes {
        if let exclusion = node.exclusion { print("  image \(node.id) excluded: \(exclusion.rawValue)") }
    }

    if let output = options.output {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try report.jsonData().write(to: output.appendingPathComponent("report.json"))
        for pair in report.pairs where pair.matches.count > 0 {
            let image = try PairRenderer.render(pair, report: report)
            try PairRenderer.writePNG(image, to: output.appendingPathComponent("pair-\(pair.a)-\(pair.b)-\(pair.source.rawValue).png"))
        }
        print("\nwrote \(output.path)")
    }
}

do {
    try await run()
} catch {
    FileHandle.standardError.write(Data("stitchbench: \(error.localizedDescription)\n".utf8))
    exit(1)
}
