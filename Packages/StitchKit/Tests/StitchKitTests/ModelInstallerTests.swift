import CryptoKit
import Foundation
import Testing

@testable import StitchKit

/// Answers the installer's requests to hosts ending in `.test`: an archive, an error status, or the start of
/// a body that never ends.
final class FakeServer: URLProtocol, @unchecked Sendable {
    enum Answer {
        /// The archive in small pieces, with its Content-Length when `announced`.
        case archive(Data, announced: Bool)
        /// An error page of 1 MB, enough for several progress reports if it were downloaded.
        case status(Int)
        /// `first` bytes of a body of `total` bytes, then nothing until the task is cancelled.
        case stall(first: Int, total: Int)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var answers: [String: Answer] = [:]

    /// A URL on a host of its own, so that tests running in parallel do not see each other's answers.
    static func url(answering answer: Answer) -> URL {
        let host = "\(UUID().uuidString.lowercased()).test"
        lock.withLock { answers[host] = answer }
        return URL(string: "https://\(host)/tessera-models.zip")!
    }

    /// The installer's configuration with this protocol in front of the built-in ones.
    static func configuration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FakeServer.self] + (configuration.protocolClasses ?? [])
        return configuration
    }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host?.hasSuffix(".test") == true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let answer = Self.lock.withLock({ Self.answers[url.host ?? ""] }) else {
            client?.urlProtocol(self, didFailWithError: URLError(.cannotFindHost))
            return
        }
        switch answer {
        case .archive(let data, let announced):
            respond(url, status: 200, length: announced ? data.count : nil)
            for start in stride(from: 0, to: data.count, by: 512) {
                client?.urlProtocol(self, didLoad: data.subdata(in: start..<min(data.count, start + 512)))
            }
            client?.urlProtocolDidFinishLoading(self)
        case .status(let status):
            let page = Data(count: 1 << 20)
            respond(url, status: status, length: page.count)
            client?.urlProtocol(self, didLoad: page)
            client?.urlProtocolDidFinishLoading(self)
        case .stall(let first, let total):
            respond(url, status: 200, length: total)
            client?.urlProtocol(self, didLoad: Data(count: first))
        }
    }

    override func stopLoading() {}

    private func respond(_ url: URL, status: Int, length: Int?) {
        let headers = ["Content-Type": "application/zip"].merging(length.map { ["Content-Length": "\($0)"] } ?? [:]) { $1 }
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    }
}

/// The installer runs against zips of small fake model files served from file:// URLs or by `FakeServer`,
/// into a temporary folder that stands in for Application Support.
@Suite("Model installer")
struct ModelInstallerTests {
    /// The names `LearnedModelSet.isUsable` looks for, as in tools/package-models.sh.
    static let entries = [
        "raco_aliked_levels_768x1024_fp32.mlpackage", "raco_aliked_levels_1024x768_fp32.mlpackage",
        "raco_select_k2048_768x1024.onnx", "raco_select_k2048_1024x768.onnx",
        "aliked_descriptor_head.bin",
        "lightglue_raco_aliked_k2048_fp16.mlpackage",
    ]

    /// A temporary folder with `served/` for the archives and `support/` for the installation.
    struct Sandbox {
        let root: URL
        var support: URL { root.appendingPathComponent("support", isDirectory: true) }
        var destination: URL { support.appendingPathComponent("Models", isDirectory: true) }
        /// Stands in for the Core ML cache in Caches, which the fake set's package names would otherwise reach.
        var cache: URL { root.appendingPathComponent("cache", isDirectory: true) }

        init() throws {
            root = try Synthetic.temporaryDirectory()
            try FileManager.default.createDirectory(at: root.appendingPathComponent("served"), withIntermediateDirectories: true)
        }

        /// Zips fake model files the way tools/package-models.sh does and returns a manifest for the archive.
        func archive(_ entries: [String] = ModelInstallerTests.entries, name: String = "models.zip",
                     marker: String = "new") throws -> ModelManifest {
            let source = root.appendingPathComponent("source-\(UUID().uuidString)", isDirectory: true)
            try ModelInstallerTests.writeFakeSet(entries, in: source, marker: marker)
            let zip = root.appendingPathComponent("served/\(name)")
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
            process.arguments = ["-qry", zip.path] + entries
            process.currentDirectoryURL = source
            try process.run()
            process.waitUntilExit()
            #expect(process.terminationStatus == 0)
            return try manifest(for: zip)
        }

        func manifest(for file: URL) throws -> ModelManifest {
            ModelManifest(release: "test", archive: file.lastPathComponent, url: file,
                          sha256: try ModelInstaller.sha256(of: file))
        }

        func installer(_ manifest: ModelManifest?) -> ModelInstaller {
            var installer = ModelInstaller(manifest: manifest, destination: destination)
            installer.sessionConfiguration = { FakeServer.configuration() }
            installer.coreMLCache = cache
            return installer
        }

        /// What is left in the folder that holds the installation: nothing partial or hidden.
        var supportContents: [String] {
            ((try? FileManager.default.contentsOfDirectory(atPath: support.path)) ?? []).sorted()
        }

        func marker() -> String? {
            try? String(contentsOf: destination.appendingPathComponent("aliked_descriptor_head.bin"), encoding: .utf8)
        }

        func remove() {
            try? FileManager.default.removeItem(at: root)
        }
    }

    /// Model packages become folders with one file inside, the other models small files holding `marker`.
    static func writeFakeSet(_ entries: [String], in folder: URL, marker: String) throws {
        let manager = FileManager.default
        try manager.createDirectory(at: folder, withIntermediateDirectories: true)
        for entry in entries {
            let url = folder.appendingPathComponent(entry)
            if entry.hasSuffix(".mlpackage") {
                try manager.createDirectory(at: url.appendingPathComponent("Data"), withIntermediateDirectories: true)
                try Data("{}".utf8).write(to: url.appendingPathComponent("Manifest.json"))
            } else {
                try Data(marker.utf8).write(to: url)
            }
        }
    }

    /// Files that URLSession download tasks leave in the temporary folder when they stop.
    static func temporaryDownloads() -> Set<String> {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: NSTemporaryDirectory())) ?? []
        return Set(names.filter { $0.hasPrefix("CFNetworkDownload_") })
    }

    /// Collects progress reports from any thread.
    final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [ModelInstallProgress] = []

        func record(_ event: ModelInstallProgress) { lock.withLock { events.append(event) } }
        var all: [ModelInstallProgress] { lock.withLock { events } }
    }

    @Test("The manifest in tools/models.json decodes")
    func manifestDecoding() throws {
        let file = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../../../tools/models.json").standardized
        let manifest = try ModelManifest.decode(Data(contentsOf: file))
        #expect(manifest.version == 2)
        #expect(manifest.url.scheme == "https")
        #expect(manifest.url.lastPathComponent == manifest.archive)
        #expect(manifest.sha256.count == 64 && manifest.sha256.allSatisfy(\.isHexDigit))
        #expect((manifest.size ?? 0) > 0)

        let literal = try ModelManifest.decode(Data("""
        {"version": 2, "release": "models-9", "archive": "tessera-models-9.zip",
         "url": "https://example.org/tessera-models-9.zip", "sha256": "ab"}
        """.utf8))
        #expect(literal == ModelManifest(release: "models-9", archive: "tessera-models-9.zip",
                                         url: URL(string: "https://example.org/tessera-models-9.zip")!, sha256: "ab"))
        #expect(literal.size == nil)
        let sized = try ModelManifest.decode(Data("""
        {"version": 2, "release": "models-9", "archive": "tessera-models-9.zip",
         "url": "https://example.org/tessera-models-9.zip", "sha256": "ab", "size": 27793379}
        """.utf8))
        #expect(sized.size == 27_793_379)
        #expect(throws: (any Error).self) { try ModelManifest.decode(Data(#"{"version": 2}"#.utf8)) }
    }

    @Test("Installs a model set from an archive and reports each stage")
    func install() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let manifest = try sandbox.archive()
        let recorder = Recorder()
        let installer = sandbox.installer(manifest)
        #expect(!installer.isInstalled)

        let models = try await installer.install { recorder.record($0) }

        #expect(models.directory == sandbox.destination)
        #expect(models.isUsable && installer.isInstalled)
        #expect(sandbox.marker() == "new")
        #expect(sandbox.supportContents == ["Models"])
        let events = recorder.all
        #expect(events.first == ModelInstallProgress(stage: .downloading, received: 0, expected: nil))
        #expect(Array(events.map(\.stage).drop { $0 == .downloading }) == [.verifying, .extracting])
        let size = Int64(try #require(try manifest.url.resourceValues(forKeys: [.fileSizeKey]).fileSize))
        let received = events.filter { $0.stage == .downloading }.map(\.received)
        #expect(received == received.sorted() && received.allSatisfy { $0 <= size })
        #expect(events.last == ModelInstallProgress(stage: .extracting, received: size, expected: size))
    }

    @Test("Downloads over HTTPS in pieces, hashing the stream", arguments: [true, false])
    func servedArchive(announced: Bool) async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let local = try sandbox.archive()
        let data = try Data(contentsOf: local.url)
        var manifest = local
        manifest.url = FakeServer.url(answering: .archive(data, announced: announced))
        // Without a Content-Length the progress falls back on the size in the manifest.
        if !announced { manifest.size = Int64(data.count) }
        let recorder = Recorder()
        let before = Self.temporaryDownloads()

        try await sandbox.installer(manifest).install { recorder.record($0) }

        #expect(sandbox.marker() == "new")
        #expect(sandbox.supportContents == ["Models"])
        let size = Int64(data.count)
        #expect(recorder.all.contains(ModelInstallProgress(stage: .downloading, received: size, expected: size)))
        #expect(Self.temporaryDownloads().subtracting(before).isEmpty)
    }

    @Test("An error status fails before any of the body arrives, with the status in the message",
          arguments: [404, 500])
    func errorStatus(_ status: Int) async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let manifest = ModelManifest(release: "test", archive: "tessera-models.zip",
                                     url: FakeServer.url(answering: .status(status)), sha256: "00")
        let recorder = Recorder()
        await #expect(throws: ModelInstallError.httpStatus(status)) {
            try await sandbox.installer(manifest).install { recorder.record($0) }
        }
        #expect(recorder.all == [ModelInstallProgress(stage: .downloading, received: 0, expected: nil)])
        #expect(sandbox.supportContents.isEmpty)
        let message = ModelInstallError.httpStatus(status).localizedDescription
        #expect(message.contains("answered \(status) (\(status == 404 ? "not found" : "internal server error"))"))
    }

    @Test("Cancelling a slow download halfway leaves no partial file, here or in the temporary folder")
    func cancelMidTransfer() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let before = Self.temporaryDownloads()
        let total = 30_000_000
        let manifest = ModelManifest(release: "test", archive: "tessera-models.zip",
                                     url: FakeServer.url(answering: .stall(first: 300_000, total: total)), sha256: "00")
        let installer = sandbox.installer(manifest)
        let (reports, report) = AsyncStream.makeStream(of: ModelInstallProgress.self)
        let task = Task { try await installer.install { report.yield($0) } }

        // Wait for the first bytes, which are already in the work folder when they are reported.
        for await progress in reports where progress.received > 0 {
            #expect(progress.expected == Int64(total))
            break
        }
        let work = try #require(sandbox.supportContents.first { $0.hasPrefix(".Models-") })
        let partial = sandbox.support.appendingPathComponent(work).appendingPathComponent("archive.zip")
        #expect(try #require(partial.resourceValues(forKeys: [.fileSizeKey]).fileSize) >= 256 * 1024)

        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(sandbox.supportContents.isEmpty)
        #expect(Self.temporaryDownloads().subtracting(before).isEmpty)
    }

    @Test("Work folders left by an installer that stopped are deleted, those in use are kept")
    func abandonedWork() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let manager = FileManager.default
        func folder(_ name: String, folderAge: TimeInterval, archiveAge: TimeInterval) throws {
            let folder = sandbox.support.appendingPathComponent(name, isDirectory: true)
            let archive = folder.appendingPathComponent("archive.zip")
            try manager.createDirectory(at: folder, withIntermediateDirectories: true)
            try Data(count: 1000).write(to: archive)
            try manager.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -archiveAge)], ofItemAtPath: archive.path)
            try manager.setAttributes([.modificationDate: Date(timeIntervalSinceNow: -folderAge)], ofItemAtPath: folder.path)
        }
        let old = 2 * ModelInstaller.abandonedAfter
        let stale = ".Models-\(UUID().uuidString)", recent = ".Models-\(UUID().uuidString)"
        let writing = ".Models-\(UUID().uuidString)", other = ".Models-backup"
        try folder(stale, folderAge: old, archiveAge: old)
        try folder(recent, folderAge: ModelInstaller.abandonedAfter / 2, archiveAge: ModelInstaller.abandonedAfter / 2)
        // Created long ago, but the archive inside is still growing.
        try folder(writing, folderAge: old, archiveAge: 1)
        try folder(other, folderAge: old, archiveAge: old)

        try await sandbox.installer(try sandbox.archive()).install()

        #expect(sandbox.supportContents == [other, "Models", recent, writing].sorted())
    }

    @Test("A checksum mismatch installs nothing and leaves nothing behind")
    func checksumMismatch() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        var manifest = try sandbox.archive()
        manifest.sha256 = String(repeating: "0", count: 64)
        await #expect(throws: ModelInstallError.checksumMismatch) {
            try await sandbox.installer(manifest).install()
        }
        #expect(sandbox.supportContents.isEmpty)
    }

    @Test("Cancelling removes the partial files", arguments: [
        ModelInstallProgress.Stage.downloading, .verifying, .extracting,
    ])
    func cancellation(at stage: ModelInstallProgress.Stage) async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let manifest = try sandbox.archive()
        let installer = sandbox.installer(manifest)
        // The first report of each stage comes from the installing task itself, so cancelling the current
        // task there is the same as the app cancelling it at that point.
        let task = Task {
            try await installer.install { progress in
                if progress.stage == stage { withUnsafeCurrentTask { $0?.cancel() } }
            }
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(!FileManager.default.fileExists(atPath: sandbox.destination.path))
        #expect(sandbox.supportContents.isEmpty)
    }

    @Test("A new set replaces the installed one, which survives a failed update")
    func replacement() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        try Self.writeFakeSet(Self.entries + ["stale.onnx"], in: sandbox.destination, marker: "old")

        try await sandbox.installer(try sandbox.archive(name: "v2.zip", marker: "new")).install()
        #expect(sandbox.marker() == "new")
        #expect(!FileManager.default.fileExists(atPath: sandbox.destination.appendingPathComponent("stale.onnx").path))
        #expect(sandbox.supportContents == ["Models"])

        var broken = try sandbox.archive(name: "v3.zip", marker: "newer")
        broken.sha256 = String(repeating: "f", count: 64)
        await #expect(throws: ModelInstallError.checksumMismatch) { try await sandbox.installer(broken).install() }
        #expect(sandbox.marker() == "new")
        #expect(sandbox.supportContents == ["Models"])
    }

    @Test("An archive without a complete set is rejected")
    func incompleteArchive() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let manifest = try sandbox.archive(Self.entries.filter { !$0.hasPrefix("lightglue") })
        await #expect(throws: ModelInstallError.incompleteArchive) { try await sandbox.installer(manifest).install() }
        #expect(sandbox.supportContents.isEmpty)
    }

    @Test("A file that is not a zip archive fails to extract")
    func notAnArchive() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let file = sandbox.root.appendingPathComponent("served/garbage.zip")
        try Data((0..<4096).map { UInt8(truncatingIfNeeded: $0 &* 31) }).write(to: file)
        let manifest = try sandbox.manifest(for: file)
        let error = await #expect(throws: ModelInstallError.self) { try await sandbox.installer(manifest).install() }
        guard case .extractionFailed? = error else {
            Issue.record("expected an extraction failure, got \(String(describing: error))")
            return
        }
        #expect(sandbox.supportContents.isEmpty)
    }

    @Test("A missing archive or manifest gives a clear error")
    func missingArchive() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let missing = ModelManifest(release: "test", archive: "missing.zip",
                                    url: sandbox.root.appendingPathComponent("served/missing.zip"), sha256: "00")
        let error = await #expect(throws: ModelInstallError.self) { try await sandbox.installer(missing).install() }
        if case .network? = error {} else { Issue.record("expected a download error, got \(String(describing: error))") }
        #expect(sandbox.supportContents.isEmpty)
        await #expect(throws: ModelInstallError.noManifest) { try await sandbox.installer(nil).install() }
        #expect(ModelInstallError.httpStatus(404).localizedDescription.contains("404"))
    }

    @Test("Removing deletes the installed set and the compiled copies of its Core ML packages")
    func remove() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let installer = sandbox.installer(try sandbox.archive())
        try await installer.install()
        let digest = "0123456789abcdef"
        let compiled = ["lightglue_raco_aliked_k2048_fp16-\(digest).mlmodelc",
                        "raco_aliked_levels_768x1024_fp32-\(digest).mlmodelc"]
        // Another package of the same family, and a file that is not a compiled model.
        let unrelated = ["lightglue_raco_aliked_k2048_fp32-\(digest).mlmodelc", "raco_aliked_levels_768x1024_fp32-notes.txt"]
        for name in compiled + unrelated {
            try FileManager.default.createDirectory(at: sandbox.cache.appendingPathComponent(name),
                                                    withIntermediateDirectories: true)
        }

        try installer.remove()
        #expect(!installer.isInstalled)
        #expect(sandbox.supportContents.isEmpty)
        #expect((try FileManager.default.contentsOfDirectory(atPath: sandbox.cache.path)).sorted() == unrelated.sorted())
        try installer.remove()
    }
}
