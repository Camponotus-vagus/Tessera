import CryptoKit
import Foundation
import Testing

@testable import StitchKit

/// The installer runs against zips of small fake model files served from file:// URLs, into a temporary
/// folder that stands in for Application Support.
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
            ModelInstaller(manifest: manifest, destination: destination)
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

        let literal = try ModelManifest.decode(Data("""
        {"version": 2, "release": "models-9", "archive": "tessera-models-9.zip",
         "url": "https://example.org/tessera-models-9.zip", "sha256": "ab"}
        """.utf8))
        #expect(literal == ModelManifest(release: "models-9", archive: "tessera-models-9.zip",
                                         url: URL(string: "https://example.org/tessera-models-9.zip")!, sha256: "ab"))
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

    @Test("Removing deletes the installed set")
    func remove() async throws {
        let sandbox = try Sandbox()
        defer { sandbox.remove() }
        let installer = sandbox.installer(try sandbox.archive())
        try await installer.install()
        try installer.remove()
        #expect(!installer.isInstalled)
        #expect(sandbox.supportContents.isEmpty)
        try installer.remove()
    }
}
