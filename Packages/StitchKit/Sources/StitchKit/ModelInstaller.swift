import CryptoKit
import Darwin
import Foundation

/// The release asset holding the default model set, as listed in tools/models.json.
public struct ModelManifest: Sendable, Hashable, Codable {
    public var version: Int
    public var release: String
    public var archive: String
    public var url: URL
    public var sha256: String
    /// Size of the archive in bytes, shown before the download starts (absent from older manifests).
    public var size: Int64?

    public init(version: Int = 2, release: String, archive: String, url: URL, sha256: String, size: Int64? = nil) {
        self.version = version
        self.release = release
        self.archive = archive
        self.url = url
        self.sha256 = sha256
        self.size = size
    }

    public static func decode(_ data: Data) throws -> ModelManifest {
        try JSONDecoder().decode(ModelManifest.self, from: data)
    }

    /// Files tried, in order: `models.json` in the app's resources, then `tools/models.json` in the working
    /// directory and its parents (a source checkout).
    public static var searchPaths: [URL] {
        var paths: [URL] = []
        if let bundled = Bundle.main.url(forResource: "models", withExtension: "json") {
            paths.append(bundled)
        }
        var folder = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true).standardized
        while true {
            paths.append(folder.appendingPathComponent("tools/models.json"))
            let parent = folder.deletingLastPathComponent().standardized
            if parent.path == folder.path { break }
            folder = parent
        }
        return paths
    }

    /// The first manifest in `searchPaths` that can be read.
    public static func standard() -> ModelManifest? {
        for url in searchPaths {
            if let data = try? Data(contentsOf: url), let manifest = try? decode(data) { return manifest }
        }
        return nil
    }
}

public enum ModelInstallError: Error, LocalizedError, Sendable, Equatable {
    case noManifest
    /// No usable answer: offline, timed out, unknown host, missing file.
    case network(String)
    /// The server answered with an error status.
    case httpStatus(Int)
    case checksumMismatch
    case extractionFailed(String)
    /// The archive unpacked, but not into a usable model set.
    case incompleteArchive
    /// Files could not be written, moved or removed.
    case fileSystem(String)

    public var errorDescription: String? {
        switch self {
        case .noManifest:
            String(localized: "The list of models to download (models.json) was not found in the app or in a source checkout.")
        case .network(let reason):
            String(localized: "The models could not be downloaded: \(reason)")
        case .httpStatus(404):
            String(localized: "The models could not be downloaded: the server answered 404 (not found). The model release may not be public yet.")
        case .httpStatus(let status):
            String(localized: "The models could not be downloaded: the server answered \(status) (\(HTTPURLResponse.localizedString(forStatusCode: status))).")
        case .checksumMismatch:
            String(localized: "The downloaded archive does not match its SHA-256 checksum and was discarded. Try again.")
        case .extractionFailed(let reason):
            String(localized: "The model archive could not be extracted: \(reason)")
        case .incompleteArchive:
            String(localized: "The model archive does not contain a complete model set.")
        case .fileSystem(let reason):
            String(localized: "The models could not be installed: \(reason)")
        }
    }
}

public struct ModelInstallProgress: Sendable, Equatable {
    public enum Stage: Sendable, Equatable {
        case downloading
        case verifying
        case extracting
    }

    public var stage: Stage
    /// Bytes of the archive received so far, and its size once the server has sent it.
    public var received: Int64
    public var expected: Int64?

    public init(stage: Stage, received: Int64, expected: Int64?) {
        self.stage = stage
        self.received = received
        self.expected = expected
    }

    public var fraction: Double? {
        guard let expected, expected > 0 else { return nil }
        return min(1, Double(received) / Double(expected))
    }
}

/// Downloads the archive named in a manifest, checks its SHA-256, unpacks it and puts the model set in
/// `destination`, replacing any set already there. Cancel the calling task to stop it.
public struct ModelInstaller: Sendable {
    public var manifest: ModelManifest?
    public var destination: URL
    public var keypoints: Int
    /// Configuration of the URLSession of each download; the tests answer through a URLProtocol.
    var sessionConfiguration: @Sendable () -> URLSessionConfiguration = { .ephemeral }
    /// Where `compiledCoreMLModel` keeps the compiled packages.
    var coreMLCache = cacheDirectory("CoreML")

    /// Hidden work folders that nothing has written to for this long were left by an installer that
    /// stopped without cleaning up (a crash, a forced quit) and are deleted by the next one.
    static let abandonedAfter: TimeInterval = 10 * 60

    /// `~/Library/Application Support/Tessera/Models`, one of `LearnedModelSet.searchPaths`.
    public static var standardDestination: URL {
        URL.applicationSupportDirectory.appendingPathComponent("Tessera/Models", isDirectory: true)
    }

    public init(manifest: ModelManifest? = .standard(), destination: URL = standardDestination, keypoints: Int = 2048) {
        self.manifest = manifest
        self.destination = destination
        self.keypoints = keypoints
    }

    public var isInstalled: Bool { LearnedModelSet(directory: destination, keypoints: keypoints).isUsable }

    @discardableResult
    public func install(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> LearnedModelSet {
        guard let manifest else { throw ModelInstallError.noManifest }
        try Task.checkCancellation()
        let manager = FileManager.default
        removeAbandonedWork()
        // Everything partial lives in one hidden folder next to the destination: on the same volume, so the
        // new set moves into place with a rename, and removed whatever happens.
        let work = sibling(of: destination)
        try fileSystem { try manager.createDirectory(at: work, withIntermediateDirectories: true) }
        defer { try? manager.removeItem(at: work) }

        let archive = work.appendingPathComponent("archive.zip")
        progress(ModelInstallProgress(stage: .downloading, received: 0, expected: nil))
        let (size, digest) = try await download(manifest.url, to: archive, expected: manifest.size, progress: progress)

        progress(ModelInstallProgress(stage: .verifying, received: size, expected: size))
        guard digest == manifest.sha256.lowercased() else { throw ModelInstallError.checksumMismatch }
        try Task.checkCancellation()

        progress(ModelInstallProgress(stage: .extracting, received: size, expected: size))
        try Task.checkCancellation()
        let staging = work.appendingPathComponent("Models", isDirectory: true)
        try await extract(archive, into: staging, log: work.appendingPathComponent("ditto.log"))
        guard LearnedModelSet(directory: staging, keypoints: keypoints).isUsable else {
            throw ModelInstallError.incompleteArchive
        }
        try Task.checkCancellation()

        try fileSystem { try Self.replace(destination, with: staging, backup: work.appendingPathComponent("previous")) }
        return LearnedModelSet(directory: destination, keypoints: keypoints)
    }

    /// Deletes the installed set and the compiled copies of its Core ML packages. The folder is renamed
    /// first, so a removal that stops halfway does not leave a partial set where the engine looks for models.
    public func remove() throws {
        let manager = FileManager.default
        guard manager.fileExists(atPath: destination.path) else { return }
        let packages = ((try? manager.contentsOfDirectory(atPath: destination.path)) ?? [])
            .filter { $0.hasSuffix(".mlpackage") }.map { String($0.dropLast(".mlpackage".count)) }
        let trash = sibling(of: destination)
        try fileSystem {
            try manager.moveItem(at: destination, to: trash)
            try manager.removeItem(at: trash)
        }
        // Cache entries are named after their package, then a digest (see compiledCoreMLModel).
        for entry in (try? manager.contentsOfDirectory(atPath: coreMLCache.path)) ?? []
        where entry.hasSuffix(".mlmodelc") && packages.contains(where: { entry.hasPrefix("\($0)-") }) {
            try? manager.removeItem(at: coreMLCache.appendingPathComponent(entry))
        }
    }

    private func sibling(of url: URL) -> URL {
        url.deletingLastPathComponent()
            .appendingPathComponent(".\(url.lastPathComponent)-\(UUID().uuidString)", isDirectory: true)
    }

    /// Deletes the hidden work folders next to `destination` that nothing has written to for
    /// `abandonedAfter`. More recent ones may belong to another installer running now.
    func removeAbandonedWork(now: Date = Date()) {
        let manager = FileManager.default
        let prefix = ".\(destination.lastPathComponent)-"
        let keys: Set<URLResourceKey> = [.contentModificationDateKey]
        let siblings = (try? manager.contentsOfDirectory(at: destination.deletingLastPathComponent(),
                                                          includingPropertiesForKeys: Array(keys))) ?? []
        for folder in siblings where folder.lastPathComponent.hasPrefix(prefix)
            && UUID(uuidString: String(folder.lastPathComponent.dropFirst(prefix.count))) != nil {
            // A folder's date changes only when entries come and go; the archive's with every write.
            let contents = (try? manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: Array(keys))) ?? []
            let latest = ([folder] + contents).compactMap { try? $0.resourceValues(forKeys: keys).contentModificationDate }
                .max()
            if let latest, now.timeIntervalSince(latest) < Self.abandonedAfter { continue }
            try? manager.removeItem(at: folder)
        }
    }

    private func fileSystem<T>(_ body: () throws -> T) throws -> T {
        do {
            return try body()
        } catch let error as ModelInstallError {
            throw error
        } catch {
            throw ModelInstallError.fileSystem(error.localizedDescription)
        }
    }

    /// Streams `url` into `file` and returns the number of bytes and their SHA-256. `expected` is the size
    /// reported as long as the server has not sent one.
    private func download(_ url: URL, to file: URL, expected: Int64?,
                          progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> (size: Int64, sha256: String) {
        let delegate = DownloadDelegate(target: file, expected: expected, progress: progress)
        let session = URLSession(configuration: sessionConfiguration(), delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        // A data task, not a download task: the bytes go straight into the work folder, so nothing is left
        // in the temporary directory when the download stops.
        let task = session.dataTask(with: url)
        let result: (size: Int64, sha256: String)
        do {
            result = try await withTaskCancellationHandler {
                try await withCheckedThrowingContinuation { delegate.start(task, $0) }
            } onCancel: {
                task.cancel()
            }
        } catch {
            if Task.isCancelled { throw CancellationError() }
            if let error = error as? ModelInstallError { throw error }
            throw ModelInstallError.network(error.localizedDescription)
        }
        try Task.checkCancellation()
        return result
    }

    private func extract(_ archive: URL, into folder: URL, log: URL) async throws {
        let status: Int32
        do {
            status = try await runTool("/usr/bin/ditto", ["-x", "-k", archive.path, folder.path], errors: log)
        } catch {
            throw ModelInstallError.extractionFailed(error.localizedDescription)
        }
        try Task.checkCancellation()
        guard status == 0 else {
            let message = ((try? String(contentsOf: log, encoding: .utf8)) ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            throw ModelInstallError.extractionFailed(message.isEmpty ? "ditto exited with status \(status)" : message)
        }
    }

    static func sha256(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hex(hasher.finalize())
    }

    static func hex(_ digest: SHA256.Digest) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Puts `staging` at `destination`. An existing set is swapped out in one atomic rename where the volume
    /// supports it; otherwise it is moved aside, and put back if the new set cannot take its place.
    static func replace(_ destination: URL, with staging: URL, backup: URL) throws {
        let manager = FileManager.default
        guard manager.fileExists(atPath: destination.path) else {
            try manager.moveItem(at: staging, to: destination)
            return
        }
        // After the swap the old set sits at `staging` and goes away with the work folder.
        if renamex_np(staging.path, destination.path, UInt32(RENAME_SWAP)) == 0 { return }
        try manager.moveItem(at: destination, to: backup)
        do {
            try manager.moveItem(at: staging, to: destination)
        } catch {
            try? manager.moveItem(at: backup, to: destination)
            throw error
        }
    }
}

/// Runs a tool to completion with its standard error in `errors`; cancelling the task terminates it.
private func runTool(_ path: String, _ arguments: [String], errors: URL) async throws -> Int32 {
    FileManager.default.createFile(atPath: errors.path, contents: nil)
    let log = try FileHandle(forWritingTo: errors)
    defer { try? log.close() }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: path)
    process.arguments = arguments
    process.standardOutput = FileHandle.nullDevice
    process.standardError = log
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { continuation in
            process.terminationHandler = { continuation.resume(returning: $0.terminationStatus) }
            do {
                try process.run()
            } catch {
                process.terminationHandler = nil
                continuation.resume(throwing: error)
            }
        }
    } onCancel: {
        // A process that has not started yet runs to the end; the caller checks for cancellation afterwards.
        if process.isRunning { process.terminate() }
    }
}

/// Receives one data task's events: checks the HTTP status before any of the body arrives, writes the body
/// to `target` while hashing it, reports progress, and resumes the continuation with the size and SHA-256
/// or the error.
private final class DownloadDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    typealias Outcome = (size: Int64, sha256: String)

    private let target: URL
    private let fallbackExpected: Int64?
    private let progress: @Sendable (ModelInstallProgress) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Outcome, any Error>?
    /// Set when the task ends before `start` has stored the continuation (a task cancelled straight away).
    private var outcome: Result<Outcome, any Error>?
    // Used only by the delegate methods, which the session calls one at a time on its own queue.
    private var file: FileHandle?
    private var hasher = SHA256()
    private var received: Int64 = 0
    private var expected: Int64?
    private var reported: Int64 = 0
    /// Why the delegate stopped the task itself: an error status, or a write that failed.
    private var failure: ModelInstallError?

    init(target: URL, expected: Int64?, progress: @escaping @Sendable (ModelInstallProgress) -> Void) {
        self.target = target
        fallbackExpected = expected
        self.progress = progress
    }

    func start(_ task: URLSessionTask, _ continuation: CheckedContinuation<Outcome, any Error>) {
        lock.lock()
        if let outcome {
            lock.unlock()
            continuation.resume(with: outcome)
            return
        }
        self.continuation = continuation
        lock.unlock()
        task.resume()
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        // An error page is not downloaded: a 404 fails at once.
        if let status = (response as? HTTPURLResponse)?.statusCode, !(200..<300).contains(status) {
            failure = .httpStatus(status)
            completionHandler(.cancel)
            return
        }
        let manager = FileManager.default
        try? manager.removeItem(at: target)
        guard manager.createFile(atPath: target.path, contents: nil),
              let file = FileHandle(forWritingAtPath: target.path) else {
            failure = .fileSystem("\(target.path) cannot be created")
            completionHandler(.cancel)
            return
        }
        self.file = file
        expected = response.expectedContentLength > 0 ? response.expectedContentLength : fallbackExpected
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard let file, failure == nil else { return }
        do {
            try file.write(contentsOf: data)
        } catch {
            failure = .fileSystem(error.localizedDescription)
            dataTask.cancel()
            return
        }
        hasher.update(data: data)
        received += Int64(data.count)
        // A report every 256 KB and at the end is plenty for a progress bar.
        guard received - reported >= 256 * 1024 || received == expected else { return }
        reported = received
        progress(ModelInstallProgress(stage: .downloading, received: received, expected: expected))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        var closeError: (any Error)?
        do {
            try file?.close()
        } catch {
            closeError = error
        }
        file = nil
        let result: Result<Outcome, any Error>
        if let failure {
            result = .failure(failure)
        } else if let error {
            result = .failure(error)
        } else if let closeError {
            result = .failure(ModelInstallError.fileSystem(closeError.localizedDescription))
        } else if task.response == nil {
            result = .failure(ModelInstallError.network("no response"))
        } else {
            result = .success((received, ModelInstaller.hex(hasher.finalize())))
        }
        lock.lock()
        let continuation = self.continuation
        self.continuation = nil
        if continuation == nil { outcome = result }
        lock.unlock()
        continuation?.resume(with: result)
    }
}
