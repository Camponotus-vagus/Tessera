import AppKit
import SwiftUI

/// Receives files opened from Finder, the Dock icon or the command line.
final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor var openFiles: (([URL]) -> Void)? {
        didSet {
            if let openFiles, !pending.isEmpty {
                openFiles(pending)
                pending = []
            }
        }
    }

    @MainActor private var pending: [URL] = []

    /// Cancels a model download in progress and returns its task, so that quitting can wait for the
    /// partial files to be removed.
    @MainActor var cancelModelDownload: (() -> Task<Void, Never>?)?

    /// The file passed on to SwiftUI, which comes back through `application(_:open:)` and is already open.
    @MainActor private var forwarded: URL?

    /// SwiftUI answers an "open documents" event by bringing forward and laying out the window once for every
    /// file: with 552 files opened from Finder the window took 47 seconds to appear. All the files go to the
    /// session at once, and SwiftUI sees only the first, which is enough to open or raise its window.
    func applicationWillFinishLaunching(_ notification: Notification) {
        NSAppleEventManager.shared().setEventHandler(
            self, andSelector: #selector(openDocuments(_:withReply:)),
            forEventClass: AEEventClass(kCoreEventClass), andEventID: AEEventID(kAEOpenDocuments))
    }

    @MainActor @objc private func openDocuments(_ event: NSAppleEventDescriptor, withReply reply: NSAppleEventDescriptor) {
        guard let list = event.paramDescriptor(forKeyword: keyDirectObject) else { return }
        let items = list.numberOfItems > 0 ? (1...list.numberOfItems).compactMap { list.atIndex($0) } : [list]
        let urls = items.compactMap { $0.coerce(toDescriptorType: typeFileURL)?.fileURLValue }
        guard let first = urls.first else { return }
        application(NSApp, open: urls)
        forwarded = first
        NSApp.delegate?.application?(NSApp, open: [first])
    }

    @MainActor
    func application(_ application: NSApplication, open urls: [URL]) {
        if let forwarded, urls == [forwarded] {
            self.forwarded = nil
            return
        }
        if let openFiles {
            openFiles(urls)
        } else {
            pending += urls
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    @MainActor
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let download = cancelModelDownload?() else { return .terminateNow }
        Task { @MainActor in
            await download.value
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

@main
struct TesseraApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var session = DiagnosticSession()

    init() {
        // The window is not restorable, so the state AppKit saves holds no window. Restoring it after a crash
        // or a forced quit would leave the app without its window: never restore.
        UserDefaults.standard.register(defaults: ["ApplePersistenceIgnoreState": true])
    }

    var body: some Scene {
        Window("Tessera", id: "main") {
            ContentView()
                .environment(session)
                .task {
                    // Development shortcuts: `open -a Tessera <files> --args --analyze` (or --stitch).
                    let analyze = CommandLine.arguments.contains("--analyze")
                    let stitch = CommandLine.arguments.contains("--stitch")
                    delegate.openFiles = { urls in
                        session.add(urls) {
                            if stitch { session.stitch() } else if analyze { session.analyze() }
                        }
                    }
                    delegate.cancelModelDownload = { session.cancelModelDownload() }
                }
        }
        .defaultSize(width: 1440, height: 900)
        .restorationBehavior(.disabled)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Session") { session.reset() }
                    .keyboardShortcut("n")
            }
            CommandGroup(replacing: .importExport) {
                Button("Export Panorama…") { session.exportPanorama() }
                    .keyboardShortcut("e")
                    .disabled(session.panorama == nil || session.isExporting)
            }
            CommandMenu("Analysis") {
                Button("Analyze") { session.analyze() }
                    .keyboardShortcut("r")
                    .disabled(!session.canAnalyze)
                Button("Stitch") { session.stitch() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                    .disabled(!session.canStitch)
                Button("Stop") { session.cancel() }
                    .keyboardShortcut(".")
                    .disabled(!session.isBusy || session.isStopping)
                Divider()
                Button("Graph View") { session.tab = .graph }
                    .keyboardShortcut("1")
                Button("Pair View") { session.tab = .pair }
                    .keyboardShortcut("2")
                Button("Panorama View") { session.tab = .panorama }
                    .keyboardShortcut("3")
            }
        }
    }
}
