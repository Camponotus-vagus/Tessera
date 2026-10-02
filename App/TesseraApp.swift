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

    @MainActor
    func application(_ application: NSApplication, open urls: [URL]) {
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

    var body: some Scene {
        Window("Tessera", id: "main") {
            ContentView()
                .environment(session)
                .task {
                    // Development shortcut: `open -a Tessera --args --analyze` plus files.
                    let analyze = CommandLine.arguments.contains("--analyze")
                    delegate.openFiles = { urls in
                        session.add(urls)
                        if analyze { session.analyze() }
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
            CommandMenu("Analysis") {
                Button("Analyze") { session.analyze() }
                    .keyboardShortcut("r")
                    .disabled(!session.canAnalyze)
                Button("Stop") { session.cancel() }
                    .keyboardShortcut(".")
                    .disabled(!session.isRunning)
                Divider()
                Button("Graph View") { session.tab = .graph }
                    .keyboardShortcut("1")
                Button("Pair View") { session.tab = .pair }
                    .keyboardShortcut("2")
            }
        }
    }
}
