import AppKit
import StitchKit
import SwiftUI

/// Missing learned models: what they are for and a Download button, the progress with Cancel, or the error
/// with Retry. `compact` puts it on one line for the empty window.
struct ModelDownloadView: View {
    @Environment(DiagnosticSession.self) private var session
    var compact = false

    var body: some View {
        switch session.modelDownload {
        case .idle:
            layout(stacked: true) {
                Text("RaCo + LightGlue needs its models, downloaded once.")
                    .font(compact ? .callout : .caption).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } action: {
                Button("Download Models") { session.downloadModels() }
            }
        case .running(let progress):
            layout(stacked: false) {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: progress?.fraction)
                    Text(status(progress)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
                .frame(width: compact ? 240 : nil)
            } action: {
                Button("Cancel") { session.cancelModelDownload() }
            }
        case .failed(let message):
            layout(stacked: true) {
                Text(message).font(compact ? .callout : .caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } action: {
                Button("Retry") { session.downloadModels() }
            }
        }
    }

    /// Text and button side by side in the empty window; in the inspector the button goes below the text
    /// when `stacked`.
    @ViewBuilder
    private func layout(stacked: Bool, @ViewBuilder _ content: () -> some View, @ViewBuilder action: () -> some View)
        -> some View {
        if compact {
            HStack(spacing: 12) {
                content()
                action().fixedSize()
            }
            .frame(maxWidth: 600)
        } else if stacked {
            VStack(alignment: .leading, spacing: 6) {
                content()
                action()
            }
        } else {
            HStack {
                content()
                action()
            }
        }
    }

    private func status(_ progress: ModelInstallProgress?) -> String {
        guard let progress else { return String(localized: "Downloading…") }
        switch progress.stage {
        case .downloading:
            guard progress.received > 0 else { return String(localized: "Downloading…") }
            let received = progress.received.formatted(.byteCount(style: .file))
            guard let expected = progress.expected else { return String(localized: "Downloading… \(received)") }
            return String(localized: "Downloading… \(received) of \(expected.formatted(.byteCount(style: .file)))")
        case .verifying:
            return String(localized: "Checking the download…")
        case .extracting:
            return String(localized: "Installing…")
        }
    }
}

/// Installed learned models: where they are and, when the app downloaded them, a way to remove them.
struct InstalledModelsView: View {
    @Environment(DiagnosticSession.self) private var session
    @State private var confirmingRemoval = false

    var body: some View {
        HStack {
            Button("Show in Finder") {
                if let directory = session.configuration.learnedModels?.directory {
                    NSWorkspace.shared.activateFileViewerSelecting([directory])
                }
            }
            if session.modelsAreRemovable {
                Button("Remove Models", role: .destructive) { confirmingRemoval = true }
                    .disabled(session.isRunning)
            }
        }
        .confirmationDialog("Remove the downloaded models?", isPresented: $confirmingRemoval) {
            Button("Remove Models", role: .destructive) { session.removeModels() }
        } message: {
            Text("RaCo + LightGlue stays off until you download them again.")
        }
    }
}
