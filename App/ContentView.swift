import AppKit
import StitchKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(DiagnosticSession.self) private var session
    @State private var importing = false
    @State private var exporting = false
    @State private var showInspector = true

    var body: some View {
        @Bindable var session = session
        NavigationSplitView {
            ImageSidebar()
                .navigationSplitViewColumnWidth(260)
        } detail: {
            Group {
                if session.images.isEmpty {
                    EmptyState(importing: $importing)
                } else if session.report == nil && session.tab != .panorama {
                    ContentUnavailableView {
                        Label("Ready", systemImage: "point.3.connected.trianglepath.dotted")
                    } description: {
                        Text("\(session.images.count) photos imported. Stitch joins them into one image; Analyze only shows how they connect.")
                    } actions: {
                        HStack {
                            Button("Analyze") { session.analyze() }
                                .disabled(!session.canAnalyze)
                            Button("Stitch") { session.stitch() }
                                .buttonStyle(.borderedProminent)
                                .disabled(!session.canStitch)
                        }
                    }
                } else {
                    switch session.tab {
                    case .graph: GraphView()
                    case .pair: PairView()
                    case .panorama: PanoramaView()
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .inspector(isPresented: $showInspector) {
            InspectorView()
                .inspectorColumnWidth(min: 300, ideal: 300, max: 300)
        }
        // One minimum for the whole window, inspector included, larger than what the content needs: a
        // minimum that changes with the inspector or the content while AppKit updates the window's
        // constraints makes it abort.
        .frame(minWidth: 1200, minHeight: 640)
        .toolbar {
            ToolbarItem(placement: .navigation) {
                Button("Import", systemImage: "plus") { importing = true }
            }
            ToolbarItem(placement: .principal) {
                Picker("View", selection: $session.tab) {
                    Text("Graph").tag(DetailTab.graph)
                    Text("Pair").tag(DetailTab.pair)
                    Text("Panorama").tag(DetailTab.panorama)
                }
                .pickerStyle(.segmented)
                .disabled(session.report == nil && session.panorama == nil && !session.isStitching)
            }
            ToolbarItem(placement: .status) {
                StatusBar()
            }
            ToolbarItemGroup(placement: .primaryAction) {
                if session.isBusy {
                    Button("Stop", systemImage: "stop.fill") { session.cancel() }
                } else {
                    Button("Analyze", systemImage: "point.3.connected.trianglepath.dotted") { session.analyze() }
                        .disabled(!session.canAnalyze)
                    Button("Stitch", systemImage: "rectangle.split.3x1") { session.stitch() }
                        .disabled(!session.canStitch)
                }
                Menu("Export", systemImage: "square.and.arrow.up") {
                    Button("Export Panorama…") { session.exportPanorama() }
                        .disabled(session.panorama == nil || session.isExporting)
                    Button("Export Report…") { exporting = true }
                        .disabled(session.report == nil)
                }
                .disabled(session.panorama == nil && session.report == nil)
                Button("Settings", systemImage: "sidebar.trailing") { showInspector.toggle() }
            }
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.image], allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { session.add(urls) }
        }
        .fileExporter(
            isPresented: $exporting, document: session.report.map { ReportDocument($0.withoutLocalPaths()) },
            contentType: .json, defaultFilename: "match-report.json"
        ) { result in
            if case .failure(let error) = result { session.errorMessage = error.localizedDescription }
        }
        .dropDestination(for: URL.self) { urls, _ in
            let images = urls.filter { UTType(filenameExtension: $0.pathExtension)?.conforms(to: .image) == true }
            session.add(images)
            return !images.isEmpty
        }
        .alert("Error", isPresented: Binding(
            get: { session.errorMessage != nil }, set: { if !$0 { session.errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(session.errorMessage ?? "")
        }
    }
}

private struct EmptyState: View {
    @Environment(DiagnosticSession.self) private var session
    @Binding var importing: Bool

    var body: some View {
        ContentUnavailableView {
            Label("No photos", systemImage: "photo.on.rectangle.angled")
        } description: {
            Text("Drop the photos to join here, or import them.")
        } actions: {
            VStack(spacing: 16) {
                Button("Import Photos…") { importing = true }
                    .buttonStyle(.borderedProminent)
                if !session.lightGlueAvailable {
                    ModelDownloadView(compact: true)
                }
            }
        }
    }
}

private struct StatusBar: View {
    @Environment(DiagnosticSession.self) private var session

    var body: some View {
        if session.isBusy || session.isExporting || (session.report != nil && !session.reportIsCurrent) {
            HStack(spacing: 10) {
                if session.isBusy {
                    ProgressView().controlSize(.small)
                    Text(progressText)
                } else if session.isExporting {
                    ProgressView().controlSize(.small)
                    Text("Exporting…")
                } else {
                    Image(systemName: "exclamationmark.triangle")
                    Text("Needs a new analysis")
                }
            }
            .font(.callout)
            .lineLimit(1)
        } else if let url = session.lastExport {
            // Short, so that a long file name does not push the toolbar buttons into the overflow menu.
            HStack(spacing: 10) {
                Image(systemName: "checkmark.circle")
                Text("Saved")
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    .buttonStyle(.link)
            }
            .font(.callout)
            .lineLimit(1)
            .help(url.path)
        } else if let problem = session.report?.learnedProblem {
            // The learned matcher could not load for these settings; the reason is in the tooltip.
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle")
                Text("RootSIFT only: RaCo + LightGlue did not run")
            }
            .font(.callout)
            .lineLimit(1)
            .help(problem)
        }
    }

    private var progressText: String {
        guard let progress = session.stitchProgress ?? session.progress else {
            return session.isStitching ? String(localized: "Stitching…") : String(localized: "Analysing…")
        }
        return progress.label
    }
}

struct ReportDocument: FileDocument {
    static let readableContentTypes: [UTType] = [.json]
    var report: MatchReport

    init(_ report: MatchReport) {
        self.report = report
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else { throw CocoaError(.fileReadCorruptFile) }
        report = try MatchReport.decode(data)
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: try report.jsonData())
    }
}
