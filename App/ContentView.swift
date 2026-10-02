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
                } else if session.report == nil {
                    ContentUnavailableView {
                        Label("Ready to analyse", systemImage: "point.3.connected.trianglepath.dotted")
                    } description: {
                        Text("\(session.images.count) photos imported. Press Analyze to find how they connect.")
                    } actions: {
                        Button("Analyze") { session.analyze() }
                            .buttonStyle(.borderedProminent)
                            .disabled(!session.canAnalyze)
                    }
                } else {
                    switch session.tab {
                    case .graph: GraphView()
                    case .pair: PairView()
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
                }
                .pickerStyle(.segmented)
                .disabled(session.report == nil)
            }
            ToolbarItem(placement: .status) {
                StatusBar()
            }
            ToolbarItemGroup(placement: .primaryAction) {
                if session.isRunning {
                    Button("Stop", systemImage: "stop.fill") { session.cancel() }
                } else {
                    Button("Analyze", systemImage: "play.fill") { session.analyze() }
                        .disabled(!session.canAnalyze)
                }
                Button("Export Report", systemImage: "square.and.arrow.up") { exporting = true }
                    .disabled(session.report == nil)
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
        if session.isRunning || (session.report != nil && !session.reportIsCurrent) {
            HStack(spacing: 10) {
                if session.isRunning {
                    ProgressView().controlSize(.small)
                    Text(progressText)
                } else {
                    Image(systemName: "exclamationmark.triangle")
                    Text("Needs a new analysis")
                }
            }
            .font(.callout)
            .lineLimit(1)
        }
    }

    private var progressText: String {
        guard let progress = session.progress else { return String(localized: "Analysing…") }
        let stage = switch progress.stage {
        case "sift": String(localized: "RootSIFT keypoints")
        case "lightglue-extract": String(localized: "RaCo-ALIKED keypoints")
        case "affinity": String(localized: "Choosing pairs")
        case "sift-match": String(localized: "RootSIFT matching")
        case "lightglue-match": String(localized: "LightGlue matching")
        case "verify": String(localized: "Geometric verification")
        case "bridge": String(localized: "Pairs between separate groups")
        default: progress.stage
        }
        return progress.total > 0 ? "\(stage): \(progress.completed)/\(progress.total)" : stage
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
