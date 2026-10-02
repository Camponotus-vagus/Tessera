import StitchKit
import SwiftUI

struct InspectorView: View {
    @Environment(DiagnosticSession.self) private var session

    var body: some View {
        @Bindable var session = session
        Form {
            Section("Scene") {
                Picker("Mode", selection: $session.configuration.mode) {
                    ForEach(StitchMode.allCases, id: \.self) { Text($0.label).tag($0) }
                }
                Text(modeHelp).font(.caption).foregroundStyle(.secondary)
            }

            Section("Matcher") {
                // The last matcher that would run cannot be switched off.
                Toggle("RootSIFT", isOn: Binding(
                    get: { session.configuration.sources.contains(.rootSIFT) },
                    set: { session.setSource(.rootSIFT, enabled: $0) }
                ))
                .disabled(session.activeSources == [.rootSIFT])
                Toggle("RaCo + LightGlue", isOn: Binding(
                    get: { session.activeSources.contains(.racoLightGlue) },
                    set: { session.setSource(.racoLightGlue, enabled: $0) }
                ))
                .disabled(!session.lightGlueAvailable || session.activeSources == [.racoLightGlue])
                if session.lightGlueAvailable {
                    InstalledModelsView()
                } else {
                    ModelDownloadView()
                }
                // Only the choices that the installed set and this build can run, and a picker only when
                // there is something to choose.
                if let models = session.configuration.learnedModels {
                    let configuration = session.configuration
                    let extractors = runnable(models.extractorBackends(for: configuration),
                                              current: configuration.extractorBackend, among: LearnedBackend.allCases)
                    if extractors.count > 1 {
                        Picker("Extractor", selection: $session.configuration.extractorBackend) {
                            ForEach(extractors, id: \.self) { Text($0.label).tag($0) }
                        }
                    }
                    let matchers = runnable(models.matcherBackends(for: configuration),
                                            current: configuration.matcherBackend, among: LearnedBackend.allCases)
                    if matchers.count > 1 {
                        Picker("Matcher", selection: $session.configuration.matcherBackend) {
                            ForEach(matchers, id: \.self) { Text($0.label).tag($0) }
                        }
                    }
                    let precisions = runnable(models.matcherPrecisions(for: configuration),
                                              current: configuration.matcherPrecision, among: ["fp16", "fp32"])
                    if precisions.count > 1 {
                        Picker("Matcher precision", selection: $session.configuration.matcherPrecision) {
                            ForEach(precisions, id: \.self) { precision in
                                if precision == "fp16" {
                                    Text("fp16 (faster)").tag(precision)
                                } else {
                                    Text("fp32 (same as ONNX)").tag(precision)
                                }
                            }
                        }
                    }
                    if LearnedModelSet.onnxAvailable {
                        Toggle("Less memory for ONNX Runtime", isOn: $session.configuration.lightGlueLowMemory)
                    }
                }
            }

            Section("Pairs") {
                Picker("Compare", selection: $session.configuration.pairSelection) {
                    Text("Candidates").tag(PairSelection.proposed)
                    Text("All").tag(PairSelection.all)
                }
                Text("With more than 4 photos: consecutive shots, best neighbours by descriptor affinity, and pairs that join groups left apart.")
                    .font(.caption).foregroundStyle(.secondary)
                Stepper(value: $session.configuration.proposalNeighbours, in: 1...6) {
                    LabeledContent("Neighbours per photo", value: "\(session.configuration.proposalNeighbours)")
                }
            }

            Section("Thresholds") {
                LabeledContent("Ratio test") {
                    Slider(value: $session.configuration.ratio, in: 0.6...0.95, step: 0.05) {
                        Text(String(format: "%.2f", session.configuration.ratio))
                    }
                }
                Stepper(value: $session.configuration.inlierThreshold, in: 1...10, step: 0.5) {
                    LabeledContent("Tolerance", value: String(format: String(localized: "%.1f px at 1 MP"), session.configuration.inlierThreshold))
                }
                Stepper(value: $session.configuration.minimumInliers, in: 6...60) {
                    LabeledContent("Minimum inliers", value: "\(session.configuration.minimumInliers)")
                }
                Stepper(value: $session.configuration.siftMegapixels, in: 0.5...6, step: 0.5) {
                    LabeledContent("SIFT resolution", value: String(format: "%.1f MP", session.configuration.siftMegapixels))
                }
            }

            if let report = session.report {
                Section("Result") {
                    let joined = report.graph.components.first.map { $0.count >= 2 ? $0.count : 0 } ?? 0
                    LabeledContent("Joined photos", value: String(localized: "\(joined) of \(report.images.count)"))
                    LabeledContent("Groups", value: "\(report.graph.components.count)")
                    LabeledContent("Verified pairs",
                                   value: "\(report.evidence(filter: .best).filter { $0.verdict == .verified }.count)")
                    if let candidates = report.candidates {
                        let total = report.images.count - report.excludedByUser.count
                        LabeledContent("Pairs compared", value: String(localized: "\(candidates.count) of \(total * (total - 1) / 2)"))
                    }
                    if let duration = session.lastDuration {
                        LabeledContent("Time", value: duration.formatted(.units(allowed: [.seconds], fractionalPart: .show(length: 1))))
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    /// The supported options in their usual order, plus the current one even when it is not supported,
    /// so that the picker still shows the selection.
    private func runnable<T: Equatable>(_ supported: [T], current: T, among all: [T]) -> [T] {
        all.filter { supported.contains($0) || $0 == current }
    }

    private var modeHelp: String {
        switch session.configuration.mode {
        case .auto: String(localized: "Tries translation, similarity, affine and homography, and keeps the simplest model that explains the matches.")
        case .rotation: String(localized: "The camera rotates in place: homography between pairs.")
        case .plane: String(localized: "Tiles of a plane moved under the lens: translation, similarity or affine.")
        case .document: String(localized: "A flat original photographed in pieces from different viewpoints: affine or homography.")
        }
    }
}
