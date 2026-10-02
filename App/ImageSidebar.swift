import StitchKit
import SwiftUI

struct ImageSidebar: View {
    @Environment(DiagnosticSession.self) private var session

    var body: some View {
        @Bindable var session = session
        List(selection: $session.selectedImage) {
            Section("Photos (\(session.images.count))") {
                ForEach(session.images) { image in
                    ImageRow(image: image)
                        .tag(image.id)
                        .contextMenu {
                            Button(session.excluded.contains(image.id) ? "Include" : "Exclude") {
                                session.toggleExclusion(image.id)
                            }
                        }
                }
            }
            if let report = session.report, report.graph.components.count > 1 {
                Section("Groups") {
                    ForEach(Array(report.graph.components.enumerated()), id: \.offset) { index, members in
                        Text(index == 0 ? String(localized: "Main: \(members.count) photos")
                             : String(localized: "Group \(index + 1): \(members.map { session.name(of: $0) }.joined(separator: ", "))"))
                            .font(.callout)
                            .foregroundStyle(index == 0 ? .primary : .secondary)
                    }
                }
            }
        }
    }
}

private struct ImageRow: View {
    @Environment(DiagnosticSession.self) private var session
    let image: SourceImage

    var body: some View {
        let excluded = session.excluded.contains(image.id)
        let node = session.report?.graph.nodes.first { $0.id == image.id }
        HStack(spacing: 10) {
            Group {
                if let thumbnail = session.thumbnails[image.id] {
                    Image(decorative: thumbnail, scale: 1).resizable().scaledToFill()
                } else {
                    Rectangle().fill(.quaternary)
                }
            }
            .frame(width: 64, height: 48)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .opacity(excluded ? 0.35 : 1)

            VStack(alignment: .leading, spacing: 2) {
                Text(image.name).lineLimit(1)
                if let node, session.reportIsCurrent {
                    if let reason = node.exclusion {
                        Label(reason.label, systemImage: "xmark.circle")
                            .foregroundStyle(reason == .excludedByUser ? Color.secondary : Color.orange)
                    } else {
                        Label("joined", systemImage: "checkmark.circle")
                            .foregroundStyle(.green)
                    }
                    if node.featureCount > 0 {
                        Text("\(node.featureCount) keypoints").foregroundStyle(.secondary)
                    }
                } else {
                    Text("\(image.pixelSize.width)×\(image.pixelSize.height)")
                        .foregroundStyle(.secondary)
                }
            }
            .font(.callout)
            Spacer(minLength: 0)
            Toggle("Include", isOn: Binding(
                get: { !excluded }, set: { _ in session.toggleExclusion(image.id) }
            ))
            .labelsHidden()
            .toggleStyle(.checkbox)
            .help(excluded ? "Excluded from the analysis" : "Included in the analysis")
        }
        .padding(.vertical, 2)
    }
}
