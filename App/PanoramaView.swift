import StitchKit
import SwiftUI

struct PanoramaView: View {
    @Environment(DiagnosticSession.self) private var session
    @State private var showOutlines = false

    var body: some View {
        @Bindable var session = session
        VStack(spacing: 0) {
            if let panorama = session.panorama {
                HStack(spacing: 14) {
                    Toggle("Photo outlines", isOn: $showOutlines)
                    Toggle("Crop to a rectangle", isOn: $session.cropsPanorama)
                        .disabled(panorama.crop.isEmpty)
                        .help("The largest rectangle without empty areas")
                    Text(caption(panorama)).font(.callout).foregroundStyle(.secondary).lineLimit(1)
                    Spacer(minLength: 0)
                    Button("Stitch Again") { session.stitch() }
                        .disabled(!session.canStitch)
                    Button("Export…") { session.exportPanorama() }
                        .buttonStyle(.borderedProminent)
                        .disabled(session.isExporting)
                }
                .toggleStyle(.checkbox)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                Divider()
                if let staleness = session.panoramaStaleness, !session.isStitching {
                    StaleBanner(staleness: staleness)
                }
            }
            ZStack {
                if let panorama = session.panorama {
                    PanoramaCanvas(panorama: panorama, cropped: session.cropsPanorama && !panorama.crop.isEmpty,
                                   showOutlines: showOutlines)
                        .id(panorama.id)
                        .opacity(session.isStitching ? 0.35 : 1)
                } else if !session.isStitching {
                    placeholder
                }
                if session.isStitching {
                    StitchingCard()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            if let panorama = session.panorama, !session.isStitching {
                PanoramaFooter(panorama: panorama)
            }
        }
    }

    @ViewBuilder private var placeholder: some View {
        if session.reportReadyForStitch && session.stitchableCount < 2 {
            ContentUnavailableView {
                Label("Nothing to stitch", systemImage: "rectangle.dashed")
            } description: {
                Text("No two photos were joined. The Graph view shows why.")
            } actions: {
                Button("Show Graph") { session.tab = .graph }
            }
        } else {
            ContentUnavailableView {
                Label("Not stitched yet", systemImage: "rectangle.split.3x1")
            } description: {
                Text("Stitch joins the photos of the main group into one image.")
            } actions: {
                Button("Stitch") { session.stitch() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!session.canStitch)
            }
        }
    }

    private func caption(_ panorama: Panorama) -> String {
        let crop = session.cropsPanorama && !panorama.crop.isEmpty ? panorama.crop : nil
        let width = crop?.width ?? panorama.size.width, height = crop?.height ?? panorama.size.height
        let megapixels = Double(width * height) / 1_000_000
        return String(localized: "\(width) × \(height) px · \(megapixels.formatted(.number.precision(.fractionLength(1)))) MP · \(panorama.projection.label) · \(panorama.model.label)")
    }
}

private struct StaleBanner: View {
    @Environment(DiagnosticSession.self) private var session
    let staleness: PanoramaStaleness

    var body: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.yellow)
            Text(message).font(.callout)
            Spacer()
            Button("Stitch Again") { session.stitch() }
                .disabled(!session.canStitch)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(.yellow.opacity(0.12))
    }

    private var message: String {
        switch staleness {
        case .photos: String(localized: "Photos, exclusions or the scene mode changed.")
        case .analysis: String(localized: "A new analysis is available.")
        case .settings: String(localized: "The stitch settings changed.")
        }
    }
}

private struct StitchingCard: View {
    @Environment(DiagnosticSession.self) private var session

    var body: some View {
        let event = session.stitchProgress ?? session.progress
        VStack(spacing: 12) {
            if let event, event.total > 0 {
                ProgressView(value: Double(event.completed), total: Double(event.total))
            } else {
                ProgressView()
            }
            Text(event?.label ?? String(localized: "Stitching…")).font(.callout)
            Button("Stop") { session.cancel() }
        }
        .frame(width: 280)
        .padding(20)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct PanoramaFooter: View {
    @Environment(DiagnosticSession.self) private var session
    let panorama: Panorama
    @State private var showsPairs = false
    @State private var showsPhotos = false

    var body: some View {
        let pairs = panorama.leftOutPairs
        let photos = panorama.leftOut.keys.sorted()
        if !panorama.notes.isEmpty || !pairs.isEmpty || !photos.isEmpty {
            ScrollView {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(panorama.notes, id: \.self) { Text($0) }
                    // A few items are listed; many are summed up, with the list one click away.
                    if pairs.count <= 3 {
                        ForEach(pairs, id: \.self) { Text($0.text(session.name(of:))) }
                    } else {
                        DisclosureGroup(isExpanded: $showsPairs) {
                            ForEach(pairs, id: \.self) { Text($0.text(session.name(of:))) }
                        } label: {
                            Text("\(pairs.count) pairs were left out of the alignment because they disagree with the others")
                        }
                    }
                    if photos.count <= 6 {
                        if !photos.isEmpty {
                            Text("Not in the panorama: \(photos.map(describe).joined(separator: ", "))")
                        }
                    } else {
                        DisclosureGroup(isExpanded: $showsPhotos) {
                            ForEach(photos, id: \.self) { Text(describe($0)) }
                        } label: {
                            Text("\(photos.count) photos are not in the panorama: \(reasons(photos))")
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
            }
            .frame(maxHeight: 160)
            .fixedSize(horizontal: false, vertical: true)
            .font(.caption)
            .foregroundStyle(.secondary)
            .background(.bar)
        }
    }

    private func describe(_ id: Int) -> String {
        "\(session.name(of: id)) (\(panorama.leftOut[id]?.label ?? ""))"
    }

    /// "no verified pair (20), separate group (15)", most frequent first.
    private func reasons(_ photos: [Int]) -> String {
        var counts: [ExclusionReason: Int] = [:]
        for id in photos { if let reason = panorama.leftOut[id] { counts[reason, default: 0] += 1 } }
        return counts.sorted { ($0.value, $1.key.rawValue) > ($1.value, $0.key.rawValue) }
            .map { "\($0.key.label) (\($0.value))" }
            .joined(separator: ", ")
    }
}

/// The panorama preview with zoom and pan, and optionally each photo's outline.
private struct PanoramaCanvas: View {
    @Environment(DiagnosticSession.self) private var session
    let panorama: Panorama
    let cropped: Bool
    let showOutlines: Bool

    @State private var zoom = 1.0
    @State private var steadyZoom = 1.0
    @State private var pan = CGSize.zero
    @State private var steadyPan = CGSize.zero
    @State private var hovered: Int?

    private var area: CGRect {
        cropped ? panorama.crop.cgRectValue
            : CGRect(x: 0, y: 0, width: panorama.size.width, height: panorama.size.height)
    }

    /// The preview cut to the shown area.
    private var image: CGImage? {
        let factor = Double(panorama.preview.width) / Double(panorama.size.width)
        let rect = CGRect(x: area.minX * factor, y: area.minY * factor, width: area.width * factor,
                          height: area.height * factor).integral
        return cropped ? panorama.preview.cropping(to: rect) : panorama.preview
    }

    var body: some View {
        GeometryReader { geometry in
            let padding = 16.0
            let fit = min((geometry.size.width - 2 * padding) / area.width, (geometry.size.height - 2 * padding) / area.height)
            let scale = fit * zoom
            let origin = CGPoint(x: (geometry.size.width - area.width * scale) / 2 + pan.width,
                                 y: (geometry.size.height - area.height * scale) / 2 + pan.height)
            let map = { (p: Point2) in
                CGPoint(x: origin.x + (Double(p.x) - area.minX) * scale, y: origin.y + (Double(p.y) - area.minY) * scale)
            }
            Canvas { context, _ in
                if let image {
                    context.draw(Image(decorative: image, scale: 1),
                                 in: CGRect(x: origin.x, y: origin.y, width: area.width * scale, height: area.height * scale))
                }
                guard showOutlines else { return }
                for (id, outline) in panorama.outlines where outline.count >= 3 {
                    var path = Path()
                    path.addLines(outline.map(map))
                    path.closeSubpath()
                    let highlighted = hovered == id
                    context.stroke(path, with: .color(.yellow.opacity(highlighted ? 1 : 0.8)),
                                   style: StrokeStyle(lineWidth: highlighted ? 2.5 : 1.2, dash: highlighted ? [] : [6, 4]))
                    if highlighted, let first = outline.first {
                        context.draw(Text(session.name(of: id)).font(.caption.bold()).foregroundStyle(.yellow),
                                     at: map(first), anchor: .topLeading)
                    }
                }
            }
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                guard showOutlines, case .active(let location) = phase else {
                    hovered = nil
                    return
                }
                // The photo whose outline contains the cursor, the smallest when several do.
                let point = SIMD2(area.minX + (location.x - origin.x) / scale, area.minY + (location.y - origin.y) / scale)
                hovered = panorama.outlines
                    .filter { contains($0.value, point) }
                    .min { area(of: $0.value) < area(of: $1.value) }?
                    .key
            }
            .gesture(MagnifyGesture()
                .onChanged { zoom = max(0.5, min(40, steadyZoom * $0.magnification)) }
                .onEnded { _ in steadyZoom = zoom })
            .simultaneousGesture(DragGesture()
                .onChanged { pan = CGSize(width: steadyPan.width + $0.translation.width,
                                          height: steadyPan.height + $0.translation.height) }
                .onEnded { _ in steadyPan = pan })
            .onTapGesture(count: 2) {
                zoom = 1; steadyZoom = 1; pan = .zero; steadyPan = .zero
            }
            .overlay(alignment: .bottomTrailing) {
                Text("Pinch to zoom, drag to pan, double-click to reset")
                    .font(.caption2).foregroundStyle(.secondary).padding(8)
            }
        }
    }

    private func area(of outline: [Point2]) -> Double {
        var twice = 0.0
        for (i, p) in outline.enumerated() {
            let q = outline[(i + 1) % outline.count]
            twice += Double(p.x) * Double(q.y) - Double(q.x) * Double(p.y)
        }
        return abs(twice) / 2
    }

    /// Even-odd test: outlines of rotation panoramas are curved and need not be convex.
    private func contains(_ outline: [Point2], _ point: SIMD2<Double>) -> Bool {
        var inside = false
        var j = outline.count - 1
        for i in outline.indices {
            let a = outline[i], b = outline[j]
            if (Double(a.y) > point.y) != (Double(b.y) > point.y),
               point.x < Double(b.x - a.x) * (point.y - Double(a.y)) / Double(b.y - a.y) + Double(a.x) {
                inside.toggle()
            }
            j = i
        }
        return inside
    }
}

extension PixelRect {
    var cgRectValue: CGRect { CGRect(x: x, y: y, width: width, height: height) }
}
