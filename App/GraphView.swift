import StitchKit
import SwiftUI

struct GraphView: View {
    @Environment(DiagnosticSession.self) private var session
    @State private var showRejected = true
    @State private var showFailed = false
    @State private var showFootprints = true
    @State private var hovered: PairID?

    var body: some View {
        @Bindable var session = session
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Picker("Matcher", selection: $session.filter) {
                    Text(EvidenceFilter.best.label).tag(EvidenceFilter.best)
                    ForEach(FeatureSource.allCases, id: \.self) { source in
                        Text(source.label).tag(EvidenceFilter.only(source))
                    }
                }
                .frame(minWidth: 140, maxWidth: 260)
                Menu("Show") {
                    Toggle("Photo footprints", isOn: $showFootprints)
                    Toggle("Rejected pairs", isOn: $showRejected)
                    Toggle("Failed attempts", isOn: $showFailed)
                }
                .fixedSize()
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            Divider()
            if let report = session.report {
                GraphCanvas(
                    report: report, filter: session.filter, showFootprints: showFootprints,
                    showRejected: showRejected, showFailed: showFailed, hovered: $hovered
                )
                .overlay(alignment: .topLeading) { Legend().padding(10) }
            }
        }
    }
}

private struct Legend: View {
    var body: some View {
        HStack(spacing: 12) {
            Label("verified", systemImage: "line.diagonal").foregroundStyle(.green)
            Label("rejected", systemImage: "line.diagonal").foregroundStyle(.orange)
            Label("failed", systemImage: "line.diagonal").foregroundStyle(.secondary)
        }
        .font(.caption)
        .labelStyle(.titleAndIcon)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.regularMaterial, in: Capsule())
    }
}

/// Maps the layout plane onto the view.
private struct ViewMapping {
    var scale: Double
    var offset: CGPoint

    init(placement: NodePlacement, size: CGSize, padding: Double = 70) {
        let points = placement.outlines.values.flatMap { $0 }
        let minX = points.map(\.x).min() ?? 0, maxX = points.map(\.x).max() ?? 1
        let minY = points.map(\.y).min() ?? 0, maxY = points.map(\.y).max() ?? 1
        let width = max(maxX - minX, 1), height = max(maxY - minY, 1)
        scale = min((size.width - 2 * padding) / width, (size.height - 2 * padding) / height)
        offset = CGPoint(x: (size.width - width * scale) / 2 - minX * scale,
                         y: (size.height - height * scale) / 2 - minY * scale)
    }

    func callAsFunction(_ p: SIMD2<Double>) -> CGPoint {
        CGPoint(x: p.x * scale + offset.x, y: p.y * scale + offset.y)
    }
}

private struct GraphCanvas: View {
    @Environment(DiagnosticSession.self) private var session
    let report: MatchReport
    let filter: EvidenceFilter
    let showFootprints: Bool
    let showRejected: Bool
    let showFailed: Bool
    @Binding var hovered: PairID?

    var body: some View {
        let placement = GraphLayout.geometric(report, filter: filter)
        let edges = report.evidence(filter: filter)
        let maxInliers = max(edges.map(\.inlierCount).max() ?? 1, 1)
        GeometryReader { geometry in
            let map = ViewMapping(placement: placement, size: geometry.size)
            let visible = edges.filter { isVisible($0) }
            let nodeWidth = Self.nodeWidth(placement: placement, map: map)
            Canvas { context, _ in
                if showFootprints {
                    drawFootprints(context, placement: placement, map: map)
                }
                for pair in visible {
                    drawEdge(context, pair: pair, placement: placement, map: map, maxInliers: maxInliers)
                }
                for image in report.images {
                    drawNode(context, image: image, placement: placement, map: map, width: nodeWidth)
                }
            }
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                switch phase {
                case .active(let location):
                    hovered = nearestEdge(to: location, in: visible, placement: placement, map: map)
                case .ended:
                    hovered = nil
                }
            }
            .onTapGesture { location in
                if let node = nearestNode(to: location, placement: placement, map: map, radius: nodeWidth / 2) {
                    session.selectedImage = node
                } else if let pair = nearestEdge(to: location, in: visible, placement: placement, map: map) {
                    session.openPair(pair)
                }
            }
            .help(hovered.map { helpText(for: $0) } ?? "")
        }
    }

    private func isVisible(_ pair: PairEvidence) -> Bool {
        switch pair.verdict {
        case .verified: true
        case .lowConfidence, .degenerateInliers, .implausibleModel, .modelMismatch: showRejected
        case .tooFewMatches, .noConsistentModel: showFailed && pair.matches.count > 0
        }
    }

    private func helpText(for id: PairID) -> String {
        guard let pair = edgesForHelp.first(where: { PairID($0.a, $0.b) == id }) else { return "" }
        return String(localized: "\(session.name(of: pair.a)) ↔ \(session.name(of: pair.b)): \(pair.inlierCount) inliers of \(pair.matches.count) matches, \(pair.verdict.label). Click to open.")
    }

    private var edgesForHelp: [PairEvidence] { report.evidence(filter: filter) }

    // MARK: Drawing

    private func drawFootprints(_ context: GraphicsContext, placement: NodePlacement, map: ViewMapping) {
        let main = Set(report.graph.components.first ?? [])
        for (id, outline) in placement.outlines {
            var path = Path()
            path.addLines(outline.map { map($0) })
            path.closeSubpath()
            let tint: Color = main.contains(id) ? .accentColor : .gray
            context.fill(path, with: .color(tint.opacity(0.06)))
            context.stroke(path, with: .color(tint.opacity(0.35)), lineWidth: 1)
        }
    }

    private func drawEdge(
        _ context: GraphicsContext, pair: PairEvidence, placement: NodePlacement, map: ViewMapping, maxInliers: Int
    ) {
        guard let a = placement.centers[pair.a], let b = placement.centers[pair.b] else { return }
        let start = map(a), end = map(b)
        var path = Path()
        path.move(to: start)
        path.addLine(to: end)
        let isHovered = hovered == PairID(pair.a, pair.b)
        let strength = log(Double(pair.inlierCount + 1)) / log(Double(maxInliers + 1))
        switch pair.verdict {
        case .verified:
            context.stroke(path, with: .color(.green.opacity(isHovered ? 1 : 0.85)),
                           style: StrokeStyle(lineWidth: (1.5 + 7 * strength) * (isHovered ? 1.4 : 1), lineCap: .round))
        case .lowConfidence, .degenerateInliers, .implausibleModel, .modelMismatch:
            context.stroke(path, with: .color(.orange.opacity(isHovered ? 1 : 0.8)),
                           style: StrokeStyle(lineWidth: isHovered ? 3 : 2, lineCap: .round, dash: [7, 5]))
        case .tooFewMatches, .noConsistentModel:
            context.stroke(path, with: .color(.secondary.opacity(0.6)),
                           style: StrokeStyle(lineWidth: 1, dash: [2, 4]))
        }
        guard pair.inlierCount > 0 || isHovered else { return }
        let middle = CGPoint(x: (start.x + end.x) / 2, y: (start.y + end.y) / 2)
        let label = context.resolve(Text("\(pair.inlierCount)").font(.caption.monospacedDigit().bold())
            .foregroundStyle(.white))
        let size = label.measure(in: CGSize(width: 200, height: 40))
        let badge = CGRect(x: middle.x - size.width / 2 - 6, y: middle.y - size.height / 2 - 2,
                           width: size.width + 12, height: size.height + 4)
        context.fill(Capsule().path(in: badge), with: .color(pair.verdict.color.opacity(0.9)))
        context.draw(label, at: middle)
    }

    /// Thumbnails shrink when photos sit close together, so edges and their badges stay visible.
    static func nodeWidth(placement: NodePlacement, map: ViewMapping) -> Double {
        let points = placement.centers.values.map { map($0) }
        var nearest = Double.infinity
        for (i, p) in points.enumerated() {
            for q in points[(i + 1)...] {
                nearest = min(nearest, hypot(p.x - q.x, p.y - q.y))
            }
        }
        return nearest.isFinite ? max(44, min(120, nearest * 0.5)) : 120
    }

    private func drawNode(
        _ context: GraphicsContext, image: SourceImage, placement: NodePlacement, map: ViewMapping, width: Double
    ) {
        guard let center = placement.centers[image.id] else { return }
        let point = map(center)
        let node = report.graph.nodes.first { $0.id == image.id }
        let height = width * Double(image.pixelSize.height) / Double(max(image.pixelSize.width, 1))
        let rect = CGRect(x: point.x - width / 2, y: point.y - height / 2, width: width, height: height)
        let shape = RoundedRectangle(cornerRadius: 6).path(in: rect)
        if let thumbnail = session.thumbnails[image.id] {
            var clipped = context
            clipped.clip(to: shape)
            clipped.opacity = node?.exclusion == .excludedByUser ? 0.35 : 1
            clipped.draw(Image(decorative: thumbnail, scale: 1), in: rect)
        } else {
            context.fill(shape, with: .color(.gray.opacity(0.3)))
        }
        let selected = session.selectedImage == image.id
        let border: Color = node?.exclusion == nil ? .green : (node?.exclusion == .excludedByUser ? .gray : .red)
        context.stroke(shape, with: .color(border), lineWidth: selected ? 4 : 2)

        var caption = image.name
        if let reason = node?.exclusion { caption += "\n" + reason.label }
        let text = context.resolve(Text(caption).font(.caption2).foregroundStyle(.primary))
        context.draw(text, at: CGPoint(x: point.x, y: rect.maxY + 4), anchor: .top)
    }

    // MARK: Hit testing

    private func nearestNode(
        to location: CGPoint, placement: NodePlacement, map: ViewMapping, radius: Double
    ) -> Int? {
        placement.centers.compactMap { id, center -> (Int, Double)? in
            let p = map(center)
            let d = hypot(p.x - location.x, p.y - location.y)
            return d < radius ? (id, d) : nil
        }.min { $0.1 < $1.1 }?.0
    }

    private func nearestEdge(
        to location: CGPoint, in edges: [PairEvidence], placement: NodePlacement, map: ViewMapping
    ) -> PairID? {
        var best: (PairID, Double)?
        for pair in edges {
            guard let a = placement.centers[pair.a], let b = placement.centers[pair.b] else { continue }
            let d = distance(location, map(a), map(b))
            if d < 8, d < (best?.1 ?? .infinity) { best = (PairID(pair.a, pair.b), d) }
        }
        return best?.0
    }

    private func distance(_ p: CGPoint, _ a: CGPoint, _ b: CGPoint) -> Double {
        let dx = b.x - a.x, dy = b.y - a.y
        let length = dx * dx + dy * dy
        guard length > 0 else { return hypot(p.x - a.x, p.y - a.y) }
        let t = max(0, min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length))
        return hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy))
    }
}
