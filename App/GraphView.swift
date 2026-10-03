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
                var taken: [CGRect] = []
                for image in report.images {
                    taken += drawNode(context, image: image, placement: placement, map: map, width: nodeWidth)
                }
                // Counts last, so that no line or photo covers them, and verified pairs first, so that they
                // keep the middle of their edge when two counts or a photo compete for it.
                let ranked = visible.sorted {
                    ($0.verdict == .verified ? 1 : 0, $0.inlierCount) > ($1.verdict == .verified ? 1 : 0, $1.inlierCount)
                }
                for pair in ranked {
                    if let badge = drawBadge(context, pair: pair, placement: placement, map: map, avoiding: taken) {
                        taken.append(badge)
                    }
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
    }

    /// Draws the pair's count on its edge, at the middle or, when that is covered, at the first point along
    /// the edge clear of `taken` (photos, their names and counts already drawn), or else where it covers
    /// them least. Returns where it went.
    private func drawBadge(
        _ context: GraphicsContext, pair: PairEvidence, placement: NodePlacement, map: ViewMapping, avoiding taken: [CGRect]
    ) -> CGRect? {
        guard let a = placement.centers[pair.a], let b = placement.centers[pair.b] else { return nil }
        let start = map(a), end = map(b)
        guard pair.inlierCount > 0 || hovered == PairID(pair.a, pair.b) else { return nil }
        let label = context.resolve(Text("\(pair.inlierCount)").font(.caption.monospacedDigit().bold())
            .foregroundStyle(.white))
        let size = label.measure(in: CGSize(width: 200, height: 40))
        func badge(at t: Double) -> CGRect {
            let x = start.x + (end.x - start.x) * t, y = start.y + (end.y - start.y) * t
            return CGRect(x: x - size.width / 2 - 6, y: y - size.height / 2 - 2, width: size.width + 12, height: size.height + 4)
        }
        let candidates = [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8].map(badge(at:))
        func covered(_ candidate: CGRect) -> Double {
            taken.reduce(0) { total, other in
                let overlap = other.intersection(candidate)
                return overlap.isNull ? total : total + overlap.width * overlap.height
            }
        }
        let rect = candidates.first { covered($0) == 0 } ?? candidates.min { covered($0) < covered($1) }!
        context.fill(Capsule().path(in: rect), with: .color(pair.verdict.color.opacity(0.9)))
        context.draw(label, at: CGPoint(x: rect.midX, y: rect.midY))
        return rect
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

    /// Draws the photo and its name, and returns the rectangles they cover.
    private func drawNode(
        _ context: GraphicsContext, image: SourceImage, placement: NodePlacement, map: ViewMapping, width: Double
    ) -> [CGRect] {
        guard let rect = nodeRect(image, placement: placement, map: map, width: width) else { return [] }
        let point = CGPoint(x: rect.midX, y: rect.midY)
        let node = report.graph.nodes.first { $0.id == image.id }
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
        let size = text.measure(in: CGSize(width: 400, height: 100))
        return [rect, CGRect(x: point.x - size.width / 2, y: rect.maxY + 4, width: size.width, height: size.height)]
    }

    /// The photo's thumbnail on the canvas, without its caption.
    private func nodeRect(_ image: SourceImage, placement: NodePlacement, map: ViewMapping, width: Double) -> CGRect? {
        guard let center = placement.centers[image.id] else { return nil }
        let point = map(center)
        let height = width * Double(image.pixelSize.height) / Double(max(image.pixelSize.width, 1))
        return CGRect(x: point.x - width / 2, y: point.y - height / 2, width: width, height: height)
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
