import StitchKit
import SwiftUI

struct PairView: View {
    @Environment(DiagnosticSession.self) private var session
    @State private var layers = PairLayers()

    var body: some View {
        @Bindable var session = session
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                PairPicker()
                Picker("Matcher", selection: $session.filter) {
                    Text(EvidenceFilter.best.label).tag(EvidenceFilter.best)
                    ForEach(FeatureSource.allCases, id: \.self) { source in
                        Text(source.label).tag(EvidenceFilter.only(source))
                    }
                }
                .labelsHidden()
                .frame(minWidth: 120, maxWidth: 200)
                Menu("Layers") {
                    Toggle("Keypoints", isOn: $layers.keypoints)
                    Toggle("Tentative", isOn: $layers.tentative)
                    Toggle("Inliers", isOn: $layers.inliers)
                    Toggle("Outliers", isOn: $layers.outliers)
                    Toggle("Overlap", isOn: $layers.overlap)
                }
                .fixedSize()
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            Divider()
            if let id = session.selectedPair, let evidence = session.evidence(for: id) {
                PairCaption(evidence: evidence)
                // A new report gives a fresh canvas, so no hover index outlives the matches it pointed to.
                PairCanvas(evidence: evidence, layers: layers)
                    .id("\(evidence.id)-\(session.report?.createdAt.timeIntervalSince1970 ?? 0)")
            } else {
                ContentUnavailableView("Choose a pair", systemImage: "rectangle.split.2x1",
                                       description: Text("From the menu above, or click an edge in the graph."))
            }
        }
        .onAppear {
            if let pair = session.selectedPair { session.openPair(pair) }
        }
    }
}

struct PairLayers: Equatable {
    var keypoints = true
    var tentative = false
    var inliers = true
    var outliers = true
    var overlap = true
}

private struct PairPicker: View {
    @Environment(DiagnosticSession.self) private var session

    var body: some View {
        let pairs = (session.report?.evidence(filter: session.filter) ?? [])
            .sorted { lhs, rhs in
                lhs.verdict == rhs.verdict ? lhs.inlierCount > rhs.inlierCount : lhs.verdict == .verified
            }
        Picker("Pair", selection: Binding(
            get: { session.selectedPair },
            set: { if let pair = $0 { session.openPair(pair) } }
        )) {
            ForEach(pairs) { pair in
                Text("\(session.name(of: pair.a)) ↔ \(session.name(of: pair.b))  ·  \(pair.inlierCount) inliers  ·  \(pair.verdict.label)")
                    .tag(Optional(PairID(pair.a, pair.b)))
            }
        }
        .frame(minWidth: 160, maxWidth: .infinity)
    }
}

private struct PairCaption: View {
    @Environment(DiagnosticSession.self) private var session
    let evidence: PairEvidence

    var body: some View {
        HStack(spacing: 18) {
            Label(evidence.verdict.label, systemImage: evidence.verdict == .verified ? "checkmark.seal.fill" : "xmark.seal")
                .foregroundStyle(evidence.verdict.color)
            Text(evidence.source.label)
            Text("\(evidence.matches.count) matches, \(evidence.inlierCount) inliers")
            if let model = evidence.chosenModel { Text(model.label) }
            if let fit = evidence.chosenFit { Text(String(format: String(localized: "mean error %.1f px"), fit.rmse)) }
            Text(String(format: String(localized: "coverage %.0f%%"), evidence.inlierCoverage * 100))
            if evidence.log10NFA.isFinite {
                Text(evidence.log10NFA < 0 ? "NFA 10^\(Int(evidence.log10NFA.rounded()))" : "NFA ≥ 1")
                    .help("Expected number of false alarms: below 1, chance does not explain the pair.")
            }
            Spacer()
        }
        .font(.callout.monospacedDigit())
        .padding(.horizontal, 14)
        .padding(.vertical, 6)
        .background(.bar)
    }
}

private struct PairCanvas: View {
    @Environment(DiagnosticSession.self) private var session
    let evidence: PairEvidence
    let layers: PairLayers

    @State private var zoom = 1.0
    @State private var steadyZoom = 1.0
    @State private var pan = CGSize.zero
    @State private var steadyPan = CGSize.zero
    @State private var hovered: Int?

    var body: some View {
        guard let report = session.report,
              let imageA = report.images.first(where: { $0.id == evidence.a }),
              let imageB = report.images.first(where: { $0.id == evidence.b })
        else { return AnyView(EmptyView()) }
        let inliers = Set(evidence.chosenFit?.inliers.map(Int.init) ?? [])
        let shown = hovered.flatMap { evidence.matches.indices.contains($0) ? $0 : nil }
        let keypointsA = report.features.first { $0.imageID == evidence.a && $0.source == evidence.source }?.keypoints ?? []
        let keypointsB = report.features.first { $0.imageID == evidence.b && $0.source == evidence.source }?.keypoints ?? []

        return AnyView(GeometryReader { geometry in
            let frame = PairFrame(size: geometry.size, a: imageA.pixelSize, b: imageB.pixelSize, zoom: zoom, pan: pan)
            Canvas { context, _ in
                draw(image: evidence.a, in: frame.rectA, context: context)
                draw(image: evidence.b, in: frame.rectB, context: context)

                if layers.overlap {
                    overlay(evidence.overlapA, frame.pointA, context)
                    overlay(evidence.overlapB, frame.pointB, context)
                }
                if layers.keypoints {
                    dots(keypointsA, frame.pointA, context)
                    dots(keypointsB, frame.pointB, context)
                }
                if layers.tentative {
                    lines(evidence.matches.indices, frame, .yellow.opacity(0.3), 0.8, context)
                }
                if layers.outliers {
                    lines(evidence.matches.indices.filter { !inliers.contains($0) }, frame, .red.opacity(0.35), 0.8,
                          context)
                }
                if layers.inliers {
                    lines(inliers.sorted(), frame, .green.opacity(0.85), 1.2, context)
                }
                if let hovered = shown {
                    lines([hovered], frame, .white, 3, context)
                    lines([hovered], frame, inliers.contains(hovered) ? .green : .red, 1.5, context)
                    let match = evidence.matches[hovered]
                    for point in [frame.pointA(match.a), frame.pointB(match.b)] {
                        context.stroke(Circle().path(in: CGRect(x: point.x - 6, y: point.y - 6, width: 12, height: 12)),
                                       with: .color(.white), lineWidth: 2)
                    }
                }
            }
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                if case .active(let location) = phase {
                    hovered = nearestMatch(to: location, frame: frame)
                } else {
                    hovered = nil
                }
            }
            .gesture(MagnifyGesture()
                .onChanged { zoom = max(0.5, min(12, steadyZoom * $0.magnification)) }
                .onEnded { _ in steadyZoom = zoom })
            .simultaneousGesture(DragGesture()
                .onChanged { pan = CGSize(width: steadyPan.width + $0.translation.width,
                                          height: steadyPan.height + $0.translation.height) }
                .onEnded { _ in steadyPan = pan })
            .onTapGesture(count: 2) {
                zoom = 1; steadyZoom = 1; pan = .zero; steadyPan = .zero
            }
            .overlay(alignment: .topTrailing) {
                if let hovered = shown { MatchDetail(evidence: evidence, index: hovered, isInlier: inliers.contains(hovered)) }
            }
            .overlay(alignment: .bottomTrailing) {
                Text("Pinch to zoom, drag to pan, double-click to reset")
                    .font(.caption2).foregroundStyle(.secondary).padding(8)
            }
        })
    }

    private func draw(image id: Int, in rect: CGRect, context: GraphicsContext) {
        if let image = session.previews[id] ?? session.thumbnails[id] {
            context.draw(Image(decorative: image, scale: 1), in: rect)
        } else {
            context.fill(Path(rect), with: .color(.gray.opacity(0.2)))
        }
    }

    private func overlay(_ polygon: [Point2], _ map: (Point2) -> CGPoint, _ context: GraphicsContext) {
        guard polygon.count >= 3 else { return }
        var path = Path()
        path.addLines(polygon.map(map))
        path.closeSubpath()
        context.fill(path, with: .color(.cyan.opacity(0.10)))
        context.stroke(path, with: .color(.cyan.opacity(0.8)), style: StrokeStyle(lineWidth: 1.5, dash: [6, 4]))
    }

    private func dots(_ keypoints: [Keypoint], _ map: (Point2) -> CGPoint, _ context: GraphicsContext) {
        var path = Path()
        for k in keypoints {
            let p = map(Point2(x: k.x, y: k.y))
            path.addEllipse(in: CGRect(x: p.x - 1.3, y: p.y - 1.3, width: 2.6, height: 2.6))
        }
        context.fill(path, with: .color(Color(red: 0.55, green: 0.8, blue: 1).opacity(0.55)))
    }

    private func lines<S: Sequence>(_ indices: S, _ frame: PairFrame, _ color: Color, _ width: Double,
                                    _ context: GraphicsContext) where S.Element == Int {
        var path = Path()
        for index in indices {
            let match = evidence.matches[index]
            path.move(to: frame.pointA(match.a))
            path.addLine(to: frame.pointB(match.b))
        }
        context.stroke(path, with: .color(color), lineWidth: width)
    }

    private func nearestMatch(to location: CGPoint, frame: PairFrame) -> Int? {
        var best: (Int, Double)?
        for (index, match) in evidence.matches.enumerated() {
            for point in [frame.pointA(match.a), frame.pointB(match.b)] {
                let d = hypot(point.x - location.x, point.y - location.y)
                if d < 9, d < (best?.1 ?? .infinity) { best = (index, d) }
            }
        }
        return best?.0
    }
}

/// Side-by-side placement of the two photos, with zoom and pan.
private struct PairFrame {
    let rectA: CGRect
    let rectB: CGRect
    let scaleA: Double
    let scaleB: Double

    init(size: CGSize, a: PixelSize, b: PixelSize, zoom: Double, pan: CGSize) {
        let gap = 14.0
        let available = CGSize(width: (size.width - gap) / 2, height: size.height)
        func fitted(_ s: PixelSize) -> Double {
            min(available.width / Double(s.width), available.height / Double(s.height))
        }
        let fitA = fitted(a), fitB = fitted(b)
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        func transformed(_ rect: CGRect) -> CGRect {
            CGRect(x: center.x + (rect.minX - center.x) * zoom + pan.width,
                   y: center.y + (rect.minY - center.y) * zoom + pan.height,
                   width: rect.width * zoom, height: rect.height * zoom)
        }
        let heightA = Double(a.height) * fitA, heightB = Double(b.height) * fitB
        rectA = transformed(CGRect(x: available.width - Double(a.width) * fitA, y: (size.height - heightA) / 2,
                                   width: Double(a.width) * fitA, height: heightA))
        rectB = transformed(CGRect(x: available.width + gap, y: (size.height - heightB) / 2,
                                   width: Double(b.width) * fitB, height: heightB))
        scaleA = fitA * zoom
        scaleB = fitB * zoom
    }

    func pointA(_ p: Point2) -> CGPoint {
        CGPoint(x: rectA.minX + Double(p.x) * scaleA, y: rectA.minY + Double(p.y) * scaleA)
    }

    func pointB(_ p: Point2) -> CGPoint {
        CGPoint(x: rectB.minX + Double(p.x) * scaleB, y: rectB.minY + Double(p.y) * scaleB)
    }
}

private struct MatchDetail: View {
    let evidence: PairEvidence
    let index: Int
    let isInlier: Bool

    var body: some View {
        let match = evidence.matches[index]
        VStack(alignment: .leading, spacing: 3) {
            Text("Match \(index + 1)").bold()
            Text(isInlier ? "inlier" : "outlier").foregroundStyle(isInlier ? .green : .red)
            Text(String(format: String(localized: "score %.2f"), match.score))
            if let error = reprojectionError(match) { Text(String(format: String(localized: "error %.1f px"), error)) }
            Text(String(format: "A (%.0f, %.0f)  B (%.0f, %.0f)", match.a.x, match.a.y, match.b.x, match.b.y))
                .foregroundStyle(.secondary)
        }
        .font(.caption.monospacedDigit())
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        .padding(10)
    }

    private func reprojectionError(_ match: TentativeMatch) -> Double? {
        guard let h = evidence.chosenFit?.transform else { return nil }
        let x = Double(match.a.x), y = Double(match.a.y)
        let w = h[6] * x + h[7] * y + h[8]
        guard abs(w) > 1e-9 else { return nil }
        let px = (h[0] * x + h[1] * y + h[2]) / w, py = (h[3] * x + h[4] * y + h[5]) / w
        return hypot(px - Double(match.b.x), py - Double(match.b.y))
    }
}
