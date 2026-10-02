import Foundation
import simd

/// Which matcher's evidence a view shows.
public enum EvidenceFilter: Hashable, Sendable {
    /// Per pair, the most convincing evidence from any matcher.
    case best
    case only(FeatureSource)
}

extension MatchReport {
    /// One evidence per image pair, chosen according to `filter`.
    public func evidence(filter: EvidenceFilter) -> [PairEvidence] {
        var chosen: [String: PairEvidence] = [:]
        for pair in pairs {
            if case .only(let source) = filter, pair.source != source { continue }
            let key = "\(min(pair.a, pair.b))-\(max(pair.a, pair.b))"
            if let current = chosen[key], !Self.isBetter(pair, than: current) { continue }
            chosen[key] = pair
        }
        return chosen.values.sorted { ($0.a, $0.b) < ($1.a, $1.b) }
    }

    static func isBetter(_ lhs: PairEvidence, than rhs: PairEvidence) -> Bool {
        let lv = lhs.verdict == .verified, rv = rhs.verdict == .verified
        if lv != rv { return lv }
        return lhs.inlierCount > rhs.inlierCount
    }
}

public struct NodePlacement: Sendable {
    /// Centre of each image in a common plane (pixels of the component's reference image).
    public var centers: [Int: SIMD2<Double>]
    /// Outline of each image in the same plane, for drawing footprints.
    public var outlines: [Int: [SIMD2<Double>]]
}

public enum GraphLayout {
    /// Places every component by chaining pairwise transforms along a maximum spanning tree
    /// (weights = inliers) grown from the graph's centre; components and isolated photos are laid out
    /// side by side. Homographies are replaced by a similarity fitted where the two photos overlap,
    /// so long rotation chains neither blow up nor send corners behind the camera.
    public static func geometric(_ report: MatchReport, filter: EvidenceFilter) -> NodePlacement {
        let evidence = report.evidence(filter: filter).filter { $0.verdict == .verified && $0.chosenFit != nil }
        let sizes = Dictionary(uniqueKeysWithValues: report.images.map { ($0.id, $0.pixelSize) })
        var centers: [Int: SIMD2<Double>] = [:]
        var outlines: [Int: [SIMD2<Double>]] = [:]
        var cursorX = 0.0
        let gap = 0.25 * Double(report.images.map { max($0.pixelSize.width, $0.pixelSize.height) }.max() ?? 1000)

        for component in report.graph.components + isolated(report) {
            let members = Set(component)
            let local = evidence.filter { members.contains($0.a) && members.contains($0.b) }
            let root = centre(of: component, edges: local)
            var toRoot: [Int: simd_double3x3] = [root: matrix_identity_double3x3]
            var unusable = Set<String>()
            while true {
                let frontier = local.filter { !unusable.contains($0.id) && (toRoot[$0.a] != nil) != (toRoot[$0.b] != nil) }
                guard let edge = frontier.max(by: { $0.inlierCount < $1.inlierCount }) else { break }
                guard let bToA = placement(of: edge), abs(bToA.determinant) > 1e-12 else {
                    unusable.insert(edge.id)
                    continue
                }
                if let placedA = toRoot[edge.a] {
                    toRoot[edge.b] = placedA * bToA
                } else if let placedB = toRoot[edge.b] {
                    toRoot[edge.a] = placedB * bToA.inverse
                }
            }
            // Images of the component that could not be chained (should not happen) sit at the root.
            for id in component where toRoot[id] == nil {
                toRoot[id] = matrix_identity_double3x3
            }

            var placed: [Int: [SIMD2<Double>]] = [:]
            for id in component {
                guard let size = sizes[id], let m = toRoot[id] else { continue }
                let outline = PlaneGeometry.rectangle(size).compactMap { PlaneGeometry.apply(m, $0.x, $0.y) }
                if outline.count == 4 { placed[id] = outline }
            }
            let all = placed.values.flatMap { $0 }
            guard let minX = all.map(\.x).min(), let minY = all.map(\.y).min(), let maxX = all.map(\.x).max()
            else { continue }
            let shift = SIMD2(cursorX - minX, -minY)
            for (id, outline) in placed {
                let moved = outline.map { $0 + shift }
                outlines[id] = moved
                centers[id] = moved.reduce(SIMD2(0, 0), +) / Double(moved.count)
            }
            cursorX += (maxX - minX) + gap
        }
        return NodePlacement(centers: centers, outlines: outlines)
    }

    /// Affine map from B's pixels into A's. Affine models are inverted exactly; a homography becomes the
    /// similarity that matches it at the centre of the overlap, where the two photos agree best.
    static func placement(of pair: PairEvidence) -> simd_double3x3? {
        guard let fit = pair.chosenFit else { return nil }
        let aToB = PlaneGeometry.matrix(fit.transform)
        guard fit.model == .homography else {
            return abs(aToB.determinant) > 1e-12 ? aToB.inverse : nil
        }
        let anchors = pair.overlapA.isEmpty ? fit.inliers.map { pair.matches[Int($0)].a } : pair.overlapA
        guard !anchors.isEmpty else { return nil }
        let inA = anchors.reduce(SIMD2<Double>(0, 0)) { $0 + SIMD2(Double($1.x), Double($1.y)) } / Double(anchors.count)
        guard let inB = PlaneGeometry.apply(aToB, inA.x, inA.y),
              let jacobian = PlaneGeometry.jacobian(aToB, at: inA) else { return nil }
        // Closest similarity to the Jacobian: rotation from its antisymmetric part, scale from its determinant.
        let j = jacobian
        let angle = atan2(j.columns.0.y - j.columns.1.x, j.columns.0.x + j.columns.1.y)
        let scale = abs(j.determinant).squareRoot()
        guard scale > 1e-9, scale.isFinite else { return nil }
        // B to A: rotate by -angle, divide by scale, and move inB onto inA.
        let c = cos(angle) / scale, s = sin(angle) / scale
        let tx = inA.x - (c * inB.x + s * inB.y), ty = inA.y - (-s * inB.x + c * inB.y)
        return simd_double3x3(SIMD3(c, -s, 0), SIMD3(s, c, 0), SIMD3(tx, ty, 1))
    }

    /// The member with the smallest eccentricity in the verified graph, so chains are as short as possible;
    /// ties go to the most connected image, then to the lowest id.
    static func centre(of component: [Int], edges: [PairEvidence]) -> Int {
        var neighbours: [Int: [Int]] = [:]
        for edge in edges {
            neighbours[edge.a, default: []].append(edge.b)
            neighbours[edge.b, default: []].append(edge.a)
        }
        func eccentricity(_ start: Int) -> Int {
            var distance = [start: 0], queue = [start], index = 0
            while index < queue.count {
                let node = queue[index]
                index += 1
                for next in neighbours[node, default: []] where distance[next] == nil {
                    distance[next] = distance[node]! + 1
                    queue.append(next)
                }
            }
            return distance.values.max() ?? 0
        }
        let ranked = component.map { id in (id: id, eccentricity: eccentricity(id), weight: weight(of: id, in: edges)) }
        return ranked.min { lhs, rhs in
            if lhs.eccentricity != rhs.eccentricity { return lhs.eccentricity < rhs.eccentricity }
            if lhs.weight != rhs.weight { return lhs.weight > rhs.weight }
            return lhs.id < rhs.id
        }?.id ?? component[0]
    }

    private static func weight(of id: Int, in evidence: [PairEvidence]) -> Int {
        evidence.filter { $0.a == id || $0.b == id }.reduce(0) { $0 + $1.inlierCount }
    }

    /// Images excluded by the user are not in any component; give each its own slot.
    private static func isolated(_ report: MatchReport) -> [[Int]] {
        let inComponents = Set(report.graph.components.flatMap { $0 })
        return report.images.map(\.id).filter { !inComponents.contains($0) }.map { [$0] }
    }
}
