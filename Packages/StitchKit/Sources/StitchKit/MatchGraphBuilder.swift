import Foundation

enum MatchGraphBuilder {
    static func build(
        images: [SourceImage], features: [ImageFeatures], pairs: [PairEvidence], excluded: Set<Int>,
        unreadable: Set<Int> = [], configuration: PipelineConfiguration
    ) -> MatchGraph {
        let edges = pairs.map { pair in
            GraphEdge(a: pair.a, b: pair.b, source: pair.source, matches: pair.matches.count,
                      inliers: pair.inlierCount, confidence: pair.confidence, accepted: pair.verdict == .verified)
        }

        // Union-find over accepted edges between active images.
        let active = images.map(\.id).filter { !excluded.contains($0) && !unreadable.contains($0) }
        var parent = Dictionary(uniqueKeysWithValues: active.map { ($0, $0) })
        func root(_ id: Int) -> Int {
            var node = id
            while let next = parent[node], next != node {
                parent[node] = parent[next]
                node = next
            }
            return node
        }
        for edge in edges where edge.accepted && parent[edge.a] != nil && parent[edge.b] != nil {
            let ra = root(edge.a), rb = root(edge.b)
            if ra != rb { parent[max(ra, rb)] = min(ra, rb) }
        }
        var groups: [Int: [Int]] = [:]
        for id in active {
            groups[root(id), default: []].append(id)
        }
        var strengthOf: [Int: Int] = [:]
        for edge in edges where edge.accepted && parent[edge.a] != nil {
            strengthOf[root(edge.a), default: 0] += edge.inliers
        }
        // Largest first, then strongest, then lowest id, so equal inputs always give the same main group.
        let components = groups.map { (strength: strengthOf[$0.key] ?? 0, members: $0.value.sorted()) }
            .sorted { lhs, rhs in
                if lhs.members.count != rhs.members.count { return lhs.members.count > rhs.members.count }
                if lhs.strength != rhs.strength { return lhs.strength > rhs.strength }
                return lhs.members[0] < rhs.members[0]
            }
            .map(\.members)
        let componentOf = Dictionary(uniqueKeysWithValues: components.enumerated().flatMap { index, members in
            members.map { ($0, index) }
        })
        // A main group of one photo has joined nothing.
        let mainIsJoined = (components.first?.count ?? 0) >= 2

        var pairsOf: [Int: [PairEvidence]] = [:]
        for pair in pairs {
            pairsOf[pair.a, default: []].append(pair)
            pairsOf[pair.b, default: []].append(pair)
        }
        var featuresOf: [Int: [ImageFeatures]] = [:]
        for set in features { featuresOf[set.imageID, default: []].append(set) }

        let nodes = images.map { image -> GraphNode in
            let sets = featuresOf[image.id, default: []]
            let count = sets.map(\.keypoints.count).max() ?? 0
            // The learned extractor always returns its full quota, even on a blank frame; SIFT does not.
            let reference = sets.first { $0.source == .rootSIFT }?.keypoints.count ?? count
            let component = componentOf[image.id] ?? -1
            var reason: ExclusionReason?
            if excluded.contains(image.id) {
                reason = .excludedByUser
            } else if unreadable.contains(image.id) {
                reason = .unreadable
            } else if component != 0 || !mainIsJoined {
                let touching = pairsOf[image.id, default: []]
                if reference < configuration.minimumFeatures {
                    reason = .tooFewFeatures
                } else if touching.contains(where: { $0.verdict == .verified }) {
                    reason = .separateGroup
                } else if touching.contains(where: { [.lowConfidence, .degenerateInliers, .implausibleModel, .modelMismatch].contains($0.verdict) }) {
                    reason = .weakLinkOnly
                } else {
                    reason = .noVerifiedPair
                }
            }
            return GraphNode(id: image.id, featureCount: count, component: component, exclusion: reason)
        }
        return MatchGraph(nodes: nodes, edges: edges, components: components)
    }
}
