import Foundation

/// Chooses which pairs the matchers examine, from a quick descriptor affinity between photos.
enum PairProposal {
    struct Key: Hashable, Comparable, Sendable {
        let a: Int
        let b: Int

        init(_ x: Int, _ y: Int) {
            a = min(x, y)
            b = max(x, y)
        }

        static func < (lhs: Key, rhs: Key) -> Bool { (lhs.a, lhs.b) < (rhs.a, rhs.b) }
    }

    /// Consecutive shots, the best partners of every photo, and a maximum spanning tree of the
    /// affinity so that no photo is left without a path to the others.
    static func propose(images: [SourceImage], affinity: [Key: Int], neighbours: Int) -> [CandidatePair] {
        var reasons: [Key: Set<CandidateReason>] = [:]
        func add(_ key: Key, _ reason: CandidateReason) { reasons[key, default: []].insert(reason) }

        let order = shootingOrder(images)
        for (first, second) in zip(order, order.dropFirst()) {
            add(Key(first.id, second.id), .consecutive)
        }

        for image in images {
            let partners = images.filter { $0.id != image.id }
                .map { Key(image.id, $0.id) }
                .filter { (affinity[$0] ?? 0) > 0 }
                .sorted { (affinity[$0] ?? 0, $1) > (affinity[$1] ?? 0, $0) }
                .prefix(neighbours)
            for key in partners { add(key, .neighbour) }
        }

        // Kruskal on decreasing affinity; pairs without affinity come last and still connect the tree.
        var keys: [Key] = []
        for i in images.indices {
            for j in images.indices where j > i { keys.append(Key(images[i].id, images[j].id)) }
        }
        keys.sort { (affinity[$0] ?? 0, $1) > (affinity[$1] ?? 0, $0) }
        var parent = Dictionary(uniqueKeysWithValues: images.map { ($0.id, $0.id) })
        func root(_ id: Int) -> Int {
            var node = id
            while let next = parent[node], next != node { node = next }
            return node
        }
        for key in keys {
            let ra = root(key.a), rb = root(key.b)
            guard ra != rb else { continue }
            parent[max(ra, rb)] = min(ra, rb)
            add(key, .spanningTree)
        }

        return reasons.keys.sorted().map { key in
            CandidatePair(a: key.a, b: key.b, affinity: affinity[key],
                          reasons: reasons[key, default: []].sorted { $0.rawValue < $1.rawValue })
        }
    }

    /// Untried pairs that would join groups the verified pairs left apart, best affinity first.
    /// At most `perGroupPair` per pair of groups and `limit` in all, and only pairs with some affinity,
    /// so many unrelated photos do not turn the round into matching every pair.
    static func bridges(
        groups: [Set<Int>], affinity: [Key: Int], tried: Set<Key>, perGroupPair: Int, limit: Int
    ) -> [CandidatePair] {
        var options: [Key] = []
        for i in groups.indices {
            for j in groups.indices where j > i {
                var keys: [Key] = []
                for x in groups[i] {
                    for y in groups[j] where !tried.contains(Key(x, y)) && (affinity[Key(x, y)] ?? 0) > 0 {
                        keys.append(Key(x, y))
                    }
                }
                keys.sort { (affinity[$0] ?? 0, $1) > (affinity[$1] ?? 0, $0) }
                options += keys.prefix(perGroupPair)
            }
        }
        options.sort { (affinity[$0] ?? 0, $1) > (affinity[$1] ?? 0, $0) }
        return options.prefix(max(0, limit)).map {
            CandidatePair(a: $0.a, b: $0.b, affinity: affinity[$0], reasons: [.bridge])
        }
    }

    static func components(_ ids: [Int], edges: [Key]) -> [Set<Int>] {
        var parent = Dictionary(uniqueKeysWithValues: ids.map { ($0, $0) })
        func root(_ id: Int) -> Int {
            var node = id
            while let next = parent[node], next != node { node = next }
            return node
        }
        for edge in edges where parent[edge.a] != nil && parent[edge.b] != nil {
            let ra = root(edge.a), rb = root(edge.b)
            if ra != rb { parent[max(ra, rb)] = min(ra, rb) }
        }
        var groups: [Int: Set<Int>] = [:]
        for id in ids { groups[root(id), default: []].insert(id) }
        return groups.keys.sorted().map { groups[$0]! }
    }

    /// By capture date when every photo has one, otherwise by name; ties go to the id.
    static func shootingOrder(_ images: [SourceImage]) -> [SourceImage] {
        let dated = images.allSatisfy { $0.captureDate != nil }
        return images.sorted { lhs, rhs in
            if dated, let l = lhs.captureDate, let r = rhs.captureDate, l != r { return l < r }
            let names = lhs.name.localizedStandardCompare(rhs.name)
            if names != .orderedSame { return names == .orderedAscending }
            return lhs.id < rhs.id
        }
    }
}
