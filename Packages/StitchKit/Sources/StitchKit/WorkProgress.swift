import CStitchCore
import Foundation
import Synchronization

/// Receives a native call's progress for `leaf`; the call also stops when the task is cancelled. The callback
/// runs on the calling thread, so the call must run inside the task that Stop cancels.
final class NativeProgress {
    let leaf: WorkLeaf

    init(_ leaf: WorkLeaf) { self.leaf = leaf }

    /// The callback; valid while this object lives (withExtendedLifetime around the call).
    var control: sc_progress {
        sc_progress(report: { context, fraction in
            Unmanaged<NativeProgress>.fromOpaque(context!).takeUnretainedValue().leaf.report(fraction)
            return Task.isCancelled ? 1 : 0
        }, context: Unmanaged.passUnretained(self).toOpaque())
    }
}

/// Progress of a stage whose work shows only as it goes (the alignment, the provisional layouts). Each step, when
/// it starts, takes of what is left of the bar the share its expected cost has in all the cost still expected, its
/// own included: skipped steps and estimates that were too high speed up the later steps, and the bar never goes
/// back. Emits at most one event per `granularity` of the bar or per `interval`, plus the last one, and none once
/// the current task is cancelled.
final class WorkMeter: Sendable {
    let stage: String
    private let emit: (@Sendable (ProgressEvent) -> Void)?
    private let granularity: Double
    private let interval: Duration
    private let state = Mutex<(position: Double, sent: Double, sentAt: ContinuousClock.Instant?)>((0, -1, nil))
    private let started = ContinuousClock.now

    init(stage: String, emit: (@Sendable (ProgressEvent) -> Void)?, granularity: Double = 0.0025,
         interval: Duration = .milliseconds(100)) {
        self.stage = stage
        self.emit = emit
        self.granularity = granularity
        self.interval = interval
    }

    /// Emits the start of the stage and returns its span, which `tail` expected seconds of the stage's own work
    /// follow.
    func begin(after tail: Double = 0) -> WorkSpan {
        advance(to: 0, force: true)
        return WorkSpan(meter: self, after: tail)
    }

    /// Fills the bar.
    func finish() { advance(to: 1, force: true) }

    var position: Double { state.withLock { $0.position } }

    func advance(to value: Double, force: Bool = false) {
        let now = ContinuousClock.now
        let event: ProgressEvent? = state.withLock { state in
            let position = max(state.position, min(1, value))
            state.position = position
            guard position > state.sent else { return nil }
            let due = force || position >= 1 || position - state.sent >= granularity
                || state.sentAt.map { now - $0 >= interval } ?? true
            guard due else { return nil }
            state.sent = position
            state.sentAt = now
            return ProgressEvent(stage: stage, completed: 0, total: 0, fraction: position)
        }
        if let event, !Task.isCancelled {
            ProgressTrace.log("meter \(stage) \((now - started) / .seconds(1)) \(event.fraction ?? 0)")
            emit?(event)
        }
    }
}

/// A step's place in its stage: the expected seconds of everything after it, up to the end of the stage.
struct WorkSpan: Sendable {
    let meter: WorkMeter?
    let after: Double

    static let none = WorkSpan(meter: nil, after: 0)

    func then(_ later: Double) -> WorkSpan { WorkSpan(meter: meter, after: after + later) }

    /// Starts a step expected to cost `cost` seconds: it covers the bar from where it is to
    /// start + (1 - start) * cost / (cost + after).
    func leaf(_ cost: Double) -> WorkLeaf {
        guard let meter else { return .none }
        let start = meter.position, cost = max(cost, 0)
        let end = cost + after > 0 ? start + (1 - start) * cost / (cost + after) : 1
        return WorkLeaf(meter: meter, start: start, end: end)
    }
}

/// The part of the bar one step covers.
struct WorkLeaf: Sendable {
    let meter: WorkMeter?
    let start: Double, end: Double

    static let none = WorkLeaf(meter: nil, start: 0, end: 0)

    /// The step's share done so far, in [0, 1].
    func report(_ fraction: Double) {
        meter?.advance(to: start + (end - start) * min(max(fraction, 0), 1))
    }

    func complete() { meter?.advance(to: end) }
}

/// The steps a function still expects, in order, with their expected seconds.
struct WorkPlan<Step: Hashable & Sendable>: Sendable {
    private let span: WorkSpan
    private var steps: [(step: Step, cost: Double)]

    init(_ span: WorkSpan, _ steps: [(Step, Double)]) {
        self.span = span
        self.steps = steps.map { (step: $0.0, cost: $0.1) }
    }

    /// The span of a piece of `step` that costs `cost` (all that is left of the step when nil). The steps before
    /// it are dropped, and the step keeps what is left of it, at least `floor`: the piece is followed by that and
    /// by every later step.
    mutating func next(_ step: Step, cost: Double? = nil, floor: Double = 0) -> WorkSpan {
        if let index = steps.firstIndex(where: { $0.step == step }) {
            steps.removeFirst(index)
            steps[0].cost = cost.map { max(steps[0].cost - $0, floor) } ?? 0
        }
        return span.then(steps.reduce(0) { $0 + $1.cost })
    }

    /// Sets the expected cost of `step`; 0 when it will not run.
    mutating func revise(_ step: Step, _ cost: Double) {
        if let index = steps.firstIndex(where: { $0.step == step }) { steps[index].cost = cost }
    }

    /// Scales the expected cost of the steps after `step`.
    mutating func scale(after step: Step, by factor: Double) {
        guard let index = steps.firstIndex(where: { $0.step == step }) else { return }
        for later in steps.indices where later > index { steps[later].cost *= factor }
    }
}

/// Expected seconds of the alignment's steps on an M5 MacBook Air, from the size of the problem. They only shape
/// the progress bar and never feed a result; the constants come from 291 timed solves on sets of 4 to 524 photos
/// (TESSERA_PROGRESS_TRACE=1). The linear fits were within 20% of their times; a homography's time varies with
/// its iterations (1 to 12 µs a match, the latter at the cap of 60), a rotation's ten times either way with how
/// soon its bundle adjustment converges.
enum AlignmentCost {
    struct Size: Sendable {
        /// Photos, pairs, matches, and matches the rotation uses.
        var photos: Int, pairs: Int, matches: Int, rotationMatches: Int

        /// The same problem with `photos` photos, pairs and matches scaled alike.
        func scaled(to photos: Int) -> Size {
            let r = Double(photos) / Double(max(1, self.photos))
            return Size(photos: photos, pairs: Int(Double(pairs) * r), matches: Int(Double(matches) * r),
                        rotationMatches: Int(Double(rotationMatches) * r))
        }
    }

    static func size(_ problem: AlignmentProblem) -> Size {
        let cap = Aligner.rotationCap(images: problem.images.count, pairs: problem.pairs.count)
        return Size(photos: problem.images.count, pairs: problem.pairs.count,
                    matches: problem.pairs.reduce(0) { $0 + $1.count },
                    rotationMatches: problem.pairs.reduce(0) {
                        $0 + min(($1.siftPoints.isEmpty ? $1.points.count : $1.siftPoints.count) / 4, cap)
                    })
    }

    /// Before AlignmentProblem.build: the verified pairs of the group, at most 300 matches each.
    static func size(_ report: MatchReport, component: [Int]) -> Size {
        let members = Set(component)
        var best: [PairProposal.Key: Int] = [:]
        for pair in report.pairs where pair.verdict == .verified && members.contains(pair.a) && members.contains(pair.b) {
            let key = PairProposal.Key(pair.a, pair.b)
            best[key] = max(best[key] ?? 0, min(300, pair.inlierCount))
        }
        let matches = best.values.reduce(0, +)
        let cap = Aligner.rotationCap(images: component.count, pairs: best.count)
        return Size(photos: component.count, pairs: best.count, matches: matches,
                    rotationMatches: best.values.reduce(0) { $0 + min($1, cap) })
    }

    static var build: Double { 7e-4 }

    static func solve(_ model: GlobalModel, _ s: Size) -> Double {
        let m = Double(s.matches)
        return switch model {
        case .translation: 0.23e-6 * m + 1e-3
        case .similarity: 0.42e-6 * m + 1e-3
        case .affine: 0.7e-6 * m + 1e-3
        case .homography: 8e-6 * m + 2e-3
        case .rotation: rotation(s)
        }
    }

    /// The dense J^T J product of the bundle adjustment, rows of matches times columns of photos squared, over
    /// a typical number of iterations.
    static func rotation(_ s: Size) -> Double {
        let n = Double(s.photos)
        return 1.5e-7 * n * n * Double(s.rotationMatches) + 1e-2
    }

    /// Homographies and, often, a second solve on a better anchor.
    static func homographies(_ s: Size) -> Double { 2 * solve(.homography, s) }

    static func planar(_ s: Size) -> Double { solve(.translation, s) + solve(.similarity, s) + solve(.affine, s) }

    static func choose(_ mode: StitchMode, _ s: Size, provisional: Bool) -> Double {
        switch mode {
        case .plane: return planar(s) + homographies(s)
        case .document: return homographies(s)
        case .rotation:
            return provisional && s.photos > Aligner.rotationTrial.limit ? 0 : rotation(s) + 0.2 * homographies(s)
        case .auto:
            guard s.photos > Aligner.rotationTrial.limit else { return planar(s) + homographies(s) + rotation(s) }
            // The rotation of the whole set is not expected: it runs only when the trial on 40 photos wins, and
            // then its own share of the bar, taken when it starts, is most of what is left.
            let trial = s.scaled(to: Aligner.rotationTrial.size)
            return planar(s) + homographies(s) + choose(.auto, trial, provisional: provisional)
                + 0.5 * (rotation(trial) + solve(.affine, trial))
        }
    }

    /// The steps of Aligner.align after the model is chosen, for `model`.
    static func tail(_ mode: StitchMode, model: GlobalModel, _ s: Size, provisional: Bool, automaticTrial: Bool)
        -> (trialHomographies: Double, trial: Double, leaveOut: Double, drop: Double, rechoose: Double, leaveOutAfter: Double) {
        let chosen = choose(mode, s, provisional: provisional)
        let leaves = (mode == .auto || automaticTrial) && model != .rotation && s.photos >= 40
        let trial = mode == .auto && model != .homography && model != .rotation
        return (trialHomographies: trial ? homographies(s) : 0,
                trial: trial ? align(.document, s, provisional: provisional, automaticTrial: true) + planar(s) : 0,
                leaveOut: leaves ? homographies(s) + 0.25 * chosen : 0,
                // Rounds of the drop loop: about nine on 511 drone photos, six on 162, none to two on six; with
                // a rotation, whose rounds each solve the bundle adjustment again, one at most as a rule.
                drop: (model == .rotation ? 1 : 1 + 0.8 * log2(Double(max(2, s.photos)))) * (solve(model, s) + 0.2 * chosen),
                rechoose: 0.5 * chosen,
                leaveOutAfter: leaves ? homographies(s) + 0.1 * chosen : 0)
    }

    /// The model a mode most likely ends with, before it is chosen.
    static func likelyModel(_ mode: StitchMode) -> GlobalModel {
        switch mode {
        case .plane, .auto: .affine
        case .document: .homography
        case .rotation: .rotation
        }
    }

    static func align(_ mode: StitchMode, _ s: Size, provisional: Bool, automaticTrial: Bool = false) -> Double {
        let rest = tail(mode, model: likelyModel(mode), s, provisional: provisional, automaticTrial: automaticTrial)
        return choose(mode, s, provisional: provisional) + rest.trialHomographies + rest.trial + rest.leaveOut + rest.drop
            + rest.rechoose + rest.leaveOutAfter
    }
}

/// With TESSERA_PROGRESS_TRACE=1, one line per timed step on standard error, to fit AlignmentCost's constants.
enum ProgressTrace {
    static let enabled = ProcessInfo.processInfo.environment["TESSERA_PROGRESS_TRACE"] == "1"

    static func log(_ line: @autoclosure () -> String) {
        guard enabled else { return }
        FileHandle.standardError.write(Data("trace \(line())\n".utf8))
    }
}
