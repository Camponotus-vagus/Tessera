import CStitchCore
import Foundation
import Testing

@testable import StitchKit

@Suite("Alignment progress and Stop")
struct AlignmentProgressTests {
    private static func multiply(_ a: [Double], _ b: [Double]) -> [Double] {
        (0..<9).map { k in (0..<3).reduce(0) { $0 + a[3 * (k / 3) + $1] * b[3 * $1 + k % 3] } }
    }

    private static func apply(_ m: [Double], _ x: Double, _ y: Double) -> (Double, Double) {
        let w = m[6] * x + m[7] * y + m[8]
        return ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)
    }

    private static func photos(_ count: Int, width: Int = 1200, height: Int = 900, focal35mm: Double? = nil) -> [SourceImage] {
        (0..<count).map {
            SourceImage(id: $0, url: URL(fileURLWithPath: "/dev/null"), name: "photo\($0)",
                        pixelSize: PixelSize(width: width, height: height), captureDate: nil, focalLength35mm: focal35mm)
        }
    }

    /// `count` matches inside both 1200 x 900 photos through `h` (a to b), with up to `noise` px of noise.
    private static func pair(_ a: Int, _ b: Int, _ h: [Double], count: Int, noise: Double,
                             rng: inout SplitMix64) -> AlignmentProblem.Pair {
        var points: [Float] = []
        while points.count < 4 * count {
            let x = Double.random(in: 0..<1200, using: &rng), y = Double.random(in: 0..<900, using: &rng)
            let (u, v) = apply(h, x, y)
            guard u >= 0, v >= 0, u < 1200, v < 900 else { continue }
            points += [Float(x + Double.random(in: -noise...noise, using: &rng)), Float(y + Double.random(in: -noise...noise, using: &rng)),
                       Float(u + Double.random(in: -noise...noise, using: &rng)), Float(v + Double.random(in: -noise...noise, using: &rng))]
        }
        return AlignmentProblem.Pair(a: a, b: b, points: points, sigmas: Array(repeating: 1, count: count), homography: h,
                                     siftPoints: [], siftSigmas: [])
    }

    /// A camera turning by 15 degrees a shot, `count` shots, each paired with the next and the one after it. With
    /// `exifFactor` the photos carry a 35 mm focal length that many times the true one.
    static func ring(count: Int, exifFactor: Double? = nil, seed: UInt64 = 5) -> AlignmentProblem {
        let focal = 1400.0
        let toRay: [Double] = [1 / focal, 0, -600 / focal, 0, 1 / focal, -450 / focal, 0, 0, 1]
        func turn(_ degrees: Double) -> [Double] {
            let r = degrees * .pi / 180
            return [cos(r), 0, sin(r), 0, 1, 0, -sin(r), 0, cos(r)]
        }
        func truth(_ a: Int, _ b: Int) -> [Double] {
            multiply(AlignmentProblem.inverse(toRay), multiply(turn(-15 * Double(b)), multiply(turn(15 * Double(a)), toRay)))
        }
        var rng = SplitMix64(state: seed)
        var pairs: [AlignmentProblem.Pair] = []
        for i in 0..<count {
            for step in 1...2 where i + step < count {
                pairs.append(pair(i, i + step, truth(i, i + step), count: 150, noise: 0.4, rng: &rng))
            }
        }
        // 35 mm equivalent by the diagonal: f35 = f * 43.27 / diagonal.
        let focal35 = exifFactor.map { $0 * focal * 43.2666 / 1500 }
        return AlignmentProblem(images: photos(count, focal35mm: focal35), pairs: pairs,
                                thresholds: Array(repeating: 3, count: count))
    }

    /// A ring of `count` shots whose first pair has `moved` of its 150 matches 6 px off, as if the subject moved:
    /// the bundle adjustment then runs its refit without them.
    static func movingRing(count: Int, moved: Int = 120) -> AlignmentProblem {
        var problem = ring(count: count)
        var points = problem.pairs[0].points
        for k in 0..<moved { points[4 * k + 2] += 6 }
        problem.pairs[0].points = points
        return problem
    }

    /// `columns` x `rows` tilted tiles joined by their sides and one diagonal per square, the last one shrunk to
    /// 0.15 of its tile and placed by its two pairs: the homographies stretch it beyond belief.
    static func misplacedGrid(columns: Int, rows: Int, seed: UInt64 = 17) -> AlignmentProblem {
        var rng = SplitMix64(state: seed)
        var truth: [[Double]] = (0..<(columns * rows)).map { i in
            let x = Double(i % columns) * 800, y = Double(i / columns) * 600
            let px = Double.random(in: -6e-5...6e-5, using: &rng), py = Double.random(in: -6e-5...6e-5, using: &rng)
            return multiply([1, 0, x + 600, 0, 1, y + 450, 0, 0, 1], multiply([1, 0, 0, 0, 1, 0, px, py, 1], [1, 0, -600, 0, 1, -450, 0, 0, 1]))
        }
        let last = truth.count - 1
        truth[last] = multiply(truth[last], [0.15, 0, 100, 0, 0.15, 100, 0, 0, 1])
        var pairs: [AlignmentProblem.Pair] = []
        for i in 0..<(columns * rows) {
            let c = i % columns, r = i / columns
            for j in [c + 1 < columns ? i + 1 : nil, r + 1 < rows ? i + columns : nil,
                      c + 1 < columns && r + 1 < rows && i + columns + 1 != last ? i + columns + 1 : nil].compactMap({ $0 }) {
                pairs.append(pair(i, j, multiply(AlignmentProblem.inverse(truth[j]), truth[i]), count: 60, noise: 0.3, rng: &rng))
            }
        }
        return AlignmentProblem(images: photos(columns * rows), pairs: pairs, thresholds: Array(repeating: 3, count: columns * rows))
    }

    /// 8 x 5 tilted tiles joined by their sides and one diagonal per square.
    static func grid(seed: UInt64 = 17) -> AlignmentProblem {
        var rng = SplitMix64(state: seed)
        let columns = 8, rows = 5
        let truth: [[Double]] = (0..<(columns * rows)).map { i in
            let x = Double(i % columns) * 800, y = Double(i / columns) * 600
            let px = Double.random(in: -6e-5...6e-5, using: &rng), py = Double.random(in: -6e-5...6e-5, using: &rng)
            return multiply([1, 0, x + 600, 0, 1, y + 450, 0, 0, 1], multiply([1, 0, 0, 0, 1, 0, px, py, 1], [1, 0, -600, 0, 1, -450, 0, 0, 1]))
        }
        var pairs: [AlignmentProblem.Pair] = []
        for i in 0..<(columns * rows) {
            let c = i % columns, r = i / columns
            for j in [c + 1 < columns ? i + 1 : nil, r + 1 < rows ? i + columns : nil,
                      c + 1 < columns && r + 1 < rows ? i + columns + 1 : nil].compactMap({ $0 }) {
                pairs.append(pair(i, j, multiply(AlignmentProblem.inverse(truth[j]), truth[i]), count: 60, noise: 0.3, rng: &rng))
            }
        }
        return AlignmentProblem(images: photos(columns * rows), pairs: pairs, thresholds: Array(repeating: 3, count: columns * rows))
    }

    private static func bits(_ alignment: Alignment) -> [UInt64] {
        (alignment.transforms.flatMap { $0 } + alignment.focals + alignment.pairRMS + [alignment.rms]).map(\.bitPattern)
            + [UInt64(alignment.method)]
    }

    /// Largest difference between two alignments' numbers, relative to their size.
    private static func difference(_ a: Alignment, _ b: Alignment) -> Double {
        let x = a.transforms.flatMap { $0 } + a.pairRMS + [a.rms], y = b.transforms.flatMap { $0 } + b.pairRMS + [b.rms]
        guard x.count == y.count else { return .infinity }
        return zip(x, y).map { abs($0 - $1) / max(1, abs($0)) }.max() ?? 0
    }

    @Test("Monitored and unmonitored solves give the same results")
    func identical() throws {
        let grid = Self.grid()
        for model in [GlobalModel.translation, .similarity, .affine, .homography] {
            let plain = try Aligner.solve(model, grid, anchor: 19, monitored: false)
            let watched = try Aligner.solve(model, grid, anchor: 19)
            #expect(Self.bits(plain) == Self.bits(watched), "\(model): \(Self.difference(plain, watched))")
        }
        // The ray adjustment with wave correction, with its refit, then the fixed-focal fallback: bit for bit.
        for (problem, method) in [(Self.ring(count: 24), 0), (Self.movingRing(count: 12), 0),
                                  (Self.ring(count: 16, exifFactor: 1.6), 1)] {
            let plain = try Aligner.solve(.rotation, problem, anchor: 0, monitored: false)
            let watched = try Aligner.solve(.rotation, problem, anchor: 0)
            #expect(plain.method == Int32(method))
            #expect(Self.bits(plain) == Self.bits(watched), "rotation, method \(method)")
        }
    }

    @Test("Planar solves repeated, also on several threads at once, give the same bits")
    func repeatable() async throws {
        // Accelerate's sparse Cholesky summed in a different order from run to run: 6 solves of the grid gave
        // up to 6 different affine and homography results.
        let grid = Self.grid()
        for model in [GlobalModel.translation, .similarity, .affine, .homography] {
            let first = Self.bits(try Aligner.solve(model, grid, anchor: 19, monitored: false))
            let others = try await withThrowingTaskGroup(of: [UInt64].self) { group in
                for _ in 0..<6 { group.addTask { Self.bits(try Aligner.solve(model, grid, anchor: 19, monitored: false)) } }
                return try await group.reduce(into: [[UInt64]]()) { $0.append($1) }
            }
            #expect(others.allSatisfy { $0 == first }, "\(model)")
        }
    }

    /// Counts the calls of the progress callback and asks for a stop at call `stopAt`.
    private final class Counter {
        var calls = 0
        var fractions: [Double] = []
        let stopAt: Int
        init(stopAt: Int) { self.stopAt = stopAt }
    }

    /// Calls sc_align as Aligner.solve does, with a callback that stops it at call `stopAt`.
    private static func rawAlign(_ model: sc_align_model, _ problem: AlignmentProblem, stopAt: Int)
        -> (status: Int32, error: String, calls: Int, fractions: [Double], untouched: Bool, result: sc_align_result) {
        let n = problem.images.count
        // The focal length prior from the 35 mm equivalent, as Aligner.solve gives it.
        let images = problem.images.map { image in
            let diagonal = Double(image.pixelSize.width * image.pixelSize.width + image.pixelSize.height * image.pixelSize.height).squareRoot()
            return sc_align_image(width: Int32(image.pixelSize.width), height: Int32(image.pixelSize.height),
                                  focal: image.focalLength35mm.map { $0 * diagonal / 43.2666 } ?? 0)
        }
        var transforms = [Double](repeating: .nan, count: 9 * n)
        var focals = [Double](repeating: .nan, count: n)
        var pairRMS = [Double](repeating: .nan, count: problem.pairs.count)
        var result = sc_align_result(ok: 7, rms: 7, iterations: 7)
        var message = [CChar](repeating: 0, count: 256)
        let counter = Counter(stopAt: stopAt)
        let control = sc_progress(report: { context, fraction in
            let counter = Unmanaged<Counter>.fromOpaque(context!).takeUnretainedValue()
            counter.calls += 1
            counter.fractions.append(fraction)
            return counter.calls >= counter.stopAt ? 1 : 0
        }, context: Unmanaged.passUnretained(counter).toOpaque())
        let flat = problem.pairs.map(\.points)
        let status = withExtendedLifetime(counter) {
            withUnsafePointer(to: control) { control in
                var pairs: [sc_align_pair] = []
                var buffers: [UnsafeMutableBufferPointer<Float>] = []
                for (pair, points) in zip(problem.pairs, flat) {
                    let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: points.count)
                    _ = buffer.initialize(from: points)
                    buffers.append(buffer)
                    let h = pair.homography
                    pairs.append(sc_align_pair(a: Int32(pair.a), b: Int32(pair.b), count: Int32(points.count / 4),
                                               points: UnsafePointer(buffer.baseAddress!), sigma: nil,
                                               homography: (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8])))
                }
                defer { buffers.forEach { $0.deallocate() } }
                return sc_align(model, images, Int32(n), pairs, Int32(pairs.count), 0, -1, &transforms, &focals,
                                &pairRMS, &result, control, &message, message.count)
            }
        }
        let untouched = (transforms + focals + pairRMS).allSatisfy(\.isNaN)
        return (status, String(cString: message), counter.calls, counter.fractions, untouched, result)
    }

    @Test("A stopped sc_align returns 2, writes nothing and makes no further call")
    func stoppedNative() {
        let grid = Self.grid(), ring = Self.ring(count: 24)
        // The rotation also with its refit (a moving subject) and with the fixed-focal fallback (EXIF 1.6 times off).
        for (model, problem) in [(SC_ALIGN_HOMOGRAPHY, grid), (SC_ALIGN_AFFINE, grid), (SC_ALIGN_ROTATION, ring),
                                 (SC_ALIGN_ROTATION, Self.movingRing(count: 12)),
                                 (SC_ALIGN_ROTATION, Self.ring(count: 16, exifFactor: 1.6))] {
            // The calls of a whole run, then stops at the first, the second, a few and the last.
            let whole = Self.rawAlign(model, problem, stopAt: .max)
            #expect(whole.status == 0 && whole.calls >= 3, "\(model): \(whole.status) after \(whole.calls) calls")
            for k in Set([1, 2, min(5, whole.calls), whole.calls / 4, whole.calls / 2, 3 * whole.calls / 4, whole.calls])
                .filter({ $0 >= 1 }).sorted() {
                let run = Self.rawAlign(model, problem, stopAt: k)
                #expect(run.status == 2 && run.error == "cancelled", "\(model) at \(k): \(run.status) \(run.error)")
                #expect(run.calls == k, "\(model) at \(k): \(run.calls) calls")
                #expect(run.untouched && run.result.ok == 0 && run.result.rms == 0, "\(model) at \(k)")
                #expect(zip(run.fractions, run.fractions.dropFirst()).allSatisfy { $0 <= $1 }
                        && run.fractions.allSatisfy { $0 >= 0 && $0 <= 1 }, "\(model) at \(k): \(run.fractions)")
            }
        }
    }

    /// Events seen by a test's meter; touched only from the task that runs the alignment.
    final class Events: @unchecked Sendable {
        var count = 0
        var fractions: [Double] = []
        var cancelledAt: ContinuousClock.Instant?
    }

    /// A meter that emits every advance and cancels the current task at event `cancelAt`.
    static func meter(_ events: Events, cancelAt: Int = .max, stage: String = "align") -> WorkMeter {
        WorkMeter(stage: stage, emit: { event in
            events.count += 1
            events.fractions.append(event.fraction ?? -1)
            if events.count == cancelAt {
                events.cancelledAt = .now
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }, granularity: 0, interval: .zero)
    }

    /// The moment the test asked for a stop, read by the task when its solve throws.
    final class StopClock: @unchecked Sendable {
        private let lock = NSLock()
        private var asked: ContinuousClock.Instant?
        var askedAt: ContinuousClock.Instant? {
            get { lock.lock(); defer { lock.unlock() }; return asked }
            set { lock.lock(); asked = newValue; lock.unlock() }
        }
    }

    /// The task a test stops from another thread.
    final class TaskBox: @unchecked Sendable {
        private let lock = NSLock()
        private var held: Task<(Result<Alignment, Error>, Duration?), Never>?
        var task: Task<(Result<Alignment, Error>, Duration?), Never>? {
            get { lock.lock(); defer { lock.unlock() }; return held }
            set { lock.lock(); held = newValue; lock.unlock() }
        }
    }

    @Test("Stop from outside a bundle adjustment takes effect at once and leaves nothing behind", arguments: [10, 40])
    func cancelledAdjustment(after events: Int) async throws {
        let problem = Self.ring(count: 24)
        let clock = StopClock(), box = TaskBox(), seen = Events()
        // At event `events`, a thread of its own cancels the task 20 ms later, at no particular poll, as Stop does;
        // the task measures from that moment to the throw.
        let task = Task { () -> (Result<Alignment, Error>, Duration?) in
            let meter = WorkMeter(stage: "align", emit: { _ in
                seen.count += 1
                guard seen.count == events else { return }
                Thread.detachNewThread {
                    usleep(20_000)
                    clock.askedAt = .now
                    box.task?.cancel()
                }
            }, granularity: 0, interval: .zero)
            let result = Result { try Aligner.solve(.rotation, problem, anchor: 0, progress: meter.begin()) }
            return (result, clock.askedAt.map { ContinuousClock.now - $0 })
        }
        box.task = task
        let (result, latency) = await task.value
        switch result {
        case .success: Issue.record("the solve finished although it was stopped after \(events) events")
        case .failure(let error): #expect(error is CancellationError, "\(error)")
        }
        let measured = try #require(latency)
        #expect(measured < .milliseconds(500), "\(measured)")
        // Nothing is left over: a later solve equals a fresh one.
        let after = try Aligner.solve(.rotation, problem, anchor: 0)
        let fresh = try Aligner.solve(.rotation, Self.ring(count: 24), anchor: 0)
        #expect(Self.bits(after) == Self.bits(fresh))
    }

    /// Runs `body` once to count its solves, then once per solve, cancelled as that solve starts: every result
    /// must be CancellationError, never a layout and never another error, whichever `try?` the solve sits behind.
    private func stopsAtEverySolve(_ name: String, minimum: Int, _ body: @escaping @Sendable () throws -> Void) async throws {
        final class Counter: @unchecked Sendable { var solves = 0 }
        let counter = Counter()
        try Aligner.$solveObserver.withValue({ _ in counter.solves += 1 }) { try body() }
        try #require(counter.solves >= minimum, "\(name): \(counter.solves) solves")
        for k in 1...counter.solves {
            let outcome = await Task { () -> Result<Void, Error> in
                let seen = Counter()
                return Result {
                    try Aligner.$solveObserver.withValue({ _ in
                        seen.solves += 1
                        if seen.solves == k { withUnsafeCurrentTask { $0?.cancel() } }
                    }) { try body() }
                }
            }.value
            switch outcome {
            case .success: Issue.record("\(name): stopped at solve \(k) of \(counter.solves), the alignment returned a layout")
            case .failure(let error): #expect(error is CancellationError, "\(name), solve \(k): \(error)")
            }
        }
    }

    @Test("A cancelled alignment never returns a layout, whichever solve the Stop lands on")
    func cancelledAlignment() async throws {
        // Automatic mode's leave-out and drop loop, as its Document trial runs them: no bundle adjustment.
        let large = Self.misplacedGrid(columns: 8, rows: 5)
        try await stopsAtEverySolve("leave-out", minimum: 8) {
            let result = try Aligner.align(large, mode: .document, centre: 19, straighten: true, automaticTrial: true)
            #expect(!result.misplaced.isEmpty)
        }
        // The drop loop: a false pair between two far tiles, 500 px off, is left out and the model chosen again.
        var withFalsePair = Self.grid()
        var rng = SplitMix64(state: 3)
        withFalsePair.pairs.append(Self.pair(0, 39, [1, 0, -500, 0, 1, 40, 0, 0, 1], count: 60, noise: 0.3, rng: &rng))
        let falsePair = withFalsePair
        try await stopsAtEverySolve("drop loop", minimum: 4) {
            let result = try Aligner.align(falsePair, mode: .document, centre: 19, straighten: true)
            #expect(result.dropped.contains { $0.a == 0 && $0.b == 39 })
        }
        // Automatic mode with its rotation, its Document trial and their planar solves, on a small set.
        let small = Self.misplacedGrid(columns: 4, rows: 3)
        try await stopsAtEverySolve("automatic", minimum: 10) {
            _ = try Aligner.align(small, mode: .auto, centre: 5, straighten: true)
        }
    }

    @Test("Alignment progress rises to the end without going back")
    func risingProgress() throws {
        for (name, problem, mode, centre) in [("grid auto", Self.grid(), StitchMode.auto, 19), ("grid plane", Self.grid(), .plane, 19),
                                              ("grid document", Self.grid(), .document, 19),
                                              ("ring auto", Self.ring(count: 24), .auto, 0)] {
            let events = Events()
            let meter = Self.meter(events)
            _ = try Aligner.align(problem, mode: mode, centre: centre, straighten: true, progress: meter.begin())
            meter.finish()
            #expect(events.count >= 10, "\(name): \(events.count) events")
            #expect(events.fractions.allSatisfy { $0 >= 0 && $0 <= 1 }, "\(name)")
            #expect(zip(events.fractions, events.fractions.dropFirst()).allSatisfy { $0 < $1 }, "\(name): \(events.fractions)")
            #expect(events.fractions.last == 1, "\(name)")
        }
    }

    @Test("Work spans share out what is left of the bar")
    func workSpans() {
        let events = Events()
        let meter = Self.meter(events)
        let span = meter.begin(after: 3)
        // A step of 1 second followed by 3: a quarter of the bar.
        let first = span.leaf(1)
        #expect(first.start == 0 && first.end == 0.25)
        first.report(0.5)
        #expect(meter.position == 0.125)
        first.complete()
        // A plan of three steps; the first is skipped, so the second takes more of what is left.
        var plan = WorkPlan<Int>(WorkSpan(meter: meter, after: 0), [(0, 1), (1, 1), (2, 2)])
        let second = plan.next(1).leaf(1)
        #expect(abs(second.end - (0.25 + 0.75 * 1 / 3)) < 1e-12, "\(second.end)")
        second.complete()
        // A loop's pieces keep a floor in reserve, so the bar does not fill before the loop ends.
        plan.revise(2, 4)
        for _ in 0..<5 {
            plan.next(2, cost: 1, floor: 1).leaf(1).complete()
        }
        #expect(meter.position < 1, "\(meter.position)")
        // Advancing never goes back, and the bar ends at 1.
        meter.advance(to: 0.1)
        #expect(meter.position > 0.5)
        meter.finish()
        #expect(meter.position == 1 && events.fractions.last == 1)
        #expect(zip(events.fractions, events.fractions.dropFirst()).allSatisfy { $0 < $1 })
        // No meter, no work: the none span and leaf do nothing.
        WorkSpan.none.leaf(5).report(0.5)
        WorkLeaf.none.complete()
    }

    @Test("A meter emits no event once its task is cancelled")
    func silentAfterCancel() async {
        let events = Events()
        await Task {
            let meter = Self.meter(events)
            let leaf = meter.begin().leaf(1)
            leaf.report(0.3)
            withUnsafeCurrentTask { $0?.cancel() }
            leaf.report(0.6)
            meter.finish()
        }.value
        #expect(events.fractions == [0, 0.3], "\(events.fractions)")
    }

    /// 4 x 3 tiles of a synthetic scene, 600 x 450 px, 480 px across and 360 px down.
    private static func tiles(in directory: URL) throws -> [URL] {
        let scene = Synthetic.scene(width: 2100, height: 1200, seed: 61)
        var urls: [URL] = []
        for row in 0..<3 {
            for column in 0..<4 {
                let url = directory.appendingPathComponent(String(format: "tile_r%d_c%d.png", row, column))
                try Synthetic.write(Synthetic.tile(of: scene, origin: CGPoint(x: column * 480, y: row * 360),
                                                   size: CGSize(width: 600, height: 450)), to: url)
                urls.append(url)
            }
        }
        return urls
    }

    /// Engine events, recorded from any thread. Once `stage` is a third done, it cancels the task the test set in
    /// `stop`, as Stop cancels the app's task: by then a solve is running.
    final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [ProgressEvent] = []
        private var stopped = false
        var cancelStage: String?
        var stop: (@Sendable () -> Void)?

        func record(_ event: ProgressEvent) {
            lock.lock()
            events.append(event)
            let cancel = !stopped && event.stage == cancelStage && (event.fraction ?? 0) >= 1.0 / 3
            if cancel { stopped = true }
            let stop = self.stop
            lock.unlock()
            if cancel { stop?() }
        }

        var all: [ProgressEvent] {
            lock.lock()
            defer { lock.unlock() }
            return events
        }
    }

    @Test("An analysis cancelled while laying out the photos stops; uncancelled, its counted stages rise to the end")
    func cancelledLayout() async throws {
        let directory = try Synthetic.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let urls = try Self.tiles(in: directory)
        var configuration = PipelineConfiguration()
        configuration.sources = [.rootSIFT]
        configuration.proposalNeighbours = 1
        let engine = StitchEngine()

        let recorder = Recorder()
        let report = try await engine.analyze(urls: urls, configuration: configuration) { recorder.record($0) }
        let events = recorder.all
        #expect(events.contains { $0.stage == "layout" && $0.fraction == 1 })
        for stage in ["affinity", "verify"] {
            let counted = events.filter { $0.stage == stage && $0.total > 0 }
            #expect(!counted.isEmpty, "no \(stage) events")
            // Each round of verification counts from 0 again; within a round the count never goes back.
            var last: ProgressEvent?
            for event in counted {
                if let previous = last, previous.total == event.total, previous.completed < previous.total {
                    #expect(event.completed >= previous.completed, "\(stage): \(previous.completed) then \(event.completed)")
                }
                last = event
            }
            #expect(counted.contains { $0.completed == $0.total }, "\(stage) never reaches its total")
        }

        let stopping = Recorder()
        stopping.cancelStage = "layout"
        let analysis = Task { () -> Result<MatchReport, Error> in
            await Result { try await engine.analyze(urls: urls, configuration: configuration) { stopping.record($0) } }
        }
        stopping.stop = { analysis.cancel() }
        let result = await analysis.value
        switch result {
        case .success: Issue.record("an analysis cancelled while laying out the photos returned a report")
        case .failure(let error): #expect(error is CancellationError, "\(error)")
        }

        // A stitch cancelled a third of the way through its alignment stops without an error either.
        let stitching = Recorder()
        stitching.cancelStage = "align"
        let stitch = Task { () -> Result<Panorama, Error> in
            await Result { try await engine.stitch(report) { stitching.record($0) } }
        }
        stitching.stop = { stitch.cancel() }
        let stitched = await stitch.value
        switch stitched {
        case .success: Issue.record("a stitch cancelled while aligning returned a panorama")
        case .failure(let error): #expect(error is CancellationError, "\(error)")
        }
    }
}

extension Result where Failure == Error {
    /// Captures the outcome of an async throwing body.
    init(_ body: () async throws -> Success) async {
        do { self = .success(try await body()) } catch { self = .failure(error) }
    }
}
