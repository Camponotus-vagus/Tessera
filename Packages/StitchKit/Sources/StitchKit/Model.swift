import Foundation

/// Where a set of keypoints and matches comes from.
public enum FeatureSource: String, Sendable, Codable, CaseIterable {
    case rootSIFT
    case racoLightGlue
}

/// How the photos were taken; decides which motion models are tried.
public enum StitchMode: String, Sendable, Codable, CaseIterable {
    /// Try every model and keep the simplest one that explains the matches.
    case auto
    /// Camera rotating about a point (landscape panoramas).
    case rotation
    /// Tiles of a plane moved under the camera (microscope, trays, traps).
    case plane
    /// A flat original photographed in pieces from different viewpoints.
    case document

    public var candidateModels: [MotionModel] {
        switch self {
        case .auto: [.translation, .similarity, .affine, .homography]
        case .rotation: [.homography]
        case .plane: [.translation, .similarity, .affine]
        case .document: [.affine, .homography]
        }
    }
}

public enum MotionModel: String, Sendable, Codable, CaseIterable, Comparable {
    case translation
    case similarity
    case affine
    case homography

    public var degreesOfFreedom: Int {
        switch self {
        case .translation: 2
        case .similarity: 4
        case .affine: 6
        case .homography: 8
        }
    }

    public var minimumSample: Int {
        switch self {
        case .translation: 1
        case .similarity: 2
        case .affine: 3
        case .homography: 4
        }
    }

    public static func < (lhs: MotionModel, rhs: MotionModel) -> Bool {
        lhs.degreesOfFreedom < rhs.degreesOfFreedom
    }
}

public struct PixelSize: Sendable, Codable, Hashable {
    public var width: Int
    public var height: Int

    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
    }
}

public struct Point2: Sendable, Codable, Hashable {
    public var x: Float
    public var y: Float

    public init(x: Float, y: Float) {
        self.x = x
        self.y = y
    }
}

public struct Keypoint: Sendable, Codable, Hashable {
    /// Position in original image pixels (after EXIF orientation).
    public var x: Float
    public var y: Float
    /// Diameter in original pixels, 0 when the detector has no scale.
    public var size: Float
    public var angle: Float
    public var response: Float
}

/// Input photo as seen by the engine.
public struct SourceImage: Sendable, Codable, Hashable, Identifiable {
    public var id: Int
    public var url: URL
    public var name: String
    /// Oriented size of the full-resolution image.
    public var pixelSize: PixelSize
    public var captureDate: Date?
    public var focalLength35mm: Double?

    public init(id: Int, url: URL, name: String, pixelSize: PixelSize, captureDate: Date?, focalLength35mm: Double?) {
        self.id = id
        self.url = url
        self.name = name
        self.pixelSize = pixelSize
        self.captureDate = captureDate
        self.focalLength35mm = focalLength35mm
    }
}

public struct ImageFeatures: Sendable, Codable {
    public var imageID: Int
    public var source: FeatureSource
    /// Size of the image the detector actually ran on.
    public var workingSize: PixelSize
    public var keypoints: [Keypoint]
    public var extractionSeconds: Double
}

public struct TentativeMatch: Sendable, Codable, Hashable {
    public var a: Point2
    public var b: Point2
    /// Matcher confidence in [0, 1].
    public var score: Float
}

public struct ModelFit: Sendable, Codable {
    public var model: MotionModel
    /// Row-major 3x3 matrix mapping image A onto image B, in original pixels, with the sign that
    /// gives z > 0 at the inliers.
    public var transform: [Double]
    /// Indices into `PairEvidence.matches`.
    public var inliers: [Int32]
    public var rmse: Double
    public var medianError: Double
}

public enum PairVerdict: String, Sendable, Codable {
    case verified
    case tooFewMatches
    case noConsistentModel
    /// The transform folds the image, or scales/shears a planar subject implausibly.
    case implausibleModel
    /// Inliers cover a thin strip or a spot (repeated labels, text lines).
    case degenerateInliers
    /// Not distinguishable from chance (a-contrario test).
    case lowConfidence
    /// A homography explains far more matches than the models the mode allows (plane mode).
    case modelMismatch
}

/// Everything known about one image pair from one matcher.
public struct PairEvidence: Sendable, Codable, Identifiable {
    public var a: Int
    public var b: Int
    public var source: FeatureSource
    public var matches: [TentativeMatch]
    public var fits: [ModelFit]
    public var chosenModel: MotionModel?
    /// Region of A that falls inside B under the chosen model (empty without a model).
    public var overlapA: [Point2] = []
    /// The same region seen in B.
    public var overlapB: [Point2] = []
    /// Tentative matches whose point in A lies in the overlap.
    public var matchesInOverlap = 0
    /// Area of the inliers' convex hull, without its outermost layer, divided by the overlap area (0...1).
    public var inlierCoverage: Double
    /// Brown-Lowe confidence n_inliers / (8 + 0.3 * n_overlap), with n_overlap = matchesInOverlap.
    public var confidence: Double
    /// log10 of the a-contrario number of false alarms; the pair is significant below 0.
    public var log10NFA: Double = .infinity
    public var verdict: PairVerdict
    public var matchingSeconds: Double
    public var verificationSeconds: Double

    public var id: String { "\(a)-\(b)-\(source.rawValue)" }

    public var chosenFit: ModelFit? {
        fits.first { $0.model == chosenModel }
    }

    public var inlierCount: Int { chosenFit?.inliers.count ?? 0 }
}

public enum ExclusionReason: String, Sendable, Codable {
    /// The file could not be opened or decoded, or is truncated.
    case unreadable
    case tooFewFeatures
    case noVerifiedPair
    case weakLinkOnly
    case separateGroup
    case excludedByUser
}

public struct GraphNode: Sendable, Codable, Identifiable {
    public var id: Int
    public var featureCount: Int
    public var component: Int
    public var exclusion: ExclusionReason?
}

public struct GraphEdge: Sendable, Codable, Hashable {
    public var a: Int
    public var b: Int
    public var source: FeatureSource
    public var matches: Int
    public var inliers: Int
    public var confidence: Double
    public var accepted: Bool
}

public struct MatchGraph: Sendable, Codable {
    public var nodes: [GraphNode]
    public var edges: [GraphEdge]
    /// Image ids per connected component, largest first.
    public var components: [[Int]]
}

public struct PipelineConfiguration: Sendable, Codable {
    public var mode: StitchMode = .auto
    public var sources: [FeatureSource] = [.rootSIFT]
    /// Megapixels of the image SIFT runs on.
    public var siftMegapixels: Double = 1.5
    public var siftMaxFeatures: Int = 6000
    public var ratio: Float = 0.8
    public var mutualCheck = true
    /// Split RaCo-ALIKED + LightGlue models; nil disables the learned matcher.
    public var learnedModels: LearnedModelSet? = LearnedModelSet.standard()
    /// Dense half on Core ML plus sparse half on the CPU, or the whole extractor on ONNX Runtime.
    public var extractorBackend: LearnedBackend = .coreMLGPU
    /// "fp32" gives the same keypoints as the ONNX extractor; "fp16" is needed for the Neural Engine.
    public var extractorPrecision = "fp32"
    /// Core ML computes only ALIKED's low-resolution feature levels and the descriptor head runs in C++
    /// on the pixels it needs; false keeps the full-resolution feature map and the ONNX head.
    public var fastDescriptorHead = true
    /// With the fast descriptor head, true selects the keypoints in C++ instead of with the ONNX select
    /// model; nil or false keeps ONNX Runtime. Optional so that earlier reports still decode.
    public var nativeKeypointSelection: Bool?
    public var matcherBackend: LearnedBackend = .coreMLGPU
    /// "fp16" (fastest, GPU or Neural Engine) or "fp32" (same matches as the ONNX matcher).
    public var matcherPrecision = "fp16"
    /// LightGlue match confidence below which a pair of keypoints is dropped.
    public var matchThreshold: Float = 0.1
    public var lightGlueLowMemory = false
    /// Which pairs both matchers examine.
    public var pairSelection: PairSelection = .proposed
    /// Best partners per photo, by descriptor affinity, among the candidate pairs.
    public var proposalNeighbours = 2
    /// Extra pairs tried between every two disconnected groups, per round (up to three rounds).
    public var bridgePairs = 2
    /// Inlier threshold in pixels of a 1 MP image; scaled to the real resolution.
    public var inlierThreshold: Double = 3.0
    public var maxIterations = 5000
    public var ransacConfidence = 0.999
    public var seed: Int32 = 1
    public var minimumFeatures = 50
    public var minimumInliers = 15
    /// Inlier hull over the overlap area.
    public var minimumCoverage = 0.05
    /// Inlier hull over the area of image A.
    public var minimumHullFraction = 0.005

    public init() {}
}

public enum PairSelection: String, Sendable, Codable, CaseIterable {
    case all
    /// Consecutive shots, best partners by descriptor affinity, and a spanning tree of the affinity.
    case proposed
}

public enum CandidateReason: String, Sendable, Codable {
    /// Every pair was matched (few photos, or the user asked for all pairs).
    case all
    case consecutive
    case neighbour
    case spanningTree
    /// Tried after verification to join groups the first candidates left apart.
    case bridge
}

/// A pair the matchers examined, and why it was chosen.
public struct CandidatePair: Sendable, Codable, Hashable {
    public var a: Int
    public var b: Int
    /// Mutual nearest neighbours among the best descriptors of the two photos.
    public var affinity: Int?
    public var reasons: [CandidateReason]
}

/// Versioned, self-contained result of a diagnostic run.
public struct MatchReport: Sendable, Codable {
    public static let currentVersion = 2

    public var version: Int = MatchReport.currentVersion
    public var createdAt: Date
    public var configuration: PipelineConfiguration
    public var images: [SourceImage]
    public var features: [ImageFeatures]
    public var pairs: [PairEvidence]
    public var graph: MatchGraph
    public var excludedByUser: [Int]
    /// Pairs that were matched (version 2 and later).
    public var candidates: [CandidatePair]?
    /// Which learned extractor and matcher actually ran, when the learned matcher was used.
    public var learnedPipeline: String?

    public func evidence(_ a: Int, _ b: Int, source: FeatureSource) -> PairEvidence? {
        pairs.first { $0.source == source && (($0.a == a && $0.b == b) || ($0.a == b && $0.b == a)) }
    }
}
