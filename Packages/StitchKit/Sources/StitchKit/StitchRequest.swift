import CoreGraphics
import Foundation

/// How the panorama is projected.
public enum Projection: String, Sendable, Codable, CaseIterable {
    /// Chosen from the alignment: the photos' plane for tiles and documents, and for a rotating camera a
    /// projection that fits its field of view.
    case automatic
    /// The reference photo's plane: tiles, trays, documents.
    case flat
    case rectilinear
    case cylindrical
    /// Equirectangular, for very wide or tall sweeps.
    case spherical
}

/// Output resolution relative to the photos.
public enum OutputSize: Sendable, Codable, Hashable {
    case full
    /// Fraction of full resolution, in (0, 1].
    case fraction(Double)

    var scale: Double {
        switch self {
        case .full: 1
        case .fraction(let value): min(1, max(0.01, value))
        }
    }
}

public enum Blending: String, Sendable, Codable, CaseIterable {
    /// Multi-band blending across the seams.
    case multiBand
    /// Each pixel from one photo, values untouched: for measurements.
    case none
}

public enum SeamFinding: String, Sendable, Codable, CaseIterable {
    /// Seams along low-contrast paths, around objects (graph cut).
    case graphCut
    /// Seams halfway between photos.
    case voronoi
}

public enum ExposureCompensation: String, Sendable, Codable, CaseIterable {
    /// One gain per photo and colour channel.
    case channels
    /// Gains varying over each photo in blocks, for vignetting or uneven light.
    case blocks
    case none
}

/// Settings for joining the photos of a report into one image.
public struct StitchRequest: Sendable, Codable, Hashable {
    public var projection: Projection = .automatic
    public var size: OutputSize = .full
    public var blending: Blending = .multiBand
    public var seams: SeamFinding = .graphCut
    public var exposure: ExposureCompensation = .channels
    /// Rotating camera: level the horizon (wave correction).
    public var straighten = true

    public init() {}

    /// The measurement preset: no gains, seams around objects, pixels copied from one photo.
    public static var originalPixels: StitchRequest {
        var request = StitchRequest()
        request.blending = .none
        request.exposure = .none
        return request
    }
}

/// The global motion model the alignment settled on.
public enum GlobalModel: String, Sendable, Codable {
    case translation
    case similarity
    case affine
    case homography
    case rotation
}

public struct PixelRect: Sendable, Codable, Hashable {
    public var x: Int
    public var y: Int
    public var width: Int
    public var height: Int

    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    public var isEmpty: Bool { width <= 0 || height <= 0 }
    var cgRect: CGRect { CGRect(x: x, y: y, width: width, height: height) }
}

public struct PanoramaTimings: Sendable, Codable {
    public var alignment = 0.0
    /// Small copies, exposure gains and seams.
    public var seams = 0.0
    /// Full-resolution decode, warp and feed.
    public var compositing = 0.0
    public var blending = 0.0
    public var total = 0.0
}

/// A stitched panorama, with what is needed to show and export it.
public struct Panorama: Sendable, Identifiable {
    public let id = UUID()
    public var request: StitchRequest
    public var model: GlobalModel
    /// The projection used (never .automatic).
    public var projection: Projection
    /// The whole covered area; outside the photos alpha is 0.
    public var pixels: PanoramaPixels
    /// Largest rectangle without empty pixels, in `pixels` coordinates.
    public var crop: PixelRect
    /// 8-bit copy of the whole area, long side at most 4096, same colour space.
    public var preview: CGImage
    /// Outline of each photo in panorama pixels (y down).
    public var outlines: [Int: [Point2]]
    /// Photos in the panorama.
    public var imageIDs: [Int]
    /// Photos of the report that are not in it, with the reason.
    public var leftOut: [Int: ExclusionReason]
    /// `MatchReport.createdAt` of the analysis it was built from.
    public var reportCreatedAt: Date
    /// RMS transfer error of the correspondences after global alignment, full-resolution pixels.
    public var alignmentError: Double
    /// Things the user should know: a lower resolution than asked, pairs left out of the alignment, ...
    public var notes: [String]
    public var timings: PanoramaTimings

    public var size: PixelSize { PixelSize(width: pixels.width, height: pixels.height) }

    /// Cropping by default for a rotating camera, whose edges are always ragged; tiles and documents keep
    /// everything, so no specimen at the border is cut.
    public var cropsByDefault: Bool { model == .rotation && !crop.isEmpty }

    /// "IMG_5355-5360": first and last photo, without the part they share.
    public static func suggestedName(first: String, last: String) -> String {
        let a = (first as NSString).deletingPathExtension, b = (last as NSString).deletingPathExtension
        guard a != b, !b.isEmpty else { return a }
        var common = zip(a, b).prefix { $0 == $1 }.count
        // Back up to the start of a run of digits, so "IMG_5355" and "IMG_5360" give "IMG_5355-5360".
        while common > 0, b[b.index(b.startIndex, offsetBy: common - 1)].isNumber { common -= 1 }
        return "\(a)-\(b.dropFirst(common))"
    }
}
