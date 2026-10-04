import StitchKit
import SwiftUI

extension StitchMode {
    var label: String {
        switch self {
        case .auto: String(localized: "Automatic")
        case .rotation: String(localized: "Rotation (landscape)")
        case .plane: String(localized: "Plane (microscope, trays)")
        case .document: String(localized: "Document")
        }
    }
}

extension MotionModel {
    var label: String {
        switch self {
        case .translation: String(localized: "translation")
        case .similarity: String(localized: "similarity")
        case .affine: String(localized: "affine")
        case .homography: String(localized: "homography")
        }
    }
}

extension FeatureSource {
    var label: String {
        switch self {
        case .rootSIFT: "RootSIFT"
        case .racoLightGlue: "RaCo + LightGlue"
        }
    }
}

extension EvidenceFilter {
    var label: String {
        switch self {
        case .best: String(localized: "Best")
        case .only(let source): source.label
        }
    }
}

extension PairVerdict {
    var label: String {
        switch self {
        case .verified: String(localized: "verified")
        case .tooFewMatches: String(localized: "too few matches")
        case .noConsistentModel: String(localized: "no consistent model")
        case .implausibleModel: String(localized: "implausible transform")
        case .degenerateInliers: String(localized: "inliers on a thin strip")
        case .lowConfidence: String(localized: "not distinguishable from chance")
        case .modelMismatch: String(localized: "perspective change, not plane tiles")
        }
    }

    var color: Color {
        switch self {
        case .verified: .green
        case .lowConfidence, .degenerateInliers, .implausibleModel, .modelMismatch: .orange
        case .tooFewMatches, .noConsistentModel: .secondary
        }
    }
}

extension ExclusionReason {
    var label: String {
        switch self {
        case .unreadable: String(localized: "cannot be read")
        case .tooFewFeatures: String(localized: "too few features")
        case .noVerifiedPair: String(localized: "no verified pair")
        case .weakLinkOnly: String(localized: "weak links only")
        case .separateGroup: String(localized: "separate group")
        case .inconsistentPairs: String(localized: "only pairs that disagree with the others")
        case .misplaced: String(localized: "placed implausibly by its pairs")
        case .excludedByUser: String(localized: "excluded")
        }
    }
}

extension LearnedBackend {
    var label: String {
        switch self {
        case .onnxCPU: "CPU (ONNX Runtime)"
        case .coreMLGPU: "GPU (Core ML)"
        case .coreMLNeuralEngine: "Neural Engine (Core ML)"
        case .coreMLAll: String(localized: "Automatic (Core ML)")
        }
    }
}

extension Projection {
    var label: String {
        switch self {
        case .automatic: String(localized: "Automatic")
        case .flat: String(localized: "Flat")
        case .rectilinear: String(localized: "Rectilinear")
        case .cylindrical: String(localized: "Cylindrical")
        case .spherical: String(localized: "Spherical")
        }
    }
}

extension GlobalModel {
    var label: String {
        switch self {
        case .translation: String(localized: "translation")
        case .similarity: String(localized: "similarity")
        case .affine: String(localized: "affine")
        case .homography: String(localized: "homography")
        case .rotation: String(localized: "rotating camera")
        }
    }
}

extension ProgressEvent {
    /// What the engine is doing, with the count when it has one.
    var label: String {
        let name = switch stage {
        case "sift": String(localized: "RootSIFT keypoints")
        case "lightglue-extract": String(localized: "RaCo-ALIKED keypoints")
        case "affinity": String(localized: "Choosing pairs")
        case "sift-match": String(localized: "RootSIFT matching")
        case "lightglue-match", "lightglue-early": String(localized: "LightGlue matching")
        case "verify": String(localized: "Geometric verification")
        case "bridge": String(localized: "Pairs between separate groups")
        case "overlap": String(localized: "Pairs the layout predicts")
        case "layout": String(localized: "Laying out the photos")
        case "align": String(localized: "Aligning the photos")
        case "seams": String(localized: "Finding seams")
        case "exposure": String(localized: "Seams and exposure")
        case "compose": String(localized: "Compositing")
        case "blend": String(localized: "Blending")
        default: stage
        }
        if let fraction { return "\(name): \(fraction.formatted(.percent.precision(.fractionLength(0))))" }
        return total > 0 ? "\(name): \(completed)/\(total)" : name
    }
}
