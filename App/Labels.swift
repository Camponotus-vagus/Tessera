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
