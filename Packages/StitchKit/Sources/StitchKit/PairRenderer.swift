import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Draws a pair side by side with tentative matches, inliers and outliers.
/// Mirrors the app's Pair view so results can be exported or checked from the command line.
public enum PairRenderer {
    public struct Style: Sendable {
        public var longSide = 1400
        public var showKeypoints = true
        public var showOutliers = true

        public init() {}
    }

    public static func render(_ pair: PairEvidence, report: MatchReport, style: Style = Style()) throws -> CGImage {
        guard let imageA = report.images.first(where: { $0.id == pair.a }),
              let imageB = report.images.first(where: { $0.id == pair.b })
        else { throw StitchError.engine("pair refers to unknown images") }
        let thumbA = try ImageLoader.thumbnail(url: imageA.url, longSide: style.longSide)
        let thumbB = try ImageLoader.thumbnail(url: imageB.url, longSide: style.longSide)
        let scaleA = Double(thumbA.width) / Double(imageA.pixelSize.width)
        let scaleB = Double(thumbB.width) / Double(imageB.pixelSize.width)
        let gap = 16
        let caption = 44
        let width = thumbA.width + gap + thumbB.width
        let height = max(thumbA.height, thumbB.height) + caption

        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw StitchError.engine("cannot create drawing context") }

        // Flip to a top-left origin so image coordinates can be used directly.
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        context.setFillColor(CGColor(gray: 0.1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        let originB = CGFloat(thumbA.width + gap)
        let top = CGFloat(caption)
        drawImage(thumbA, in: CGRect(x: 0, y: top, width: CGFloat(thumbA.width), height: CGFloat(thumbA.height)),
                  context: context)
        drawImage(thumbB, in: CGRect(x: originB, y: top, width: CGFloat(thumbB.width), height: CGFloat(thumbB.height)),
                  context: context)

        func pointA(_ p: Point2) -> CGPoint { CGPoint(x: Double(p.x) * scaleA, y: Double(top) + Double(p.y) * scaleA) }
        func pointB(_ p: Point2) -> CGPoint {
            CGPoint(x: Double(originB) + Double(p.x) * scaleB, y: Double(top) + Double(p.y) * scaleB)
        }

        if style.showKeypoints {
            context.setFillColor(CGColor(red: 0.6, green: 0.8, blue: 1, alpha: 0.35))
            for features in report.features where features.source == pair.source {
                guard features.imageID == pair.a || features.imageID == pair.b else { continue }
                for k in features.keypoints {
                    let p = features.imageID == pair.a ? pointA(Point2(x: k.x, y: k.y)) : pointB(Point2(x: k.x, y: k.y))
                    context.fillEllipse(in: CGRect(x: p.x - 1.2, y: p.y - 1.2, width: 2.4, height: 2.4))
                }
            }
        }

        let inliers = Set(pair.chosenFit?.inliers.map(Int.init) ?? [])
        if style.showOutliers {
            context.setStrokeColor(CGColor(red: 1, green: 0.25, blue: 0.2, alpha: 0.25))
            context.setLineWidth(0.8)
            for (index, match) in pair.matches.enumerated() where !inliers.contains(index) {
                context.move(to: pointA(match.a))
                context.addLine(to: pointB(match.b))
            }
            context.strokePath()
        }
        context.setStrokeColor(CGColor(red: 0.2, green: 1, blue: 0.3, alpha: 0.75))
        context.setLineWidth(1.0)
        for index in inliers.sorted() {
            let match = pair.matches[index]
            context.move(to: pointA(match.a))
            context.addLine(to: pointB(match.b))
        }
        context.strokePath()

        // Caption is drawn by the caller in the app; here a coloured band encodes the verdict.
        let band: CGColor = switch pair.verdict {
        case .verified: CGColor(red: 0.2, green: 0.7, blue: 0.3, alpha: 1)
        case .lowConfidence, .degenerateInliers, .implausibleModel, .modelMismatch: CGColor(red: 0.9, green: 0.7, blue: 0.1, alpha: 1)
        case .tooFewMatches, .noConsistentModel: CGColor(red: 0.85, green: 0.2, blue: 0.2, alpha: 1)
        }
        context.setFillColor(band)
        context.fill(CGRect(x: 0, y: 0, width: width, height: 8))

        guard let image = context.makeImage() else { throw StitchError.engine("cannot render pair") }
        return image
    }

    public static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw StitchError.engine("cannot write \(url.lastPathComponent)") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw StitchError.engine("cannot write \(url.lastPathComponent)")
        }
    }

    private static func drawImage(_ image: CGImage, in rect: CGRect, context: CGContext) {
        // CGContext.draw expects a bottom-left origin, so undo the flip locally.
        context.saveGState()
        context.translateBy(x: rect.minX, y: rect.maxY)
        context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(origin: .zero, size: rect.size))
        context.restoreGState()
    }
}
