// Renders Tessera's app icon: two overlapping photos of one landscape, joined by match lines
// between the same peaks, as in the pair view.
// usage: swift tools/icon/make-icon.swift <output.png>   (1024 x 1024)
import AppKit
import CoreGraphics

let output = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "icon.png")
let space = CGColorSpace(name: CGColorSpace.displayP3)!
let context = CGContext(data: nil, width: 1024, height: 1024, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
func color(_ r: Double, _ g: Double, _ b: Double, _ a: Double = 1) -> CGColor {
    CGColor(colorSpace: space, components: [r, g, b, a])!
}
func gradient(_ colors: [CGColor]) -> CGGradient {
    CGGradient(colorsSpace: space, colors: colors as CFArray, locations: nil)!
}

// Background: the macOS icon grid's 824 pt rounded square (y up).
let body = CGRect(x: 100, y: 100, width: 824, height: 824)
let shape = CGPath(roundedRect: body, cornerWidth: 185, cornerHeight: 185, transform: nil)
context.saveGState()
context.setShadow(offset: CGSize(width: 0, height: -12), blur: 28, color: color(0, 0, 0, 0.35))
context.addPath(shape)
context.setFillColor(color(0.06, 0.1, 0.19))
context.fillPath()
context.restoreGState()
context.saveGState()
context.addPath(shape)
context.clip()
context.drawLinearGradient(gradient([color(0.13, 0.2, 0.36), color(0.04, 0.07, 0.14)]),
                           start: CGPoint(x: 512, y: 924), end: CGPoint(x: 512, y: 100), options: [])

// The landscape, in scene units: 900 wide, 520 tall, y up.
let sceneWidth = 900.0, sceneHeight = 520.0
let ridge: [CGPoint] = [
    CGPoint(x: 0, y: 150), CGPoint(x: 90, y: 230), CGPoint(x: 170, y: 190), CGPoint(x: 280, y: 330),
    CGPoint(x: 360, y: 250), CGPoint(x: 450, y: 380), CGPoint(x: 540, y: 270), CGPoint(x: 610, y: 320),
    CGPoint(x: 700, y: 210), CGPoint(x: 800, y: 280), CGPoint(x: 900, y: 180),
]
let near: [CGPoint] = [
    CGPoint(x: 0, y: 90), CGPoint(x: 150, y: 140), CGPoint(x: 330, y: 100), CGPoint(x: 520, y: 150),
    CGPoint(x: 700, y: 95), CGPoint(x: 900, y: 130),
]
func drawScene(_ c: CGContext, warmth: Double) {
    c.drawLinearGradient(gradient([color(1.0, 0.62 + 0.05 * warmth, 0.38), color(0.98, 0.84, 0.6),
                                   color(0.55, 0.75, 0.95 - 0.05 * warmth)]),
                         start: CGPoint(x: 0, y: 0), end: CGPoint(x: 0, y: sceneHeight), options: [])
    c.setFillColor(color(1, 0.96, 0.82))
    c.fillEllipse(in: CGRect(x: 610, y: 370, width: 90, height: 90))
    let far = CGMutablePath()
    far.move(to: CGPoint(x: 0, y: 0))
    for p in ridge { far.addLine(to: p) }
    far.addLine(to: CGPoint(x: sceneWidth, y: 0))
    far.closeSubpath()
    c.addPath(far)
    c.setFillColor(color(0.33, 0.33, 0.55))
    c.fillPath()
    let front = CGMutablePath()
    front.move(to: CGPoint(x: 0, y: 0))
    for p in near { front.addLine(to: p) }
    front.addLine(to: CGPoint(x: sceneWidth, y: 0))
    front.closeSubpath()
    c.addPath(front)
    c.setFillColor(color(0.17, 0.24, 0.36))
    c.fillPath()
}

// Two photos looking at overlapping parts of the scene.
struct Photo { var center: CGPoint; var angle: Double; var window: ClosedRange<Double>; var warmth: Double }
let inner = CGSize(width: 360, height: 280)
let photos = [
    Photo(center: CGPoint(x: 345, y: 540), angle: 6, window: 0...560, warmth: 1),
    Photo(center: CGPoint(x: 680, y: 486), angle: -5, window: 340...900, warmth: -1),
]
// Scene point -> icon point, through a photo.
func place(_ p: CGPoint, in photo: Photo) -> CGPoint {
    let scale = inner.width / (photo.window.upperBound - photo.window.lowerBound)
    let local = CGPoint(x: (p.x - photo.window.lowerBound) * scale - inner.width / 2,
                        y: p.y * inner.height / sceneHeight - inner.height / 2)
    return local.applying(CGAffineTransform(translationX: photo.center.x, y: photo.center.y)
        .rotated(by: photo.angle * .pi / 180))
}
for photo in photos {
    let frame = CGAffineTransform(translationX: photo.center.x, y: photo.center.y).rotated(by: photo.angle * .pi / 180)
    var t = frame
    let border = CGPath(roundedRect: CGRect(x: -inner.width / 2 - 16, y: -inner.height / 2 - 16,
                                            width: inner.width + 32, height: inner.height + 32),
                        cornerWidth: 26, cornerHeight: 26, transform: &t)
    context.saveGState()
    context.setShadow(offset: CGSize(width: 0, height: -8), blur: 22, color: color(0, 0, 0, 0.5))
    context.addPath(border)
    context.setFillColor(color(0.97, 0.97, 0.97))
    context.fillPath()
    context.restoreGState()
    context.saveGState()
    context.concatenate(frame)
    context.addPath(CGPath(roundedRect: CGRect(x: -inner.width / 2, y: -inner.height / 2, width: inner.width,
                                               height: inner.height), cornerWidth: 12, cornerHeight: 12, transform: nil))
    context.clip()
    let scale = inner.width / (photo.window.upperBound - photo.window.lowerBound)
    context.translateBy(x: -inner.width / 2, y: -inner.height / 2)
    context.scaleBy(x: scale, y: inner.height / sceneHeight)
    context.translateBy(x: -photo.window.lowerBound, y: 0)
    drawScene(context, warmth: photo.warmth)
    context.restoreGState()
}

// Matches: peaks and corners of the ridge inside the overlap.
let features = [CGPoint(x: 360, y: 250), CGPoint(x: 450, y: 380), CGPoint(x: 540, y: 270), CGPoint(x: 520, y: 150),
                CGPoint(x: 400, y: 120)]
context.setLineCap(.round)
for feature in features {
    let a = place(feature, in: photos[0]), b = place(feature, in: photos[1])
    context.setStrokeColor(color(0.16, 0.92, 0.42))
    context.setLineWidth(8)
    context.move(to: a)
    context.addLine(to: b)
    context.strokePath()
    for p in [a, b] {
        context.setFillColor(color(1, 1, 1))
        context.fillEllipse(in: CGRect(x: p.x - 14, y: p.y - 14, width: 28, height: 28))
        context.setFillColor(color(0.16, 0.92, 0.42))
        context.fillEllipse(in: CGRect(x: p.x - 8.5, y: p.y - 8.5, width: 17, height: 17))
    }
}
context.restoreGState()

let image = context.makeImage()!
try NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])!.write(to: output)
print(output.path)
