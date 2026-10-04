import Foundation
import simd

/// Planar helpers for 3x3 transforms stored row-major, as in `ModelFit.transform`.
enum PlaneGeometry {
    static func matrix(_ rowMajor: [Double]) -> simd_double3x3 {
        // simd matrices are built from columns.
        simd_double3x3(
            SIMD3(rowMajor[0], rowMajor[3], rowMajor[6]),
            SIMD3(rowMajor[1], rowMajor[4], rowMajor[7]),
            SIMD3(rowMajor[2], rowMajor[5], rowMajor[8])
        )
    }

    static func apply(_ m: simd_double3x3, _ x: Double, _ y: Double) -> SIMD2<Double>? {
        let p = m * SIMD3(x, y, 1)
        guard p.z > 1e-9 else { return nil }
        return SIMD2(p.x / p.z, p.y / p.z)
    }

    /// Local linear part of the transform at `point`, by central differences.
    static func jacobian(_ m: simd_double3x3, at point: SIMD2<Double>) -> simd_double2x2? {
        let h = 0.5
        guard let px = apply(m, point.x + h, point.y), let nx = apply(m, point.x - h, point.y),
              let py = apply(m, point.x, point.y + h), let ny = apply(m, point.x, point.y - h)
        else { return nil }
        return simd_double2x2((px - nx) / (2 * h), (py - ny) / (2 * h))
    }

    /// Singular values of a 2x2 matrix, largest first.
    static func singularValues(_ m: simd_double2x2) -> (Double, Double) {
        let a = m.columns.0.x, c = m.columns.0.y, b = m.columns.1.x, d = m.columns.1.y
        let s1 = a * a + b * b + c * c + d * d
        let s2 = ((a * a + b * b - c * c - d * d) * (a * a + b * b - c * c - d * d)
                  + 4 * (a * c + b * d) * (a * c + b * d)).squareRoot()
        return (((s1 + s2) / 2).squareRoot(), (max(0, (s1 - s2) / 2)).squareRoot())
    }

    static func rectangle(_ size: PixelSize) -> [SIMD2<Double>] {
        let w = Double(size.width), h = Double(size.height)
        return [SIMD2(0, 0), SIMD2(w, 0), SIMD2(w, h), SIMD2(0, h)]
    }

    /// Region of image A that lands inside image B under `aToB`, as a convex polygon in A.
    /// With q = H p, the conditions z > 0, 0 <= x <= w z and 0 <= y <= h z are linear in p, so A's
    /// rectangle is clipped by five half-planes; this stays correct when part of A maps behind the camera.
    static func overlapInA(aToB: [Double], sizeA: PixelSize, sizeB: PixelSize) -> [SIMD2<Double>] {
        guard aToB.count == 9, aToB.allSatisfy(\.isFinite), abs(matrix(aToB).determinant) > 1e-12 else { return [] }
        let h = aToB, w = Double(sizeB.width), height = Double(sizeB.height)
        let x = { (p: SIMD2<Double>) in h[0] * p.x + h[1] * p.y + h[2] }
        let y = { (p: SIMD2<Double>) in h[3] * p.x + h[4] * p.y + h[5] }
        let z = { (p: SIMD2<Double>) in h[6] * p.x + h[7] * p.y + h[8] }
        let minimumDepth = 1e-8
        let polygon = clip(rectangle(sizeA), by: [
            { z($0) - minimumDepth }, { x($0) }, { w * z($0) - x($0) }, { y($0) }, { height * z($0) - y($0) },
        ])
        return signedArea(polygon) > 0 ? polygon : []
    }

    /// The same projective map with the sign that puts most of `points` in front of the camera (z > 0).
    /// H and -H are the same homography, but normalising h33 to 1 can pick the sign that puts them behind.
    static func facingPoints(_ rowMajor: [Double], _ points: [SIMD2<Double>]) -> [Double] {
        let behind = points.filter { rowMajor[6] * $0.x + rowMajor[7] * $0.y + rowMajor[8] < 0 }.count
        return 2 * behind > points.count ? rowMajor.map { -$0 } : rowMajor
    }

    static func signedArea(_ polygon: [SIMD2<Double>]) -> Double {
        guard polygon.count >= 3 else { return 0 }
        var twice = 0.0
        for index in polygon.indices {
            let p = polygon[index], q = polygon[(index + 1) % polygon.count]
            twice += p.x * q.y - q.x * p.y
        }
        return twice / 2
    }

    static func contains(_ polygon: [SIMD2<Double>], _ point: SIMD2<Double>) -> Bool {
        guard polygon.count >= 3, signedArea(polygon) != 0 else { return false }
        // Convex, counter-clockwise in a y-down frame means every cross product has the same sign.
        var sign = 0.0
        for index in polygon.indices {
            let p = polygon[index], q = polygon[(index + 1) % polygon.count]
            let cross = (q.x - p.x) * (point.y - p.y) - (q.y - p.y) * (point.x - p.x)
            if cross != 0 {
                if sign == 0 {
                    sign = cross
                } else if (cross > 0) != (sign > 0) {
                    return false
                }
            }
        }
        return true
    }

    /// Sutherland-Hodgman clipping of a convex polygon against an image rectangle.
    static func clip(_ polygon: [SIMD2<Double>], to size: PixelSize) -> [SIMD2<Double>] {
        let w = Double(size.width), h = Double(size.height)
        return clip(polygon, by: [{ $0.x }, { w - $0.x }, { $0.y }, { h - $0.y }])
    }

    /// The part of convex `polygon` inside convex `other`, whichever way either is wound.
    static func clip(_ polygon: [SIMD2<Double>], by other: [SIMD2<Double>]) -> [SIMD2<Double>] {
        let orientation = signedArea(other) >= 0 ? 1.0 : -1.0
        let halfPlanes = other.indices.map { index -> (SIMD2<Double>) -> Double in
            let p = other[index], q = other[(index + 1) % other.count]
            return { point in orientation * ((q.x - p.x) * (point.y - p.y) - (q.y - p.y) * (point.x - p.x)) }
        }
        return clip(polygon, by: halfPlanes)
    }

    /// Sutherland-Hodgman clipping by half-planes, each given as a function that is >= 0 inside.
    /// The functions must be affine in the point, so edge intersections can be interpolated.
    static func clip(_ polygon: [SIMD2<Double>], by halfPlanes: [(SIMD2<Double>) -> Double]) -> [SIMD2<Double>] {
        var output = polygon
        for distance in halfPlanes {
            let input = output
            output = []
            guard !input.isEmpty else { break }
            for index in input.indices {
                let current = input[index]
                let previous = input[(index + input.count - 1) % input.count]
                let dc = distance(current), dp = distance(previous)
                if dc >= 0 {
                    if dp < 0 { output.append(previous + (current - previous) * (dp / (dp - dc))) }
                    output.append(current)
                } else if dp >= 0 {
                    output.append(previous + (current - previous) * (dp / (dp - dc)))
                }
            }
        }
        return output
    }
}
