// Global alignment of the photos of one connected group from their verified correspondences.
//
// Planar scenes (tiles, trays, documents) get one transform per photo onto the anchor's plane: translation,
// similarity and affine by weighted linear least squares with Huber reweighting, homographies by
// Levenberg-Marquardt. OpenCV's affine estimator, bundle adjusters and AffineWarper are not used: they are
// exact only for pure translations. A rotating camera goes through cv::detail: rotations from the pairwise
// homographies, ray bundle adjustment, and wave correction.

#include "stitchcore.h"

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>

#include "block_cholesky.hpp"
#include "common.hpp"
#include "interruptible_adjuster.hpp"
#include "progress.hpp"

using stitchcore::Monitor;
using stitchcore::write_error;

namespace {

struct Correspondence {
    cv::Vec2d a, b;
    double sigma;
};

struct Pair {
    int a, b;
    std::vector<Correspondence> points;
    cv::Matx33d homography;  // a -> b, raw pixels
};

cv::Matx33d matrix(const double values[9]) {
    return cv::Matx33d(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
                       values[8]);
}

void store(const cv::Matx33d &m, double *out) {
    for (int i = 0; i < 9; ++i) out[i] = m.val[i];
}

bool finite(const cv::Matx33d &m) {
    return std::all_of(m.val, m.val + 9, [](double v) { return std::isfinite(v); });
}

// Projective map of a point; false when the point lands behind the plane or at infinity.
bool apply(const cv::Matx33d &m, const cv::Vec2d &p, cv::Vec2d &out) {
    const cv::Vec3d q = m * cv::Vec3d(p[0], p[1], 1);
    if (!(q[2] > 1e-12) || !std::isfinite(q[0]) || !std::isfinite(q[1])) return false;
    out = cv::Vec2d(q[0] / q[2], q[1] / q[2]);
    return true;
}

// sqrt(|det J|) of the projective map at p: how many output pixels one input pixel becomes.
double stretch(const cv::Matx33d &m, const cv::Vec2d &p) {
    const cv::Vec3d q = m * cv::Vec3d(p[0], p[1], 1);
    const double w = q[2];
    if (std::fabs(w) < 1e-12) return INFINITY;
    const double x = q[0] / w, y = q[1] / w;
    const double j00 = (m(0, 0) - x * m(2, 0)) / w, j01 = (m(0, 1) - x * m(2, 1)) / w;
    const double j10 = (m(1, 0) - y * m(2, 0)) / w, j11 = (m(1, 1) - y * m(2, 1)) / w;
    return std::sqrt(std::fabs(j00 * j11 - j01 * j10));
}

double huber(double scaled, double k = 2.0) { return scaled <= k ? 1.0 : k / scaled; }

// Runs body(p) for every p below `count`, `grain` at a time on GCD's threads at the caller's priority, or on
// this thread when there is a single chunk. Each p runs on one thread from start to end, so what the body
// writes for p, and every sum taken afterwards in the order of p, is the same however many threads run.
template <class F>
void each(size_t count, size_t grain, const F &body) {
    const size_t chunks = (count + grain - 1) / grain;
    if (chunks < 2) {
        for (size_t p = 0; p < count; ++p) body(p);
        return;
    }
    const F *run = &body;
    dispatch_apply(chunks, DISPATCH_APPLY_AUTO, ^(size_t k) {
        const size_t end = std::min(count, (k + 1) * grain);
        for (size_t p = k * grain; p < end; ++p) (*run)(p);
    });
}

// Pairs per chunk of the parallel loops: a few hundred correspondences each.
constexpr size_t pair_grain = 4;

// Transfer error of every correspondence a -> b through the global transforms (image -> mosaic).
void transfer_errors(const std::vector<cv::Matx33d> &G, const std::vector<Pair> &pairs, double *pair_rms,
                     double &rms) {
    std::vector<double> sums(pairs.size());
    each(pairs.size(), pair_grain, [&](size_t p) {
        const cv::Matx33d to_b = G[pairs[p].b].inv() * G[pairs[p].a];
        double sum = 0;
        for (const Correspondence &c : pairs[p].points) {
            cv::Vec2d q;
            const double e = apply(to_b, c.a, q) ? cv::norm(q - c.b) : 1e6;
            sum = std::fma(e, e, sum);
        }
        sums[p] = sum;
    });
    double total = 0;
    size_t count = 0;
    for (size_t p = 0; p < pairs.size(); ++p) {
        const size_t n = pairs[p].points.size();
        if (pair_rms) pair_rms[p] = n ? std::sqrt(sums[p] / n) : 0;
        total += sums[p];
        count += n;
    }
    rms = count ? std::sqrt(total / count) : 0;
}

// Maximum spanning tree from `anchor` (weights: correspondence counts) as parent links, for initialisation.
std::vector<std::pair<int, int>> spanning_order(int n, const std::vector<Pair> &pairs, int anchor) {
    std::vector<bool> placed(n, false);
    placed[anchor] = true;
    std::vector<std::pair<int, int>> order;  // (pair index, image placed by it)
    for (int step = 1; step < n; ++step) {
        int best = -1;
        size_t weight = 0;
        for (size_t p = 0; p < pairs.size(); ++p) {
            if (placed[pairs[p].a] == placed[pairs[p].b]) continue;
            if (best < 0 || pairs[p].points.size() > weight) {
                best = static_cast<int>(p);
                weight = pairs[p].points.size();
            }
        }
        if (best < 0) break;
        const int image = placed[pairs[best].a] ? pairs[best].b : pairs[best].a;
        placed[image] = true;
        order.emplace_back(best, image);
    }
    return order;
}

// Normal equations with `d` unknowns per photo, coupled only between the photos of a pair: stored by
// d x d blocks (the lower block triangle, diagonal blocks whole, each block column-major) and solved with
// a sparse block Cholesky, analysed once and refactored at every step. Dense, they grew with the cube of
// the number of photos: 4184 unknowns for 524 homographies. Accelerate's sparse Cholesky gave different
// last bits from run to run, which could flip the drop and model choices; BlockCholesky gives the same.
class BlockNormalEquations {
public:
    BlockNormalEquations(int blocks, int d, const std::vector<std::pair<int, int>> &links) : blocks_(blocks), d_(d) {
        std::vector<std::vector<int>> rows(blocks);
        for (int j = 0; j < blocks; ++j) rows[j].push_back(j);
        for (const auto &[a, b] : links) {
            if (a < 0 || b < 0 || a == b) continue;
            rows[std::min(a, b)].push_back(std::max(a, b));
        }
        size_t stored = 0;
        for (int j = 0; j < blocks; ++j) {
            std::sort(rows[j].begin(), rows[j].end());
            rows[j].erase(std::unique(rows[j].begin(), rows[j].end()), rows[j].end());
            for (int i : rows[j]) index_[key(i, j)] = static_cast<int>(stored++);
        }
        values_.assign(stored * d * d, 0.0);
        rhs_.assign(static_cast<size_t>(blocks) * d, 0.0);
        cholesky_ = std::make_unique<stitchcore::BlockCholesky>(blocks, d, rows);
    }

    BlockNormalEquations(const BlockNormalEquations &) = delete;
    BlockNormalEquations &operator=(const BlockNormalEquations &) = delete;

    void clear() {
        std::fill(values_.begin(), values_.end(), 0.0);
        std::fill(rhs_.begin(), rhs_.end(), 0.0);
    }

    // The stored block that holds entries (bi, bj) and (bj, bi): column-major, row bi when bi >= bj.
    double *block(int bi, int bj) {
        return values_.data() + static_cast<size_t>(index_.at(key(std::max(bi, bj), std::min(bi, bj)))) * d_ * d_;
    }

    double &rhs(int k) { return rhs_[k]; }

    // Solves (A + damping * D) x = rhs + damping * D * towards, with D the diagonal of A on the first `damped`
    // unknowns of each block (all of them when negative, at least 1e-6 of the block's largest) and zero elsewhere, so that damping pulls those
    // unknowns towards `towards` (zero when it is null); false when the matrix is not positive definite.
    bool solve(std::vector<double> &x, double damping, const std::vector<double> *towards = nullptr,
               int damped = -1) {
        std::vector<double> values = values_;
        x = rhs_;
        if (damping != 0) {
            for (int j = 0; j < blocks_; ++j) {
                double *diagonal = values.data() + static_cast<size_t>(index_.at(key(j, j))) * d_ * d_;
                // An unknown no match constrains (every point of a photo on its centre row) gets a floor.
                double largest = 0;
                for (int k = 0; k < d_; ++k) largest = std::max(largest, diagonal[k * d_ + k]);
                for (int k = 0; k < (damped < 0 ? d_ : std::min(damped, d_)); ++k) {
                    const double extra = damping * std::max(diagonal[k * d_ + k], 1e-6 * largest);
                    if (towards) x[j * d_ + k] += extra * (*towards)[j * d_ + k];
                    diagonal[k * d_ + k] += extra;
                }
            }
        }
        if (!cholesky_->factor(values.data())) return false;
        cholesky_->solve(x.data());
        return std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); });
    }

private:
    static long long key(int i, int j) { return (static_cast<long long>(i) << 32) | static_cast<unsigned>(j); }

    int blocks_, d_;
    std::map<long long, int> index_;
    std::vector<double> values_, rhs_;
    std::unique_ptr<stitchcore::BlockCholesky> cholesky_;
};

// The blocks of the normal equations that a pair of photos fills (blocks ba and bb, -1 for the
// anchor): each residual row adds +w Ja^T Ja and +w Jb^T Jb on the diagonal, -w Ja^T Jb across.
struct PairBlocks {
    double *aa = nullptr, *bb = nullptr, *ab = nullptr;
    bool ab_row_a = true;  // the cross block is stored with a's unknowns as rows

    PairBlocks(BlockNormalEquations &system, int ba, int bb_) {
        if (ba >= 0) aa = system.block(ba, ba);
        if (bb_ >= 0) bb = system.block(bb_, bb_);
        if (ba >= 0 && bb_ >= 0) {
            ab = system.block(ba, bb_);
            ab_row_a = ba > bb_;
        }
    }

    // Adds one pair's sums (PairSums), in its own order.
    void add(int d, const double *sums) {
        const double *saa = sums, *sbb = sums + d * d, *sab = sums + 2 * d * d;
        for (int j = 0; j < d; ++j) {
            for (int i = 0; i < d; ++i) {
                if (aa) aa[j * d + i] += saa[j * d + i];
                if (bb) bb[j * d + i] += sbb[j * d + i];
                if (ab) ab[ab_row_a ? j * d + i : i * d + j] += sab[j * d + i];
            }
        }
    }
};

// One pair's share of the normal equations, summed over its correspondences on one thread: blocks aa, bb
// and ab (a's unknowns as rows), each d x d column-major, then the right-hand side of a and of b.
struct PairSums {
    int d;
    double *aa, *bb, *ab, *ra, *rb;

    static size_t stride(int d) { return static_cast<size_t>(3 * d * d + 2 * d); }

    PairSums(int d, double *sums) : d(d), aa(sums), bb(sums + d * d), ab(sums + 2 * d * d), ra(sums + 3 * d * d),
                                    rb(sums + 3 * d * d + d) {
        std::fill(sums, sums + stride(d), 0.0);
    }

    // One residual row r: Ja and Jb are its derivatives with respect to a's and b's d unknowns, and the
    // right-hand side gets -w Ja r for a and +w Jb r for b.
    void add(const double *Ja, const double *Jb, double w, double r) {
        for (int j = 0; j < d; ++j) {
            for (int i = 0; i < d; ++i) {
                aa[j * d + i] += w * Ja[i] * Ja[j];
                bb[j * d + i] += w * Jb[i] * Jb[j];
                ab[j * d + i] -= w * Ja[i] * Jb[j];
            }
        }
        for (int i = 0; i < d; ++i) {
            ra[i] -= w * Ja[i] * r;
            rb[i] += w * Jb[i] * r;
        }
    }
};

// Adds every pair's sums to the normal equations, in the order of the pairs.
void gather(BlockNormalEquations &system, std::vector<PairBlocks> &blocks, const std::vector<double> &sums,
            const std::vector<int> &slot_a, const std::vector<int> &slot_b, int d) {
    system.clear();
    const size_t stride = PairSums::stride(d);
    for (size_t p = 0; p < blocks.size(); ++p) {
        const double *s = sums.data() + p * stride;
        blocks[p].add(d, s);
        for (int i = 0; i < d; ++i) {
            if (slot_a[p] >= 0) system.rhs(slot_a[p] + i) += s[3 * d * d + i];
            if (slot_b[p] >= 0) system.rhs(slot_b[p] + i) += s[3 * d * d + d + i];
        }
    }
}

// MARK: Translation, similarity, affine

// M_i(p) = L_i (p - c_i) + t_i for every photo but the anchor, whose M is the identity.
// The residual M_a(p_a) - M_b(p_b) is linear in the parameters.
int dof(sc_align_model model) {
    switch (model) {
        case SC_ALIGN_TRANSLATION: return 2;
        case SC_ALIGN_SIMILARITY: return 4;
        default: return 6;
    }
}

// Rows of d M(p) / d theta (2 x dof) and the constant part of M(p), for a centred point.
void linear_rows(sc_align_model model, const cv::Vec2d &centred, double J[2][6], cv::Vec2d &constant) {
    const double x = centred[0], y = centred[1];
    for (int r = 0; r < 2; ++r) std::fill(J[r], J[r] + 6, 0.0);
    constant = cv::Vec2d(0, 0);
    switch (model) {
        case SC_ALIGN_TRANSLATION:
            J[0][0] = 1; J[1][1] = 1;
            constant = centred;
            break;
        case SC_ALIGN_SIMILARITY:  // L = [a -b; b a], theta = (a, b, tx, ty)
            J[0][0] = x; J[0][1] = -y; J[0][2] = 1;
            J[1][0] = y; J[1][1] = x;  J[1][3] = 1;
            break;
        default:  // theta = (a11, a12, a21, a22, tx, ty)
            J[0][0] = x; J[0][1] = y; J[0][4] = 1;
            J[1][2] = x; J[1][3] = y; J[1][5] = 1;
            break;
    }
}

cv::Matx33d linear_transform(sc_align_model model, const double *theta, const cv::Vec2d &c) {
    cv::Matx22d L;
    cv::Vec2d t(theta[model == SC_ALIGN_TRANSLATION ? 0 : model == SC_ALIGN_SIMILARITY ? 2 : 4],
                theta[model == SC_ALIGN_TRANSLATION ? 1 : model == SC_ALIGN_SIMILARITY ? 3 : 5]);
    switch (model) {
        case SC_ALIGN_TRANSLATION: L = cv::Matx22d(1, 0, 0, 1); break;
        case SC_ALIGN_SIMILARITY: L = cv::Matx22d(theta[0], -theta[1], theta[1], theta[0]); break;
        default: L = cv::Matx22d(theta[0], theta[1], theta[2], theta[3]); break;
    }
    const cv::Vec2d offset = t - L * c;
    return cv::Matx33d(L(0, 0), L(0, 1), offset[0], L(1, 0), L(1, 1), offset[1], 0, 0, 1);
}

bool solve_linear(sc_align_model model, const std::vector<cv::Size> &sizes, const std::vector<Pair> &pairs,
                  int anchor, std::vector<cv::Matx33d> &G, int &iterations, Monitor &monitor) {
    const int n = static_cast<int>(sizes.size()), d = dof(model);
    std::vector<int> slot(n, -1);
    int unknowns = 0;
    for (int i = 0; i < n; ++i) if (i != anchor) slot[i] = unknowns++ * d;
    const int P = unknowns * d;
    std::vector<cv::Vec2d> centre(n);
    for (int i = 0; i < n; ++i) centre[i] = cv::Vec2d(sizes[i].width * 0.5, sizes[i].height * 0.5);

    // Start from the identity on each photo's own centre: L = I, t = c.
    std::vector<double> theta(P, 0.0);
    for (int i = 0; i < n; ++i) {
        if (slot[i] < 0) continue;
        double *t = theta.data() + slot[i];
        if (model == SC_ALIGN_SIMILARITY) { t[0] = 1; t[2] = centre[i][0]; t[3] = centre[i][1]; }
        else if (model == SC_ALIGN_AFFINE) { t[0] = 1; t[3] = 1; t[4] = centre[i][0]; t[5] = centre[i][1]; }
        else { t[0] = centre[i][0]; t[1] = centre[i][1]; }
    }
    if (P == 0) {
        G.assign(n, cv::Matx33d::eye());
        return true;
    }

    std::vector<cv::Matx33d> M(n);
    auto evaluate = [&](int image, const cv::Vec2d &p, cv::Vec2d &value) {
        if (slot[image] < 0) { value = p; return; }
        apply(M[image], p, value);
    };

    std::vector<std::pair<int, int>> links;
    auto block = [&](int image) { return slot[image] < 0 ? -1 : slot[image] / d; };
    for (const Pair &pair : pairs) links.emplace_back(block(pair.a), block(pair.b));
    BlockNormalEquations system(unknowns, d, links);
    std::vector<PairBlocks> blocks;
    std::vector<int> slot_a, slot_b;
    for (const Pair &pair : pairs) {
        blocks.emplace_back(system, block(pair.a), block(pair.b));
        slot_a.push_back(slot[pair.a]);
        slot_b.push_back(slot[pair.b]);
    }
    std::vector<double> sums(pairs.size() * PairSums::stride(d));
    for (iterations = 0; iterations < 8; ++iterations) {
        monitor.check(iterations / 8.0);
        for (int i = 0; i < n; ++i)
            if (slot[i] >= 0) M[i] = linear_transform(model, theta.data() + slot[i], centre[i]);
        each(pairs.size(), pair_grain, [&](size_t p) {
            const Pair &pair = pairs[p];
            PairSums pair_sums(d, sums.data() + p * PairSums::stride(d));
            for (const Correspondence &c : pair.points) {
                cv::Vec2d ma, mb;
                evaluate(pair.a, c.a, ma);
                evaluate(pair.b, c.b, mb);
                // First pass: plain least squares; then Huber weights on the current residual.
                const double scaled = iterations == 0 ? 0 : cv::norm(ma - mb) / c.sigma;
                const double w = huber(scaled) / (c.sigma * c.sigma);
                double Ja[2][6], Jb[2][6];
                cv::Vec2d ka, kb;
                linear_rows(model, c.a - centre[pair.a], Ja, ka);
                linear_rows(model, c.b - centre[pair.b], Jb, kb);
                if (slot[pair.a] < 0) ka = c.a;
                if (slot[pair.b] < 0) kb = c.b;
                // r = Ja theta_a + ka - (Jb theta_b + kb); normal equations of sum w |r|^2.
                const cv::Vec2d k = ka - kb;
                for (int r = 0; r < 2; ++r) pair_sums.add(Ja[r], Jb[r], w, k[r]);
            }
        });
        gather(system, blocks, sums, slot_a, slot_b, d);
        // A degenerate pair (collinear points for an affine fit, a single point for a similarity) leaves the
        // matrix singular, and roundoff can still let the factorisation through, with any value in the free
        // directions. A tiny ridge on the linear part (all but the last two unknowns of a photo) keeps it at
        // its current value there, with the translation taking up the rest, and moves the other unknowns by
        // a part in a billion.
        std::vector<double> x;
        if (!system.solve(x, 1e-9, &theta, d - 2)) return false;
        double change = 0;
        for (int i = 0; i < P; ++i) {
            change = std::max(change, std::fabs(x[i] - theta[i]));
            theta[i] = x[i];
        }
        if (iterations > 0 && change < 1e-6) break;
    }
    G.assign(n, cv::Matx33d::eye());
    for (int i = 0; i < n; ++i) {
        if (slot[i] >= 0) G[i] = linear_transform(model, theta.data() + slot[i], centre[i]);
        if (!finite(G[i])) return false;
    }
    return true;
}

// MARK: Homographies

// Each photo's coordinates are normalised by its centre and half diagonal; the mosaic uses the anchor's
// normalisation. G_i = N_anchor^-1 * H_i * N_i, with H_i(2,2) = 1 and 8 parameters per photo.
cv::Matx33d normaliser(const cv::Size &size) {
    const double s = 2.0 / std::hypot(size.width, size.height);
    return cv::Matx33d(s, 0, -s * size.width * 0.5, 0, s, -s * size.height * 0.5, 0, 0, 1);
}

void homography_rows(const cv::Matx33d &H, const cv::Vec2d &p, double J[2][8], cv::Vec2d &value) {
    const cv::Vec3d q = H * cv::Vec3d(p[0], p[1], 1);
    const double w = q[2], x = q[0] / w, y = q[1] / w;
    value = cv::Vec2d(x, y);
    const double X = p[0] / w, Y = p[1] / w, W = 1 / w;
    const double row0[8] = {X, Y, W, 0, 0, 0, -x * X, -x * Y};
    const double row1[8] = {0, 0, 0, X, Y, W, -y * X, -y * Y};
    std::copy(row0, row0 + 8, J[0]);
    std::copy(row1, row1 + 8, J[1]);
}

cv::Matx33d from_parameters(const double *h) { return cv::Matx33d(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1); }

// Share of the homography solve done at iteration k: linear in k when it runs to the cap, quicker at first
// because most solves converge long before it.
double homography_fraction(double k) { return std::max(k / 60.0, 1.0 - std::exp(-k / 12.0)); }

bool solve_homographies(const std::vector<cv::Size> &sizes, const std::vector<Pair> &pairs, int anchor,
                        std::vector<cv::Matx33d> &G, int &iterations, Monitor &monitor) {
    const int n = static_cast<int>(sizes.size());
    std::vector<cv::Matx33d> N(n);
    for (int i = 0; i < n; ++i) N[i] = normaliser(sizes[i]);
    const cv::Matx33d Nm = N[anchor], Nm_inv = Nm.inv();
    const double mosaic_scale = 1.0 / Nm(0, 0);  // pixels of the anchor per normalised unit

    // Initialise by chaining the pairs along a maximum spanning tree, each pair by the affine map that
    // matches its homography where its correspondences are. Chaining the homographies themselves piled up
    // their perspective branch by branch, and at the end of a long branch of drone photos it reached the
    // horizon (scales of 9 to 60); the optimisation then adds the perspective the photos call for.
    auto local_affine = [](const Pair &pair) {
        cv::Vec2d c(0, 0);
        for (const Correspondence &k : pair.points) c += k.a;
        c *= 1.0 / std::max<size_t>(1, pair.points.size());
        const cv::Matx33d &H = pair.homography;
        const cv::Vec3d q = H * cv::Vec3d(c[0], c[1], 1);
        const double w = q[2], x = q[0] / w, y = q[1] / w;
        const double j00 = (H(0, 0) - x * H(2, 0)) / w, j01 = (H(0, 1) - x * H(2, 1)) / w;
        const double j10 = (H(1, 0) - y * H(2, 0)) / w, j11 = (H(1, 1) - y * H(2, 1)) / w;
        return cv::Matx33d(j00, j01, x - j00 * c[0] - j01 * c[1], j10, j11, y - j10 * c[0] - j11 * c[1], 0, 0, 1);
    };
    std::vector<cv::Matx33d> init(n, cv::Matx33d::eye());
    std::vector<bool> placed(n, false);
    placed[anchor] = true;
    for (const auto &[p, image] : spanning_order(n, pairs, anchor)) {
        const Pair &pair = pairs[p];
        const cv::Matx33d A = local_affine(pair);
        init[image] = image == pair.b ? init[pair.a] * A.inv() : init[pair.b] * A;
        placed[image] = true;
    }
    if (!std::all_of(placed.begin(), placed.end(), [](bool v) { return v; })) return false;
    monitor.check(0);

    std::vector<int> slot(n, -1);
    int unknowns = 0;
    for (int i = 0; i < n; ++i) if (i != anchor) slot[i] = unknowns++ * 8;
    const int P = unknowns * 8;
    std::vector<double> theta(P);
    auto normalised = [&](int i, const cv::Matx33d &g) {
        cv::Matx33d h = Nm * g * N[i].inv();
        return h * (1.0 / h(2, 2));
    };
    for (int i = 0; i < n; ++i) {
        if (slot[i] < 0) continue;
        const cv::Matx33d h = normalised(i, init[i]);
        std::copy(h.val, h.val + 8, theta.begin() + slot[i]);
    }
    std::vector<cv::Matx33d> Hn(n);
    auto refresh = [&](const std::vector<double> &t) {
        for (int i = 0; i < n; ++i) Hn[i] = slot[i] < 0 ? cv::Matx33d::eye() : from_parameters(t.data() + slot[i]);
    };

    // Normalised photo coordinates of every correspondence, computed once.
    struct Point { cv::Vec2d a, b; double sigma; };
    std::vector<std::vector<Point>> normal(pairs.size());
    for (size_t p = 0; p < pairs.size(); ++p) {
        for (const Correspondence &c : pairs[p].points) {
            cv::Vec2d a, b;
            apply(N[pairs[p].a], c.a, a);
            apply(N[pairs[p].b], c.b, b);
            normal[p].push_back({a, b, c.sigma});
        }
    }

    // Where each pair's correspondences start in the list of weights.
    std::vector<size_t> first(pairs.size() + 1, 0);
    for (size_t p = 0; p < pairs.size(); ++p) first[p + 1] = first[p] + normal[p].size();

    // Residuals in the mosaic plane, rescaled to photo pixels by the local stretch of each side: summed
    // pair by pair, then over the pairs in order.
    std::vector<double> pair_energy(pairs.size());
    auto energy = [&](const std::vector<double> *weights) -> double {
        each(pairs.size(), pair_grain, [&](size_t p) {
            const Pair &pair = pairs[p];
            double total = 0;
            size_t k = first[p];
            for (const Point &c : normal[p]) {
                cv::Vec2d ma, mb;
                if (!apply(Hn[pair.a], c.a, ma) || !apply(Hn[pair.b], c.b, mb)) {
                    total = INFINITY;
                    break;
                }
                const double sa = stretch(Hn[pair.a], c.a) * mosaic_scale * N[pair.a](0, 0);
                const double sb = stretch(Hn[pair.b], c.b) * mosaic_scale * N[pair.b](0, 0);
                const double to_pixels = mosaic_scale * 2.0 / (sa + sb);
                const double error = cv::norm(ma - mb) * to_pixels / c.sigma;
                const double w = weights ? (*weights)[k] : 1.0;
                total = std::fma(w * error, error, total);
                ++k;
            }
            pair_energy[p] = total;
        });
        double total = 0;
        for (double e : pair_energy) total += e;
        return total;
    };

    refresh(theta);
    double lambda = 1e-3;
    std::vector<double> weights;
    bool converged = false;
    std::vector<std::pair<int, int>> links;
    auto block = [&](int image) { return slot[image] < 0 ? -1 : slot[image] / 8; };
    for (const Pair &pair : pairs) links.emplace_back(block(pair.a), block(pair.b));
    BlockNormalEquations system(unknowns, 8, links);
    std::vector<PairBlocks> blocks;
    std::vector<int> slot_a, slot_b;
    for (const Pair &pair : pairs) {
        blocks.emplace_back(system, block(pair.a), block(pair.b));
        slot_a.push_back(slot[pair.a]);
        slot_b.push_back(slot[pair.b]);
    }
    std::vector<double> sums(pairs.size() * PairSums::stride(8));
    weights.assign(first.back(), 0.0);
    for (iterations = 0; iterations < 60 && !converged; ++iterations) {
        monitor.check(homography_fraction(iterations));
        // Huber weights on the current residuals, scale factors frozen for this step; the right-hand side
        // holds -g, the descent direction.
        each(pairs.size(), pair_grain, [&](size_t p) {
            const Pair &pair = pairs[p];
            PairSums pair_sums(8, sums.data() + p * PairSums::stride(8));
            size_t k = first[p];
            for (const Point &c : normal[p]) {
                double Ja[2][8], Jb[2][8];
                cv::Vec2d ma, mb;
                homography_rows(Hn[pair.a], c.a, Ja, ma);
                homography_rows(Hn[pair.b], c.b, Jb, mb);
                const double sa = stretch(Hn[pair.a], c.a) * mosaic_scale * N[pair.a](0, 0);
                const double sb = stretch(Hn[pair.b], c.b) * mosaic_scale * N[pair.b](0, 0);
                const double to_pixels = mosaic_scale * 2.0 / (sa + sb) / c.sigma;
                const cv::Vec2d r = (ma - mb) * to_pixels;
                const double hw = huber(cv::norm(r));
                weights[k++] = hw;
                const double w = hw * to_pixels * to_pixels;
                const cv::Vec2d rm = ma - mb;
                for (int row = 0; row < 2; ++row) pair_sums.add(Ja[row], Jb[row], w, rm[row]);
            }
        });
        gather(system, blocks, sums, slot_a, slot_b, 8);
        const double before = energy(&weights);
        bool improved = false;
        for (int attempt = 0; attempt < 10 && !improved; ++attempt) {
            monitor.check(homography_fraction(iterations));
            std::vector<double> step;
            if (!system.solve(step, lambda)) {
                lambda *= 10;
                continue;
            }
            std::vector<double> candidate = theta;
            for (int i = 0; i < P; ++i) candidate[i] += step[i];
            const std::vector<cv::Matx33d> saved = Hn;
            refresh(candidate);
            const double after = energy(&weights);
            if (after < before) {
                theta = candidate;
                lambda = std::max(lambda / 10, 1e-9);
                improved = true;
                converged = (before - after) < 1e-9 * before;
            } else {
                Hn = saved;
                lambda *= 10;
            }
        }
        if (!improved) break;
    }
    refresh(theta);
    G.assign(n, cv::Matx33d::eye());
    for (int i = 0; i < n; ++i) {
        G[i] = Nm_inv * Hn[i] * N[i];
        G[i] = G[i] * (1.0 / G[i](2, 2));
        if (!finite(G[i])) return false;
    }
    return true;
}

// Worst stretch over every photo's corners when `anchor`'s plane is the mosaic; INFINITY when a corner
// lands behind the plane.
double worst_stretch(const std::vector<cv::Matx33d> &G, const std::vector<cv::Size> &sizes, int anchor) {
    const cv::Matx33d to_anchor = G[anchor].inv();
    double worst = 1;
    for (size_t i = 0; i < G.size(); ++i) {
        const cv::Matx33d M = to_anchor * G[i];
        const cv::Vec2d corners[4] = {{0, 0}, {double(sizes[i].width), 0},
                                      {double(sizes[i].width), double(sizes[i].height)}, {0, double(sizes[i].height)}};
        for (const cv::Vec2d &c : corners) {
            cv::Vec2d q;
            if (!apply(M, c, q)) return INFINITY;
            const double s = stretch(M, c);
            if (!std::isfinite(s) || s <= 0) return INFINITY;
            worst = std::max(worst, std::max(s, 1 / s));
        }
    }
    return worst;
}

// MARK: Rotation

double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

cv::Matx33d centring(const cv::Size &size) {
    return cv::Matx33d(1, 0, size.width * 0.5, 0, 1, size.height * 0.5, 0, 0, 1);
}

// Homography from photo a to photo b implied by two cameras: x_b ~ K_b R_b^T R_a K_a^-1 x_a.
cv::Matx33d camera_homography(const cv::detail::CameraParams &a, const cv::detail::CameraParams &b) {
    cv::Mat Ka, Kb, Ra, Rb;
    a.K().convertTo(Ka, CV_64F);
    b.K().convertTo(Kb, CV_64F);
    a.R.convertTo(Ra, CV_64F);
    b.R.convertTo(Rb, CV_64F);
    cv::Mat H = Kb * Rb.t() * Ra * Ka.inv();
    return cv::Matx33d(H.ptr<double>());
}

// cv::detail's bundle adjusters read keypoints and matches: every correspondence becomes a pair of keypoints.
void detail_problem(const std::vector<cv::Size> &sizes, const std::vector<Pair> &pairs,
                    std::vector<cv::detail::ImageFeatures> &features, std::vector<cv::detail::MatchesInfo> &matches,
                    std::vector<double> &homography_focals) {
    const int n = static_cast<int>(sizes.size());
    features.assign(n, cv::detail::ImageFeatures());
    matches.assign(static_cast<size_t>(n) * n, cv::detail::MatchesInfo());
    homography_focals.clear();
    for (int i = 0; i < n; ++i) {
        features[i].img_idx = i;
        features[i].img_size = sizes[i];
    }
    for (const Pair &pair : pairs) {
        cv::detail::MatchesInfo info;
        info.src_img_idx = pair.a;
        info.dst_img_idx = pair.b;
        for (const Correspondence &c : pair.points) {
            const int qa = static_cast<int>(features[pair.a].keypoints.size());
            const int qb = static_cast<int>(features[pair.b].keypoints.size());
            features[pair.a].keypoints.emplace_back(cv::Point2f(float(c.a[0]), float(c.a[1])), 1.f);
            features[pair.b].keypoints.emplace_back(cv::Point2f(float(c.b[0]), float(c.b[1])), 1.f);
            info.matches.emplace_back(qa, qb, 0.f);
        }
        info.inliers_mask.assign(info.matches.size(), 1);
        info.num_inliers = static_cast<int>(info.matches.size());
        // cv::detail expects H in coordinates centred on each photo.
        cv::Matx33d H = centring(sizes[pair.b]).inv() * pair.homography * centring(sizes[pair.a]);
        H = H * (1.0 / H(2, 2));
        info.H = cv::Mat(H, true);
        info.confidence = 2.0;  // above the bundle adjuster's threshold of 1, below the matcher's cap of 3
        double f0 = 0, f1 = 0;
        bool ok0 = false, ok1 = false;
        cv::detail::focalsFromHomography(info.H, f0, f1, ok0, ok1);
        if (ok0 && ok1 && f0 > 0 && f1 > 0) homography_focals.push_back(std::sqrt(f0 * f1));
        cv::detail::MatchesInfo reverse = info;
        std::swap(reverse.src_img_idx, reverse.dst_img_idx);
        reverse.H = info.H.inv();
        for (cv::DMatch &m : reverse.matches) std::swap(m.queryIdx, m.trainIdx);
        matches[static_cast<size_t>(pair.a) * n + pair.b] = info;
        matches[static_cast<size_t>(pair.b) * n + pair.a] = reverse;
    }
}

// The pairs without the correspondences far off the cameras' fit, or empty when there is nothing to refit.
// The limit is three times the median error, and at least 2 px. A pair that would keep fewer than 8 of its
// points (or fewer than all of them, when it has fewer than 8) mostly shows something that moved: it is left
// out of the refit. Nothing is trimmed when no point is beyond the limit, when more than 30% are, or when
// the pairs left no longer join every photo.
std::vector<Pair> trimmed(int n, const std::vector<Pair> &pairs, const std::vector<cv::detail::CameraParams> &cameras) {
    std::vector<std::vector<double>> errors(pairs.size());
    std::vector<double> all;
    for (size_t p = 0; p < pairs.size(); ++p) {
        const cv::Matx33d H = camera_homography(cameras[pairs[p].a], cameras[pairs[p].b]);
        for (const Correspondence &c : pairs[p].points) {
            cv::Vec2d q;
            errors[p].push_back(apply(H, c.a, q) ? cv::norm(q - c.b) : INFINITY);
            all.push_back(errors[p].back());
        }
    }
    const double limit = std::max(3 * median(all), 2.0);
    std::vector<Pair> kept;
    size_t removed = 0;
    for (size_t p = 0; p < pairs.size(); ++p) {
        Pair pair = pairs[p];
        pair.points.clear();
        for (size_t k = 0; k < pairs[p].points.size(); ++k) {
            if (errors[p][k] <= limit) pair.points.push_back(pairs[p].points[k]);
        }
        removed += pairs[p].points.size() - pair.points.size();
        if (pair.points.size() >= std::min<size_t>(8, pairs[p].points.size())) kept.push_back(std::move(pair));
    }
    if (removed == 0 || removed > 0.3 * all.size()) return {};
    if (spanning_order(n, kept, 0).size() != size_t(n - 1)) return {};
    return kept;
}

bool solve_rotation(const sc_align_image *images, int n, const std::vector<Pair> &pairs, int wave,
                    std::vector<cv::detail::CameraParams> &cameras, int &method, std::string &problem,
                    Monitor &monitor) {
    std::vector<cv::Size> sizes(n);
    for (int i = 0; i < n; ++i) sizes[i] = cv::Size(images[i].width, images[i].height);

    // Shares of the call: setting up 0-0.03, the ray adjustment 0.03-0.60, the fixed-focal fallback or the
    // refit 0.60-0.97.
    monitor.range(0, 0.03);
    monitor.check(0);
    std::vector<cv::detail::ImageFeatures> features;
    std::vector<cv::detail::MatchesInfo> matches;
    std::vector<double> homography_focals;
    detail_problem(sizes, pairs, features, matches, homography_focals);
    monitor.check(0.5);

    // Focal prior: EXIF for every photo, else the homographies, else 72 degrees across the long side.
    std::vector<double> priors(n);
    bool exif = true;
    for (int i = 0; i < n; ++i) exif = exif && images[i].focal > 0;
    const double long_side = std::max(sizes[0].width, sizes[0].height);
    double fallback = long_side * 0.5 / std::tan(36.0 * CV_PI / 180.0);
    const double from_homographies = median(homography_focals);
    const double hfov = 2 * std::atan(long_side * 0.5 / std::max(from_homographies, 1e-9)) * 180 / CV_PI;
    const bool homographies_usable = from_homographies > 0 && hfov > 5 && hfov < 150;
    if (homographies_usable) fallback = from_homographies;
    for (int i = 0; i < n; ++i) priors[i] = exif ? images[i].focal : fallback;
    const bool trusted_prior = exif || homographies_usable;

    cameras.assign(n, cv::detail::CameraParams());
    for (int i = 0; i < n; ++i) {
        cameras[i].focal = priors[i];
        cameras[i].aspect = 1;
        cameras[i].ppx = sizes[i].width * 0.5;
        cameras[i].ppy = sizes[i].height * 0.5;
    }
    cv::detail::HomographyBasedEstimator estimator(true);
    if (!estimator(features, matches, cameras)) {
        problem = "the rotations could not be estimated";
        return false;
    }
    for (auto &camera : cameras) {
        cv::Mat r;
        camera.R.convertTo(r, CV_32F);
        camera.R = r;
    }
    monitor.check(1);
    const std::vector<cv::detail::CameraParams> initial = cameras;

    auto plausible = [&](const std::vector<cv::detail::CameraParams> &result) {
        std::vector<double> focals;
        for (const auto &c : result) {
            if (!std::isfinite(c.focal) || c.focal <= 0 || cv::countNonZero(c.R != c.R) > 0) return false;
            focals.push_back(c.focal);
        }
        const double m = median(focals);
        const double prior = median(priors);
        if (trusted_prior && (m < 0.67 * prior || m > 1.5 * prior)) return false;
        const auto [lo, hi] = std::minmax_element(focals.begin(), focals.end());
        return *hi <= 1.05 * *lo || !exif;
    };

    // Method 0 refines focal lengths and rotations, method 1 only the rotations, the focal length staying at
    // the prior; both for at most `iterations`, over the share [from, to] of the call. With a monitor the
    // adjusters report and stop through it; their results are the same.
    auto adjust = [&](int kind, const std::vector<cv::detail::ImageFeatures> &f,
                      const std::vector<cv::detail::MatchesInfo> &m, std::vector<cv::detail::CameraParams> &c,
                      double from, double to, int iterations = 200) {
        monitor.range(from, to);
        monitor.check(0);
        cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
        if (kind == 0) {
            adjuster = monitor.active()
                           ? cv::Ptr<cv::detail::BundleAdjusterBase>(
                                 new stitchcore::Interruptible<cv::detail::BundleAdjusterRay>(monitor))
                           : cv::Ptr<cv::detail::BundleAdjusterBase>(cv::makePtr<cv::detail::BundleAdjusterRay>());
            adjuster->setTermCriteria(
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iterations, 1e-8));
        } else {
            // The adjuster's default allows 1000 iterations, each costing minutes with hundreds of photos.
            adjuster = monitor.active()
                           ? cv::Ptr<cv::detail::BundleAdjusterBase>(
                                 new stitchcore::Interruptible<cv::detail::BundleAdjusterReproj>(monitor))
                           : cv::Ptr<cv::detail::BundleAdjusterBase>(cv::makePtr<cv::detail::BundleAdjusterReproj>());
            adjuster->setRefinementMask(cv::Mat::zeros(3, 3, CV_8U));
            adjuster->setTermCriteria(
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, iterations, DBL_EPSILON));
        }
        adjuster->setConfThresh(1.0);
        try {
            return (*adjuster)(f, m, c) && plausible(c);
        } catch (const cv::Exception &) {
            return false;
        }
    };
    method = 0;
    bool ok = adjust(0, features, matches, cameras, 0.03, 0.60);
    if (!ok) {
        method = 1;
        cameras = initial;
        ok = adjust(1, features, matches, cameras, 0.60, 0.97);
    }
    if (!ok) {
        problem = "the camera focal length does not fit the matches (not a rotating camera?)";
        return false;
    }
    // The ray adjuster weighs every match alike, so whatever moved between shots (water, people) pulls the
    // whole panorama: refit once from its result without the matches far off it. Not with the focal length
    // fixed at the prior, where the errors measure the wrong focal length more than anything that moved.
    if (method == 0) {
        monitor.range(0.60, 0.97);
        monitor.check(0);
        try {
            const std::vector<Pair> kept = trimmed(n, pairs, cameras);
            if (!kept.empty()) {
                detail_problem(sizes, kept, features, matches, homography_focals);
                std::vector<cv::detail::CameraParams> refit = cameras;
                if (adjust(0, features, matches, refit, 0.60, 0.97, 20)) {
                    // The adjuster fixes the photo at the centre of its spanning tree, which the trimmed
                    // matches can move: keep the first result's reference photo where it was.
                    int reference = 0;
                    double closest = INFINITY;
                    for (int i = 0; i < n; ++i) {
                        cv::Mat R;
                        cameras[i].R.convertTo(R, CV_64F);
                        const double d = cv::norm(R, cv::Mat::eye(3, 3, CV_64F));
                        if (d < closest) {
                            closest = d;
                            reference = i;
                        }
                    }
                    cv::Mat first, second;
                    cameras[reference].R.convertTo(first, CV_64F);
                    refit[reference].R.convertTo(second, CV_64F);
                    const cv::Mat gauge = first * second.t();
                    for (auto &camera : refit) {
                        cv::Mat R;
                        camera.R.convertTo(R, CV_64F);
                        R = gauge * R;
                        R.convertTo(camera.R, CV_32F);
                    }
                    cameras = refit;
                }
            }
        } catch (const std::exception &) {
            // The refit is optional: keep the first result.
        }
    }

    // Wave correction levels the horizon; it needs a wide enough sweep to know where "up" is.
    if (wave >= 0 && n >= 3) {
        std::vector<double> yaws;
        for (const auto &c : cameras) {
            cv::Mat R64;
            c.R.convertTo(R64, CV_64F);
            const cv::Vec3d axis(R64.at<double>(0, 2), R64.at<double>(1, 2), R64.at<double>(2, 2));
            yaws.push_back(std::atan2(axis[0], axis[2]));
        }
        std::sort(yaws.begin(), yaws.end());
        double gap = yaws.front() + 2 * CV_PI - yaws.back();
        for (size_t i = 1; i < yaws.size(); ++i) gap = std::max(gap, yaws[i] - yaws[i - 1]);
        const double extent = (2 * CV_PI - gap) * 180 / CV_PI;
        if (extent >= 30) {
            std::vector<cv::Mat> rotations;
            for (const auto &c : cameras) rotations.push_back(c.R.clone());
            const cv::detail::WaveCorrectKind kind =
                wave == 2 ? cv::detail::autoDetectWaveCorrectKind(rotations) : cv::detail::WaveCorrectKind(wave);
            cv::detail::waveCorrect(rotations, kind);
            for (int i = 0; i < n; ++i) cameras[i].R = rotations[i];
        }
    }
    return true;
}

}  // namespace

extern "C" int32_t sc_align(sc_align_model model, const sc_align_image *images, int32_t image_count,
                            const sc_align_pair *pairs, int32_t pair_count, int32_t anchor, int32_t wave,
                            double *transforms, double *focals, double *pair_rms, sc_align_result *result,
                            const sc_progress *progress, char *error, size_t error_length) {
    if (result) *result = sc_align_result{};
    if (images == nullptr || transforms == nullptr || result == nullptr || image_count < 2 || pair_count < 1 ||
        pairs == nullptr || anchor < 0 || anchor >= image_count || (model == SC_ALIGN_ROTATION && focals == nullptr)) {
        write_error(error, error_length, "invalid alignment input");
        return 1;
    }
    Monitor monitor(progress);
    try {
        std::vector<cv::Size> sizes(image_count);
        for (int i = 0; i < image_count; ++i) {
            if (images[i].width < 1 || images[i].height < 1) throw std::runtime_error("invalid image size");
            sizes[i] = cv::Size(images[i].width, images[i].height);
        }
        std::vector<Pair> list;
        std::vector<int> origin;  // index in `pairs` of each entry of `list`
        for (int p = 0; p < pair_count; ++p) {
            const sc_align_pair &in = pairs[p];
            if (in.a < 0 || in.b < 0 || in.a >= image_count || in.b >= image_count || in.a == in.b ||
                in.count < 1 || in.points == nullptr) {
                throw std::runtime_error("invalid pair");
            }
            Pair pair{in.a, in.b, {}, matrix(in.homography)};
            for (int k = 0; k < in.count; ++k) {
                const float *v = in.points + 4 * k;
                const double sigma = in.sigma ? std::max(0.1, double(in.sigma[k])) : 1.0;
                if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]) || !std::isfinite(v[3])) continue;
                pair.points.push_back({cv::Vec2d(v[0], v[1]), cv::Vec2d(v[2], v[3]), sigma});
            }
            if (!pair.points.empty()) {
                list.push_back(std::move(pair));
                origin.push_back(p);
            }
        }
        // A pair without a finite point gets an error of 0.
        std::vector<double> list_rms(list.size(), 0.0);
        auto report_pairs = [&] {
            if (!pair_rms) return;
            std::fill(pair_rms, pair_rms + pair_count, 0.0);
            for (size_t p = 0; p < list.size(); ++p) pair_rms[origin[p]] = list_rms[p];
        };
        // Every photo must be reachable from the anchor.
        if (spanning_order(image_count, list, anchor).size() != size_t(image_count - 1)) {
            write_error(error, error_length, "the pairs do not connect every photo");
            return 1;
        }
        monitor.check(0);

        std::vector<cv::Matx33d> G;
        int iterations = 0;
        if (model == SC_ALIGN_ROTATION) {
            std::vector<cv::detail::CameraParams> cameras;
            int method = 0;
            std::string problem;
            if (!solve_rotation(images, image_count, list, wave, cameras, method, problem, monitor)) {
                write_error(error, error_length, problem);
                return 1;
            }
            for (int i = 0; i < image_count; ++i) {
                cv::Mat R64;
                cameras[i].R.convertTo(R64, CV_64F);
                store(cv::Matx33d(R64.ptr<double>()), transforms + 9 * i);
                focals[i] = cameras[i].focal;
            }
            // Transfer errors through the camera homographies, in the same units as the planar models.
            double total = 0;
            size_t count = 0;
            for (size_t p = 0; p < list.size(); ++p) {
                const cv::Matx33d H = camera_homography(cameras[list[p].a], cameras[list[p].b]);
                double sum = 0;
                for (const Correspondence &c : list[p].points) {
                    cv::Vec2d q;
                    const double e = apply(H, c.a, q) ? cv::norm(q - c.b) : 1e6;
                    sum = std::fma(e, e, sum);
                }
                list_rms[p] = std::sqrt(sum / list[p].points.size());
                total += sum;
                count += list[p].points.size();
            }
            report_pairs();
            result->ok = 1;
            result->rms = count ? std::sqrt(total / count) : 0;
            result->iterations = method;
            return 0;
        }

        bool solved = model == SC_ALIGN_HOMOGRAPHY ? solve_homographies(sizes, list, anchor, G, iterations, monitor)
                                                   : solve_linear(model, sizes, list, anchor, G, iterations, monitor);
        if (!solved) {
            write_error(error, error_length, "the alignment did not converge");
            return 1;
        }
        double rms = 0;
        transfer_errors(G, list, list_rms.data(), rms);
        report_pairs();
        for (int i = 0; i < image_count; ++i) store(G[i], transforms + 9 * i);
        result->ok = 1;
        result->rms = rms;
        result->iterations = iterations;
        return 0;
    } catch (const stitchcore::Cancelled &) {
        write_error(error, error_length, "cancelled");
        return 2;
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return 1;
    } catch (...) {
        write_error(error, error_length, "unexpected failure in the alignment");
        return 1;
    }
}

extern "C" double sc_alignment_stretch(const double *transforms, const int32_t *sizes, int32_t image_count,
                                       int32_t anchor) {
    if (transforms == nullptr || sizes == nullptr || image_count < 1 || anchor < 0 || anchor >= image_count) {
        return INFINITY;
    }
    std::vector<cv::Matx33d> G(image_count);
    std::vector<cv::Size> s(image_count);
    for (int i = 0; i < image_count; ++i) {
        G[i] = matrix(transforms + 9 * i);
        s[i] = cv::Size(sizes[2 * i], sizes[2 * i + 1]);
    }
    try {
        return worst_stretch(G, s, anchor);
    } catch (const std::exception &) {
        return INFINITY;
    }
}
