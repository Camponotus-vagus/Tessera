// Sparse Cholesky factorisation of a symmetric positive definite matrix made of d x d blocks, for the normal
// equations of the planar alignment. Accelerate's sparse Cholesky runs its elimination tree on several threads
// and sums the updates in the order they finish, so the same matrix gave different last bits from run to run;
// here every sum runs in a fixed order on the calling thread, and the same matrix always gives the same bits.
//
// Ordering: minimum fill on the graph of the blocks (ties to the lowest degree, then the lowest index), which
// needs 10-15% fewer block products than minimum degree on the alignment graphs; minimum degree past 8192
// blocks. Columns of L with the same structure below them are grouped into supernodes, stored by block rows, so
// that the blocks a row has in a supernode are contiguous: every update of a block of L by a supernode is one
// product of two contiguous d x (w d) panels. Factorisation: each block of L is computed once, in column order:
// it starts from its block of A, takes the product of every earlier supernode that reaches it (a list fixed in
// the analysis, in increasing column order), and is then divided by the diagonal block of its column.
// For d = 2, 4, 6, 8 the kernels run on NEON, in registers, with a fixed order of fused multiply-adds, and the
// product of each supernode is summed from zero on its own before it is subtracted from the block, as in the
// update buffers of a supernodal solver: the sum of a block is split into one chain per supernode, of its width
// times d terms, which keeps the rounding error close to Accelerate's. Other d (not used by the alignment) take plain loops with one
// running sum.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "block_cholesky.hpp needs IEEE NaN and infinity: factor() rejects them and solve() reports failure with NaN"
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#define STITCHCORE_CHOL_NEON 1
#endif

namespace stitchcore {

class BlockCholesky {
public:
    // `columns[j]` lists the rows i >= j of the stored blocks of column j: strictly increasing, in [0, blocks),
    // the diagonal first. Throws std::invalid_argument otherwise, and std::length_error for a factor whose
    // offsets do not fit an int (hundreds of gigabytes).
    BlockCholesky(int blocks, int d, const std::vector<std::vector<int>> &columns) : n_(blocks), d_(d) {
        if (blocks < 0 || d < 1 || columns.size() < static_cast<size_t>(blocks))
            throw std::invalid_argument("BlockCholesky: wrong sizes");
        for (int j = 0; j < blocks; ++j) {
            const std::vector<int> &rows = columns[j];
            if (rows.empty() || rows[0] != j) throw std::invalid_argument("BlockCholesky: a column lacks its diagonal");
            for (size_t k = 1; k < rows.size(); ++k)
                if (rows[k] <= rows[k - 1] || rows[k] >= blocks)
                    throw std::invalid_argument("BlockCholesky: rows out of order or out of range");
        }
        order(columns);
        analyse(columns);
    }

    // Factors the matrix whose stored blocks are `values`, in the order of `columns` given to the constructor,
    // each block column-major with row i, column j; false when it is not positive definite. The diagonal blocks
    // are read from their lower triangle only, as Accelerate does.
    bool factor(const double *values) {
        factored_ = false;
#if STITCHCORE_CHOL_NEON
        switch (d_) {
            case 2: factored_ = factor_neon<2>(values); break;
            case 4: factored_ = factor_neon<4>(values); break;
            case 6: factored_ = factor_neon<6>(values); break;
            case 8: factored_ = factor_neon<8>(values); break;
            default: factored_ = factor_generic(values); break;
        }
#else
        factored_ = factor_generic(values);
#endif
        return factored_;
    }

    // Overwrites `x` (n * d values, the right-hand side b) with the solution of A y = b. Without a successful factor() (none yet, or
    // the last one returned false) it fills `x` with quiet NaN instead, so that a caller that checks the
    // solution for finite values rejects it.
    void solve(double *x) const {
        if (!factored_) {
            const size_t count = static_cast<size_t>(n_) * d_;
            for (size_t k = 0; k < count; ++k) x[k] = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        switch (d_) {
            case 2: solve_d<2>(x); break;
            case 4: solve_d<4>(x); break;
            case 6: solve_d<6>(x); break;
            case 8: solve_d<8>(x); break;
            default: solve_d<0>(x); break;
        }
    }

private:
    // One block of L: where it goes, the block of A it starts from (-1 for fill), and its updates.
    struct Task {
        int target;       // offset in L_, in blocks
        int source;       // stored block of A, or -1
        int first, last;  // entries_[first .. last)
        int transposed;   // the stored block is A(j, i): take its transpose
    };
    // L(i, j) -= P_i P_j^T, with P_i the `width` blocks at block offset `a` (d x width d, column-major) and P_j those at `b`.
    struct Entry {
        int a, b, width;
    };

    // Minimum fill: eliminating a block joins its remaining neighbours into a clique; each step eliminates the
    // block whose elimination adds the fewest new edges, ties to the lowest degree and then the lowest index.
    // The graph is kept as bitsets; when a block goes, only its neighbours have their counts updated, and the
    // blocks next to both ends of each new edge lose one missing pair.
    void order(const std::vector<std::vector<int>> &columns) {
        // The bitsets take n^2 / 8 bytes: past 8192 blocks (8 MB), minimum degree on adjacency lists.
        if (n_ > 8192) {
            order_minimum_degree(columns);
            return;
        }
        const int n = n_, W = (n + 63) / 64;
        std::vector<uint64_t> graph(static_cast<size_t>(n) * W, 0);
        // Words [low, high] of each row may be nonzero: the loops below only visit those.
        std::vector<int> low(n, W), high(n, -1);
        auto row = [&](int v) { return graph.data() + static_cast<size_t>(v) * W; };
        auto link = [&](int a, int b) {
            row(a)[b >> 6] |= uint64_t(1) << (b & 63);
            low[a] = std::min(low[a], b >> 6);
            high[a] = std::max(high[a], b >> 6);
        };
        for (int j = 0; j < n; ++j)
            for (int i : columns[j])
                if (i != j) {
                    link(i, j);
                    link(j, i);
                }
        auto count_degree = [&](int v) {
            int count = 0;
            const uint64_t *a = row(v);
            for (int w = low[v]; w <= high[v]; ++w) count += __builtin_popcountll(a[w]);
            return count;
        };
        // Pairs of neighbours of v that are not adjacent.
        auto count_fill = [&](int v) {
            long missing = 0;
            const uint64_t *a = row(v);
            for (int w = low[v]; w <= high[v]; ++w)
                for (uint64_t bits = a[w]; bits; bits &= bits - 1) {
                    const uint64_t *b = row(w * 64 + __builtin_ctzll(bits));
                    for (int x = low[v]; x <= high[v]; ++x) missing += __builtin_popcountll(a[x] & ~b[x]);
                    missing -= 1;  // the neighbour itself
                }
            return missing / 2;
        };
        std::vector<int> degree(n);
        std::vector<long> fill(n);
        std::vector<char> alive(n, 1);
        for (int v = 0; v < n; ++v) {
            degree[v] = count_degree(v);
            fill[v] = count_fill(v);
        }
        perm_.clear();
        pattern_.assign(n, {});
        std::vector<uint64_t> clique(W, 0);
        std::vector<int> members;
        for (int step = 0; step < n; ++step) {
            int best = -1;
            for (int v = 0; v < n; ++v)
                if (alive[v] && (best < 0 || fill[v] < fill[best] || (fill[v] == fill[best] && degree[v] < degree[best]))) best = v;
            if (fill[best] == 0 && degree[best] == n - step - 1) {
                // What is left is a clique: no more fill, every degree the same, so the rest goes by index.
                std::vector<int> rest;
                for (int v = 0; v < n; ++v)
                    if (alive[v]) rest.push_back(v);
                for (size_t k = 0; k < rest.size(); ++k) {
                    perm_.push_back(rest[k]);
                    pattern_[rest[k]].assign(rest.begin() + k + 1, rest.end());
                }
                break;
            }
            alive[best] = 0;
            perm_.push_back(best);
            const int clo = low[best], chi = high[best];
            for (int w = clo; w <= chi; ++w) clique[w] = row(best)[w];
            members.clear();
            for (int w = clo; w <= chi; ++w)
                for (uint64_t bits = clique[w]; bits; bits &= bits - 1) members.push_back(w * 64 + __builtin_ctzll(bits));
            pattern_[best] = members;
            for (int x : members) row(x)[best >> 6] &= ~(uint64_t(1) << (best & 63));
            for (int w = clo; w <= chi; ++w) row(best)[w] = 0;
            // Each member x keeps its neighbours outside the clique (A) and gains the members it was not
            // next to (B): the pairs it loses are those with `best`, one per block of A, and those among the
            // members it was already next to, which become adjacent (counted below with the other new edges);
            // it gains one missing pair for each block of A that is not next to a block of B.
            for (int x : members) {
                const uint64_t *a = row(x);
                int outside = 0, gained = 0;
                for (int w = low[x]; w <= high[x]; ++w) {
                    const uint64_t in = w >= clo && w <= chi ? clique[w] : 0;
                    outside += __builtin_popcountll(a[w] & ~in);
                }
                long added = 0;
                for (int w = clo; w <= chi; ++w)
                    for (uint64_t bits = clique[w] & ~a[w]; bits; bits &= bits - 1) {
                        const int y = w * 64 + __builtin_ctzll(bits);
                        if (y == x) continue;
                        ++gained;
                        const uint64_t *b = row(y);
                        for (int z = low[x]; z <= high[x]; ++z) {
                            const uint64_t in = z >= clo && z <= chi ? clique[z] : 0;
                            added += __builtin_popcountll(a[z] & ~in & ~b[z]);
                        }
                    }
                fill[x] += added - outside;
                degree[x] += gained - 1;
            }
            // New edges x - y: every block next to both, in the clique or not, has one missing pair less.
            for (int x : members) {
                const uint64_t *a = row(x);
                for (int w = x >> 6; w <= chi; ++w) {
                    uint64_t bits = clique[w] & ~a[w];
                    if (w == (x >> 6)) bits &= ~((uint64_t(2) << (x & 63)) - 1);  // y > x
                    for (; bits; bits &= bits - 1) {
                        const int y = w * 64 + __builtin_ctzll(bits);
                        const uint64_t *b = row(y);
                        for (int z = std::max(low[x], low[y]), last = std::min(high[x], high[y]); z <= last; ++z)
                            for (uint64_t common = a[z] & b[z]; common; common &= common - 1) --fill[z * 64 + __builtin_ctzll(common)];
                    }
                }
            }
            for (int x : members) {
                uint64_t *a = row(x);
                for (int w = clo; w <= chi; ++w) a[w] |= clique[w];
                a[x >> 6] &= ~(uint64_t(1) << (x & 63));
                low[x] = std::min(low[x], clo);
                high[x] = std::max(high[x], chi);
            }
            for (int w = clo; w <= chi; ++w) clique[w] = 0;
        }
    }

    // Minimum degree on adjacency lists, ties to the lowest index: the ordering for graphs too large for the
    // bitsets of order().
    void order_minimum_degree(const std::vector<std::vector<int>> &columns) {
        std::vector<std::vector<int>> adjacent(n_);
        for (int j = 0; j < n_; ++j)
            for (int i : columns[j])
                if (i != j) {
                    adjacent[i].push_back(j);
                    adjacent[j].push_back(i);
                }
        for (auto &list : adjacent) {
            std::sort(list.begin(), list.end());
            list.erase(std::unique(list.begin(), list.end()), list.end());
        }
        std::vector<bool> eliminated(n_, false);
        perm_.clear();
        pattern_.assign(n_, {});
        std::vector<int> merged;
        for (int step = 0; step < n_; ++step) {
            int best = -1;
            for (int v = 0; v < n_; ++v)
                if (!eliminated[v] && (best < 0 || adjacent[v].size() < adjacent[best].size())) best = v;
            eliminated[best] = true;
            perm_.push_back(best);
            const std::vector<int> neighbours = adjacent[best];
            pattern_[best] = neighbours;
            for (int u : neighbours) {
                merged.clear();
                std::set_union(adjacent[u].begin(), adjacent[u].end(), neighbours.begin(), neighbours.end(),
                               std::back_inserter(merged));
                merged.erase(std::remove_if(merged.begin(), merged.end(), [&](int w) { return w == u || w == best; }),
                             merged.end());
                adjacent[u].swap(merged);
            }
            adjacent[best].clear();
        }
    }

    void analyse(const std::vector<std::vector<int>> &columns) {
        std::vector<int> position(n_);
        for (int k = 0; k < n_; ++k) position[perm_[k]] = k;
        // Rows below the diagonal of each column of L, in the new order.
        std::vector<std::vector<int>> below(n_);
        for (int k = 0; k < n_; ++k) {
            for (int v : pattern_[perm_[k]]) below[k].push_back(position[v]);
            std::sort(below[k].begin(), below[k].end());
        }
        pattern_.clear();
        // Supernodes: column k + 1 joins column k's when k's rows below are k + 1 and then exactly k + 1's.
        node_.assign(n_, 0);
        node_first_.clear();
        for (int k = 0; k < n_; ++k) {
            const bool joins = k > 0 && !below[k - 1].empty() && below[k - 1][0] == k &&
                               below[k - 1].size() == below[k].size() + 1 &&
                               std::equal(below[k].begin(), below[k].end(), below[k - 1].begin() + 1);
            if (!joins) node_first_.push_back(k);
            node_[k] = static_cast<int>(node_first_.size()) - 1;
        }
        const int nodes = static_cast<int>(node_first_.size());
        node_first_.push_back(n_);
        // Storage of supernode J (columns j0 .. j0 + w - 1, rows R below them): first the diagonal part by rows,
        // row s holding its s + 1 blocks, then each row of R with its w blocks.
        node_base_.assign(nodes + 1, 0);
        node_rows_start_.assign(nodes + 1, 0);
        node_rows_.clear();
        int64_t base = 0;
        for (int J = 0; J < nodes; ++J) {
            const int j0 = node_first_[J], w = node_first_[J + 1] - j0;
            const std::vector<int> &R = below[j0 + w - 1];
            node_rows_.insert(node_rows_.end(), R.begin(), R.end());
            node_rows_start_[J + 1] = static_cast<int>(node_rows_.size());
            base += int64_t(w) * (w + 1) / 2 + int64_t(R.size()) * w;
            if (base > std::numeric_limits<int>::max()) throw std::length_error("BlockCholesky: factor too large");
            node_base_[J + 1] = static_cast<int>(base);
        }
        // Block offset of the start of row i (i in the supernode or below it) of supernode J.
        auto row_offset = [&](int J, int i) {
            const int j0 = node_first_[J], w = node_first_[J + 1] - j0;
            if (i < j0 + w) {
                const int s = i - j0;
                return node_base_[J] + s * (s + 1) / 2;
            }
            const int *R = node_rows_.data() + node_rows_start_[J];
            const int q = static_cast<int>(std::lower_bound(R, R + (node_rows_start_[J + 1] - node_rows_start_[J]), i) - R);
            return node_base_[J] + w * (w + 1) / 2 + q * w;
        };
        // The supernodes below column c that update it: (J, q) with c the q-th row of R_J.
        std::vector<std::vector<std::pair<int, int>>> sources(n_);
        for (int J = 0; J < nodes; ++J)
            for (int q = node_rows_start_[J]; q < node_rows_start_[J + 1]; ++q)
                sources[node_rows_[q]].push_back({J, q - node_rows_start_[J]});
        // Tasks column by column, the diagonal block first; the entries of each task in increasing column order.
        tasks_.clear();
        entries_.clear();
        column_start_.assign(n_ + 1, 0);
        std::vector<int> slot(n_, -1);
        std::vector<std::vector<Entry>> pending;
        for (int c = 0; c < n_; ++c) {
            const int J = node_[c], j0 = node_first_[J], t = c - j0;
            const int len = 1 + static_cast<int>(below[c].size());
            pending.assign(len, {});
            slot[c] = 0;
            for (int p = 1; p < len; ++p) slot[below[c][p - 1]] = p;
            for (const auto &[S, qc] : sources[c]) {
                const int ws = node_first_[S + 1] - node_first_[S];
                const int rbase = node_base_[S] + ws * (ws + 1) / 2;
                const int rows = node_rows_start_[S + 1] - node_rows_start_[S];
                const int *R = node_rows_.data() + node_rows_start_[S];
                for (int qi = qc; qi < rows; ++qi) pending[slot[R[qi]]].push_back({rbase + qi * ws, rbase + qc * ws, ws});
            }
            if (t > 0)
                for (int p = 0; p < len; ++p) {
                    const int i = p == 0 ? c : below[c][p - 1];
                    pending[p].push_back({row_offset(J, i), row_offset(J, c), t});
                }
            for (int p = 0; p < len; ++p) {
                const int i = p == 0 ? c : below[c][p - 1];
                Task task{};
                task.target = row_offset(J, i) + t;
                task.source = -1;
                task.first = static_cast<int>(entries_.size());
                entries_.insert(entries_.end(), pending[p].begin(), pending[p].end());
                if (entries_.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
                    throw std::length_error("BlockCholesky: factor too large");
                task.last = static_cast<int>(entries_.size());
                task.transposed = 0;
                tasks_.push_back(task);
            }
            slot[c] = -1;
            for (int i : below[c]) slot[i] = -1;
            column_start_[c + 1] = static_cast<int>(tasks_.size());
        }
        // Stored block s holds A(i, j), i >= j in the original order: in the new order it is block (max, min)
        // of L, transposed when the new row index is the smaller one.
        int s = 0;
        for (int j = 0; j < n_; ++j)
            for (int i : columns[j]) {
                const int pi = position[i], pj = position[j];
                const int c = std::min(pi, pj), r = std::max(pi, pj);
                const int p = r == c ? 0 : 1 + static_cast<int>(std::lower_bound(below[c].begin(), below[c].end(), r) - below[c].begin());
                Task &task = tasks_[column_start_[c] + p];
                task.source = s;
                task.transposed = pi < pj;
                ++s;
            }
        L_.assign(static_cast<size_t>(node_base_[nodes]) * d_ * d_, 0.0);
        inverse_.assign(static_cast<size_t>(n_) * d_, 0.0);
    }

    // Cholesky of the d x d diagonal block (lower triangle, column-major) in place; the reciprocals of its
    // diagonal go to `inverse`. False when a pivot is not positive or not finite. Here and in the other scalar
    // loops every multiply-add is an explicit fused one (std::fma), as in the vector kernels, so that the
    // result does not depend on whether the compiler contracts or vectorises a loop.
    static bool diagonal(double *Ljj, double *inverse, int d) {
        for (int c = 0; c < d; ++c) {
            double pivot = Ljj[c * d + c];
            for (int m = 0; m < c; ++m) pivot = std::fma(-Ljj[m * d + c], Ljj[m * d + c], pivot);
            if (!(pivot > 0) || !__builtin_isfinite(pivot)) return false;
            const double root = __builtin_sqrt(pivot);
            Ljj[c * d + c] = root;
            const double inv = 1.0 / root;
            inverse[c] = inv;
            for (int r = c + 1; r < d; ++r) {
                double v = Ljj[c * d + r];
                for (int m = 0; m < c; ++m) v = std::fma(-Ljj[m * d + r], Ljj[m * d + c], v);
                Ljj[c * d + r] = v * inv;
            }
            for (int r = 0; r < c; ++r) Ljj[c * d + r] = 0;
        }
        return true;
    }

    // Any d: one running sum per element of a block, over all its products.
    bool factor_generic(const double *values) {
        const int d = d_, dd = d * d;
        double *L = L_.data();
        std::vector<double> C(dd);
        for (int c = 0; c < n_; ++c) {
            double *Ljj = nullptr, *inverse = inverse_.data() + static_cast<size_t>(c) * d;
            for (int k = column_start_[c]; k < column_start_[c + 1]; ++k) {
                const Task &task = tasks_[k];
                if (task.source < 0) std::fill(C.begin(), C.end(), 0.0);
                else if (task.transposed) {
                    const double *block = values + static_cast<size_t>(task.source) * dd;
                    for (int cc = 0; cc < d; ++cc)
                        for (int r = 0; r < d; ++r) C[cc * d + r] = block[r * d + cc];
                } else std::copy(values + static_cast<size_t>(task.source) * dd, values + static_cast<size_t>(task.source + 1) * dd, C.begin());
                for (int e = task.first; e < task.last; ++e) {
                    const Entry &entry = entries_[e];
                    const double *A = L + static_cast<size_t>(entry.a) * dd, *B = L + static_cast<size_t>(entry.b) * dd;
                    const int cols = entry.width * d;
                    for (int kk = 0; kk < cols; ++kk)
                        for (int cc = 0; cc < d; ++cc) {
                            const double b = B[kk * d + cc];
                            for (int r = 0; r < d; ++r) C[cc * d + r] = std::fma(-A[kk * d + r], b, C[cc * d + r]);
                        }
                }
                double *T = L + static_cast<size_t>(task.target) * dd;
                if (k == column_start_[c]) {
                    std::copy(C.begin(), C.end(), T);
                    if (!diagonal(T, inverse, d)) return false;
                    Ljj = T;
                } else {
                    // L(i, j) = C L(j, j)^-T, column by column of the block.
                    for (int m = 0; m < d; ++m) {
                        for (int r = 0; r < d; ++r) C[m * d + r] *= inverse[m];
                        for (int cc = m + 1; cc < d; ++cc) {
                            const double l = Ljj[m * d + cc];
                            for (int r = 0; r < d; ++r) C[cc * d + r] = std::fma(-C[m * d + r], l, C[cc * d + r]);
                        }
                    }
                    std::copy(C.begin(), C.end(), T);
                }
            }
        }
        return true;
    }

#if STITCHCORE_CHOL_NEON
    // Register tile of a d x d block: rows [h R, h R + R) of all d columns, as d x R/2 pairs of rows; for
    // d <= 4 two copies, one for the even and one for the odd terms of each product, to keep enough
    // independent chains of fused multiply-adds in flight.
    template <int D>
    struct Tile {
        static constexpr int R = D == 8 ? 4 : D;  // rows in a tile
        static constexpr int V = R / 2;           // pairs of rows
        static constexpr int H = D / R;           // tiles in a block
        static constexpr int S = D <= 4 ? 2 : 1;  // copies
        float64x2_t v[S][D][V];
    };

    template <int D>
    static inline __attribute__((always_inline)) void tile_load(Tile<D> &tile, const double *block, int h, bool transposed) {
        constexpr int R = Tile<D>::R, V = Tile<D>::V, S = Tile<D>::S;
        if (!transposed) {
#pragma clang loop unroll(full)
            for (int c = 0; c < D; ++c)
#pragma clang loop unroll(full)
                for (int v = 0; v < V; ++v) tile.v[0][c][v] = vld1q_f64(block + c * D + h * R + 2 * v);
        } else {
            // Element (r, c) is block[r * D + c]: rows r, r + 1 of columns c, c + 1 are two pairs of a column of `block`.
#pragma clang loop unroll(full)
            for (int u = 0; u < D / 2; ++u)
#pragma clang loop unroll(full)
                for (int v = 0; v < V; ++v) {
                    const int r = h * R + 2 * v;
                    const float64x2_t x = vld1q_f64(block + r * D + 2 * u), y = vld1q_f64(block + (r + 1) * D + 2 * u);
                    tile.v[0][2 * u][v] = vzip1q_f64(x, y);
                    tile.v[0][2 * u + 1][v] = vzip2q_f64(x, y);
                }
        }
        if (S > 1)
#pragma clang loop unroll(full)
            for (int c = 0; c < D; ++c)
#pragma clang loop unroll(full)
                for (int v = 0; v < V; ++v) tile.v[S - 1][c][v] = vdupq_n_f64(0.0);
    }

    template <int D>
    static inline __attribute__((always_inline)) void tile_zero(Tile<D> &tile) {
#pragma clang loop unroll(full)
        for (int s = 0; s < Tile<D>::S; ++s)
#pragma clang loop unroll(full)
            for (int c = 0; c < D; ++c)
#pragma clang loop unroll(full)
                for (int v = 0; v < Tile<D>::V; ++v) tile.v[s][c][v] = vdupq_n_f64(0.0);
    }

    // tile -= A B^T over `cols` columns, A and B d x cols column-major, A already offset to the tile's rows.
    template <int D>
    static inline __attribute__((always_inline)) void tile_update(Tile<D> &tile, const double *A, const double *B, int cols) {
        constexpr int V = Tile<D>::V, S = Tile<D>::S;
        // cols is a multiple of D: one block of columns per iteration.
        for (int k0 = 0; k0 < cols; k0 += D)
#pragma clang loop unroll(full)
        for (int k = k0; k < k0 + D; k += S) {
#pragma clang loop unroll(full)
            for (int s = 0; s < S; ++s) {
                const double *a = A + (k + s) * D, *b = B + (k + s) * D;
                float64x2_t av[V], bv[D / 2];
#pragma clang loop unroll(full)
                for (int v = 0; v < V; ++v) av[v] = vld1q_f64(a + 2 * v);
#pragma clang loop unroll(full)
                for (int u = 0; u < D / 2; ++u) bv[u] = vld1q_f64(b + 2 * u);
#pragma clang loop unroll(full)
                for (int u = 0; u < D / 2; ++u)
#pragma clang loop unroll(full)
                    for (int v = 0; v < V; ++v) {
                        tile.v[s][2 * u][v] = vfmsq_laneq_f64(tile.v[s][2 * u][v], av[v], bv[u], 0);
                        tile.v[s][2 * u + 1][v] = vfmsq_laneq_f64(tile.v[s][2 * u + 1][v], av[v], bv[u], 1);
                    }
            }
        }
    }

    template <int D>
    static inline __attribute__((always_inline)) void tile_merge(Tile<D> &tile) {
        if (Tile<D>::S > 1)
#pragma clang loop unroll(full)
            for (int c = 0; c < D; ++c)
#pragma clang loop unroll(full)
                for (int v = 0; v < Tile<D>::V; ++v) tile.v[0][c][v] = vaddq_f64(tile.v[0][c][v], tile.v[Tile<D>::S - 1][c][v]);
    }

    // tile = tile L(j, j)^-T: column m is scaled by 1 / L(m, m), then taken from the columns after it.
    template <int D>
    static inline __attribute__((always_inline)) void tile_solve(Tile<D> &tile, const double *Ljj, const double *inverse) {
        constexpr int V = Tile<D>::V;
#pragma clang loop unroll(full)
        for (int m = 0; m < D; ++m) {
            const double inv = inverse[m];
#pragma clang loop unroll(full)
            for (int v = 0; v < V; ++v) tile.v[0][m][v] = vmulq_n_f64(tile.v[0][m][v], inv);
#pragma clang loop unroll(full)
            for (int u = m / 2; u < D / 2; ++u) {
                const float64x2_t l = vld1q_f64(Ljj + m * D + 2 * u);
#pragma clang loop unroll(full)
                for (int v = 0; v < V; ++v) {
                    if (2 * u > m) tile.v[0][2 * u][v] = vfmsq_laneq_f64(tile.v[0][2 * u][v], tile.v[0][m][v], l, 0);
                    if (2 * u + 1 > m) tile.v[0][2 * u + 1][v] = vfmsq_laneq_f64(tile.v[0][2 * u + 1][v], tile.v[0][m][v], l, 1);
                }
            }
        }
    }

    template <int D>
    static inline __attribute__((always_inline)) void tile_store(const Tile<D> &tile, double *block, int h) {
#pragma clang loop unroll(full)
        for (int c = 0; c < D; ++c)
#pragma clang loop unroll(full)
            for (int v = 0; v < Tile<D>::V; ++v) vst1q_f64(block + c * D + h * Tile<D>::R + 2 * v, tile.v[0][c][v]);
    }

    // Each tile of a block starts from its rows of A; the product of each entry is then summed from zero in its
    // own tile and subtracted from the block in memory (the diagonal block in place, the others in `sum`, before
    // the division by L(j, j)^T).
    template <int D>
    bool factor_neon(const double *values) {
        constexpr int DD = D * D, R = Tile<D>::R, H = Tile<D>::H;
        double *L = L_.data();
        const Task *tasks = tasks_.data();
        const Entry *entries = entries_.data();
        for (int c = 0; c < n_; ++c) {
            double *inverse = inverse_.data() + static_cast<size_t>(c) * D;
            const Task *task = tasks + column_start_[c], *end = tasks + column_start_[c + 1];
            double *Ljj = L + static_cast<size_t>(task->target) * DD;
            for (int h = 0; h < H; ++h) {
                Tile<D> tile;
                if (task->source >= 0) tile_load<D>(tile, values + static_cast<size_t>(task->source) * DD, h, false);
                else tile_zero<D>(tile);
                tile_merge<D>(tile);
                tile_store<D>(tile, Ljj, h);
                for (int e = task->first; e < task->last; ++e) {
                    Tile<D> part;
                    tile_zero<D>(part);
                    tile_update<D>(part, L + static_cast<size_t>(entries[e].a) * DD + h * R, L + static_cast<size_t>(entries[e].b) * DD, entries[e].width * D);
                    tile_merge<D>(part);
#pragma clang loop unroll(full)
                    for (int col = 0; col < D; ++col)
#pragma clang loop unroll(full)
                        for (int v = 0; v < Tile<D>::V; ++v) {
                            double *m = Ljj + col * D + h * R + 2 * v;
                            vst1q_f64(m, vaddq_f64(vld1q_f64(m), part.v[0][col][v]));
                        }
                }
            }
            if (!diagonal(Ljj, inverse, D)) return false;
            for (++task; task < end; ++task) {
                double *T = L + static_cast<size_t>(task->target) * DD;
                for (int h = 0; h < H; ++h) {
                    Tile<D> tile;
                    if (task->source >= 0) tile_load<D>(tile, values + static_cast<size_t>(task->source) * DD, h, task->transposed);
                    else tile_zero<D>(tile);
                    alignas(16) double sum[DD];
                    tile_merge<D>(tile);
                    tile_store<D>(tile, sum, h);
                    for (int e = task->first; e < task->last; ++e) {
                        Tile<D> part;
                        tile_zero<D>(part);
                        tile_update<D>(part, L + static_cast<size_t>(entries[e].a) * DD + h * R, L + static_cast<size_t>(entries[e].b) * DD, entries[e].width * D);
                        tile_merge<D>(part);
#pragma clang loop unroll(full)
                        for (int col = 0; col < D; ++col)
#pragma clang loop unroll(full)
                            for (int v = 0; v < Tile<D>::V; ++v) {
                                double *m = sum + col * D + h * R + 2 * v;
                                vst1q_f64(m, vaddq_f64(vld1q_f64(m), part.v[0][col][v]));
                            }
                    }
#pragma clang loop unroll(full)
                    for (int col = 0; col < D; ++col)
#pragma clang loop unroll(full)
                        for (int v = 0; v < Tile<D>::V; ++v) tile.v[0][col][v] = vld1q_f64(sum + col * D + h * R + 2 * v);
                    tile_merge<D>(tile);
                    tile_solve<D>(tile, Ljj, inverse);
                    tile_store<D>(tile, T, h);
                }
            }
        }
        return true;
    }
#endif

    // y_i -= P z, P d x cols column-major, cols even when D > 0.
    template <int D>
    static inline __attribute__((always_inline)) void product_subtract(double *yi, const double *P, const double *z, int cols, int d) {
#if STITCHCORE_CHOL_NEON
        if (D > 0) {
            float64x2_t even[D > 0 ? D / 2 : 1], odd[D > 0 ? D / 2 : 1];
#pragma clang loop unroll(full)
            for (int u = 0; u < D / 2; ++u) {
                even[u] = vld1q_f64(yi + 2 * u);
                odd[u] = vdupq_n_f64(0.0);
            }
            for (int k = 0; k < cols; k += 2) {
                const float64x2_t zk = vld1q_f64(z + k);
#pragma clang loop unroll(full)
                for (int u = 0; u < D / 2; ++u) {
                    even[u] = vfmsq_laneq_f64(even[u], vld1q_f64(P + k * D + 2 * u), zk, 0);
                    odd[u] = vfmsq_laneq_f64(odd[u], vld1q_f64(P + (k + 1) * D + 2 * u), zk, 1);
                }
            }
#pragma clang loop unroll(full)
            for (int u = 0; u < D / 2; ++u) vst1q_f64(yi + 2 * u, vaddq_f64(even[u], odd[u]));
            return;
        }
#endif
        for (int kk = 0; kk < cols; ++kk) {
            const double zk = z[kk];
            for (int r = 0; r < d; ++r) yi[r] = std::fma(-P[kk * d + r], zk, yi[r]);
        }
    }

    // z -= P^T y_i, P d x cols column-major, cols even when D > 0.
    template <int D>
    static inline __attribute__((always_inline)) void transposed_subtract(double *z, const double *P, const double *yi, int cols, int d) {
#if STITCHCORE_CHOL_NEON
        if (D > 0) {
            float64x2_t y[D > 0 ? D / 2 : 1];
#pragma clang loop unroll(full)
            for (int u = 0; u < D / 2; ++u) y[u] = vld1q_f64(yi + 2 * u);
            for (int k = 0; k < cols; k += 2) {
                float64x2_t acc = vld1q_f64(z + k);
#pragma clang loop unroll(full)
                for (int u = 0; u < D / 2; ++u) {
                    const float64x2_t a = vld1q_f64(P + k * D + 2 * u), b = vld1q_f64(P + (k + 1) * D + 2 * u);
                    acc = vfmsq_laneq_f64(acc, vzip1q_f64(a, b), y[u], 0);
                    acc = vfmsq_laneq_f64(acc, vzip2q_f64(a, b), y[u], 1);
                }
                vst1q_f64(z + k, acc);
            }
            return;
        }
#endif
        for (int kk = 0; kk < cols; ++kk) {
            double v = z[kk];
            for (int r = 0; r < d; ++r) v = std::fma(-P[kk * d + r], yi[r], v);
            z[kk] = v;
        }
    }

    // z_t -= sum over the rows q below a supernode of P_q^T y_q, P_q the d x d block of row q in column t of the
    // supernode (blocks `stride` doubles apart): each column of z_t is a dot product, accumulated over the rows
    // in two vectors (even and odd pairs of the block's rows), then summed.
    template <int D>
    static inline __attribute__((always_inline)) void below_transposed_subtract(double *zt, const double *P, size_t stride, const int *rows, int count,
                                                                                 const double *y) {
#if STITCHCORE_CHOL_NEON
        float64x2_t acc[D][2];
#pragma clang loop unroll(full)
        for (int c = 0; c < D; ++c) acc[c][0] = acc[c][1] = vdupq_n_f64(0.0);
        for (int q = 0; q < count; ++q, P += stride) {
            const double *yq = y + static_cast<size_t>(rows[q]) * D;
            float64x2_t yv[D / 2];
#pragma clang loop unroll(full)
            for (int u = 0; u < D / 2; ++u) yv[u] = vld1q_f64(yq + 2 * u);
#pragma clang loop unroll(full)
            for (int c = 0; c < D; ++c)
#pragma clang loop unroll(full)
                for (int u = 0; u < D / 2; ++u) acc[c][u & 1] = vfmaq_f64(acc[c][u & 1], vld1q_f64(P + c * D + 2 * u), yv[u]);
        }
#pragma clang loop unroll(full)
        for (int c = 0; c < D; c += 2)
            vst1q_f64(zt + c, vsubq_f64(vld1q_f64(zt + c), vpaddq_f64(vaddq_f64(acc[c][0], acc[c][1]), vaddq_f64(acc[c + 1][0], acc[c + 1][1]))));
#else
        for (int q = 0; q < count; ++q, P += stride) {
            const double *yq = y + static_cast<size_t>(rows[q]) * D;
            for (int c = 0; c < D; ++c) {
                double v = zt[c];
                for (int r = 0; r < D; ++r) v = std::fma(-P[c * D + r], yq[r], v);
                zt[c] = v;
            }
        }
#endif
    }

    // y_t = L(t, t)^-1 y_t, column by column.
    template <int D>
    static inline __attribute__((always_inline)) void diagonal_forward(double *yt, const double *Ltt, const double *inverse, int d) {
        if constexpr (D > 0) {
#pragma clang loop unroll(full)
            for (int c = 0; c < D; ++c) {
                const double v = yt[c] * inverse[c];
                yt[c] = v;
#pragma clang loop unroll(full)
                for (int r = c + 1; r < D; ++r) yt[r] = std::fma(-Ltt[c * D + r], v, yt[r]);
            }
        } else {
            for (int c = 0; c < d; ++c) {
                const double v = yt[c] * inverse[c];
                yt[c] = v;
                for (int r = c + 1; r < d; ++r) yt[r] = std::fma(-Ltt[c * d + r], v, yt[r]);
            }
        }
    }

    // z_t = L(t, t)^-T z_t, from the last row up.
    template <int D>
    static inline __attribute__((always_inline)) void diagonal_backward(double *zt, const double *Ltt, const double *inverse, int d) {
        if constexpr (D > 0) {
#pragma clang loop unroll(full)
            for (int c = D - 1; c >= 0; --c) {
                double v = zt[c];
#pragma clang loop unroll(full)
                for (int r = c + 1; r < D; ++r) v = std::fma(-Ltt[c * D + r], zt[r], v);
                zt[c] = v * inverse[c];
            }
        } else {
            for (int c = d - 1; c >= 0; --c) {
                double v = zt[c];
                for (int r = c + 1; r < d; ++r) v = std::fma(-Ltt[c * d + r], zt[r], v);
                zt[c] = v * inverse[c];
            }
        }
    }

    template <int D>
    void solve_d(double *x) const {
        const int d = D > 0 ? D : d_, dd = d * d;
        const double *L = L_.data();
        std::vector<double> y(static_cast<size_t>(n_) * d);
        for (int k = 0; k < n_; ++k) std::copy(x + static_cast<size_t>(perm_[k]) * d, x + static_cast<size_t>(perm_[k] + 1) * d, y.data() + static_cast<size_t>(k) * d);
        const int nodes = static_cast<int>(node_first_.size()) - 1;
        // L y = b.
        for (int J = 0; J < nodes; ++J) {
            const int j0 = node_first_[J], w = node_first_[J + 1] - j0;
            double *z = y.data() + static_cast<size_t>(j0) * d;
            for (int t = 0; t < w; ++t) {
                double *yt = z + t * d;
                const double *row = L + static_cast<size_t>(node_base_[J] + t * (t + 1) / 2) * dd;
                product_subtract<D>(yt, row, z, t * d, d);
                const double *Ltt = row + static_cast<size_t>(t) * dd, *inverse = inverse_.data() + static_cast<size_t>(j0 + t) * d;
                diagonal_forward<D>(yt, Ltt, inverse, d);
            }
            const double *rows = L + static_cast<size_t>(node_base_[J] + w * (w + 1) / 2) * dd;
            for (int q = node_rows_start_[J]; q < node_rows_start_[J + 1]; ++q, rows += static_cast<size_t>(w) * dd)
                product_subtract<D>(y.data() + static_cast<size_t>(node_rows_[q]) * d, rows, z, w * d, d);
        }
        // L^T x = y.
        for (int J = nodes - 1; J >= 0; --J) {
            const int j0 = node_first_[J], w = node_first_[J + 1] - j0;
            double *z = y.data() + static_cast<size_t>(j0) * d;
            const double *rows = L + static_cast<size_t>(node_base_[J] + w * (w + 1) / 2) * dd;
            if constexpr (D > 0) {
                for (int t = 0; t < w; ++t)
                    below_transposed_subtract<D>(z + t * d, rows + static_cast<size_t>(t) * dd, static_cast<size_t>(w) * dd, node_rows_.data() + node_rows_start_[J],
                                                 node_rows_start_[J + 1] - node_rows_start_[J], y.data());
            } else {
                for (int q = node_rows_start_[J]; q < node_rows_start_[J + 1]; ++q, rows += static_cast<size_t>(w) * dd)
                    transposed_subtract<D>(z, rows, y.data() + static_cast<size_t>(node_rows_[q]) * d, w * d, d);
            }
            for (int t = w - 1; t >= 0; --t) {
                double *zt = z + t * d;
                for (int s = t + 1; s < w; ++s)
                    transposed_subtract<D>(zt, L + static_cast<size_t>(node_base_[J] + s * (s + 1) / 2 + t) * dd, z + s * d, d, d);
                const double *Ltt = L + static_cast<size_t>(node_base_[J] + t * (t + 1) / 2 + t) * dd, *inverse = inverse_.data() + static_cast<size_t>(j0 + t) * d;
                diagonal_backward<D>(zt, Ltt, inverse, d);
            }
        }
        for (int k = 0; k < n_; ++k) std::copy(y.data() + static_cast<size_t>(k) * d, y.data() + static_cast<size_t>(k + 1) * d, x + static_cast<size_t>(perm_[k]) * d);
    }

    int n_, d_;
    bool factored_ = false;                  // the last factor() succeeded
    std::vector<int> perm_;                  // perm_[k]: the original block eliminated k-th
    std::vector<std::vector<int>> pattern_;  // during the analysis only
    std::vector<int> node_, node_first_;     // supernode of each column; first column of each supernode
    std::vector<int> node_base_;             // block offset of each supernode in L_
    std::vector<int> node_rows_start_, node_rows_;  // rows below each supernode
    std::vector<int> column_start_;          // tasks of column c: column_start_[c] .. column_start_[c + 1]
    std::vector<Task> tasks_;
    std::vector<Entry> entries_;
    std::vector<double> L_;
    std::vector<double> inverse_;  // reciprocals of the diagonal of L
};

}  // namespace stitchcore
