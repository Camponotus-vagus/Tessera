// RaCo's boundary-ranked keypoint selection without ONNX Runtime: the step between the Core ML maps
// of the levels extractor and ALIKED's descriptor head.
//
// It reproduces tools/export/split_extractor.py Sparse.keypoints, which mirrors RankerMode.boundary
// (what RankerMode.auto picks for 1024 <= K < 2560 when the short side is at least 768):
//   1. 3x3 non-maximum suppression on the logits (max_pool2d with -inf padding);
//   2. the exact top K + 256 local maxima by value. RaCoALIKED trims its pool of min(2K, 3840)
//      candidates to the end of the re-ranking window, since later ones cannot reach the output;
//   3. sub-pixel offsets: softmax of the raw 3x3 neighbourhood divided by the temperature (0.5),
//      neighbours outside the image counting as 0 (RaCo's _gather_subpixel_offsets);
//   4. the 512 candidates from K - 256 on are sampled on the ranker map (grid_sample, bilinear,
//      align_corners, border padding) and the best 256 of them, best first, take the last places;
//   5. a shift of half a pixel.
//
// Where the ONNX model leaves the order unspecified or produces NaN, this file decides:
//   - equal logits are ordered by linear index (the chunked top-k of the export returns ties in
//     whatever order ONNX Runtime's unsorted TopK leaves them), equal ranker scores by window index
//     (as ONNX Runtime's TopK does);
//   - NaN logits are never local maxima and do not suppress their neighbours. With fewer than K + 256
//     finite local maxima the pool goes on with the remaining pixels in index order, as a top-k over
//     the -inf of the suppressed map would;
//   - in the sub-pixel softmax a NaN neighbour weighs nothing, +inf neighbours share all the weight,
//     and a neighbourhood that is all -inf gives no offset;
//   - NaN ranker scores rank last.

#include "stitchcore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include <dispatch/dispatch.h>

#include "common.hpp"

using stitchcore::write_error;

namespace {

constexpr int64_t kReranked = 256;          // RaCo.boundary_reranked_count
constexpr int64_t kWindow = 512;            // RaCo.boundary_window_count
constexpr float kInverseTemperature = 2.0f;  // 1 / subpixel_temperature
constexpr float kInfinity = std::numeric_limits<float>::infinity();

struct Candidate {
    float value;
    int32_t index;
};

// Higher logit first, then lower linear index. Values are never NaN here.
inline bool before(const Candidate &a, const Candidate &b) {
    return a.value > b.value || (a.value == b.value && a.index < b.index);
}

struct Scored {
    float score;
    int32_t index;
};

// Higher ranker score first, NaN last, then lower window index.
inline bool ranks_before(const Scored &a, const Scored &b) {
    const bool a_nan = std::isnan(a.score), b_nan = std::isnan(b.score);
    if (a_nan != b_nan) return b_nan;
    if (!a_nan && a.score != b.score) return a.score > b.score;
    return a.index < b.index;
}

// Maximum over each pixel and its left and right neighbours inside the row; fmax skips NaN.
void row_maximum(const float *row, int64_t width, float *out) {
    if (width == 1) {
        out[0] = row[0];
        return;
    }
    out[0] = std::fmax(row[0], row[1]);
    for (int64_t x = 1; x + 1 < width; ++x) out[x] = std::fmax(std::fmax(row[x - 1], row[x]), row[x + 1]);
    out[width - 1] = std::fmax(row[width - 2], row[width - 1]);
}

// Local maxima of rows [first, last): pixels equal to the maximum of their 3x3 neighbourhood inside
// the image, NaN and -inf excluded. Writes them in index order to `out` and returns their number.
int64_t local_maxima(const float *logits, int64_t width, int64_t height, int64_t first, int64_t last,
                     float *rows, Candidate *out) {
    float *above = rows, *current = rows + width, *below = rows + 2 * width;
    if (first > 0) row_maximum(logits + (first - 1) * width, width, above);
    row_maximum(logits + first * width, width, current);
    int64_t count = 0;
    for (int64_t y = first; y < last; ++y) {
        const bool has_above = y > 0, has_below = y + 1 < height;
        if (has_below) row_maximum(logits + (y + 1) * width, width, below);
        const float *row = logits + y * width;
        for (int64_t x = 0; x < width; ++x) {
            float maximum = current[x];
            if (has_above) maximum = std::fmax(maximum, above[x]);
            if (has_below) maximum = std::fmax(maximum, below[x]);
            const float value = row[x];
            if (value == maximum && value > -kInfinity) out[count++] = {value, static_cast<int32_t>(y * width + x)};
        }
        // above <- current, current <- below; the old above row is reused for the next below.
        std::swap(above, current);
        std::swap(current, below);
    }
    return count;
}

// Offset of the candidate at pixel (cx, cy), in the (row, column) order of RaCo's meshgrid.
void subpixel_offset(const float *logits, int64_t width, int64_t height, int64_t cx, int64_t cy, float *dx,
                     float *dy) {
    float values[9];
    float maximum = -kInfinity;
    for (int k = 0; k < 9; ++k) {
        const int64_t x = cx + k % 3 - 1, y = cy + k / 3 - 1;
        float value = x >= 0 && x < width && y >= 0 && y < height ? logits[y * width + x] : 0.0f;
        if (std::isnan(value)) value = -kInfinity;
        values[k] = value;
        maximum = std::max(maximum, value);
    }
    float weights[9];
    float sum = 0;
    for (int k = 0; k < 9; ++k) {
        if (maximum == kInfinity) {
            weights[k] = values[k] == kInfinity ? 1.0f : 0.0f;
        } else if (maximum == -kInfinity) {
            weights[k] = k == 4 ? 1.0f : 0.0f;
        } else {
            // (v - max) / T equals v / T - max / T exactly, since 1 / T is a power of two.
            weights[k] = std::exp((values[k] - maximum) * kInverseTemperature);
        }
        sum += weights[k];
    }
    // Scaling by the reciprocal of the sum agrees with the last bits of the ONNX model more often
    // than dividing by it.
    float x = 0, y = 0;
    const float inverse = 1.0f / sum;
    for (int k = 0; k < 9; ++k) {
        const float probability = weights[k] * inverse;
        x += probability * static_cast<float>(k % 3 - 1);
        y += probability * static_cast<float>(k / 3 - 1);
    }
    *dx = x;
    *dy = y;
}

// The ranker at pixel (x, y) as grid_sample computes it (bilinear, align_corners, border padding).
float sample_ranker(const float *ranker, int64_t width, int64_t height, float x, float y) {
    // The network normalises to [-1, 1] and grid_sample maps back; the round trip is kept so the
    // weights round the same way.
    float px = 0, py = 0;
    if (width > 1) {
        const float scale = static_cast<float>(width - 1);
        const float normalised = 2 * x / scale - 1;
        px = (normalised + 1) / 2 * scale;
    }
    if (height > 1) {
        const float scale = static_cast<float>(height - 1);
        const float normalised = 2 * y / scale - 1;
        py = (normalised + 1) / 2 * scale;
    }
    px = std::isnan(px) ? 0 : std::clamp(px, 0.0f, static_cast<float>(width - 1));
    py = std::isnan(py) ? 0 : std::clamp(py, 0.0f, static_cast<float>(height - 1));
    const int64_t x0 = static_cast<int64_t>(std::floor(px)), y0 = static_cast<int64_t>(std::floor(py));
    const int64_t x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const float wx1 = px - static_cast<float>(x0), wx0 = static_cast<float>(x0 + 1) - px;
    const float wy1 = py - static_cast<float>(y0), wy0 = static_cast<float>(y0 + 1) - py;
    const float *top = ranker + y0 * width, *bottom = ranker + y1 * width;
    return wy0 * (wx0 * top[x0] + wx1 * top[x1]) + wy1 * (wx0 * bottom[x0] + wx1 * bottom[x1]);
}

}  // namespace

extern "C" int32_t sc_select_keypoints_native(const float *logits, const float *ranker, int32_t width,
                                              int32_t height, int32_t keypoints, float *out, char *error,
                                              size_t error_length) {
    if (logits == nullptr || ranker == nullptr || out == nullptr) {
        write_error(error, error_length, "missing argument");
        return 1;
    }
    if (width < 1 || height < 1) {
        write_error(error, error_length, "invalid map size");
        return 1;
    }
    const int64_t W = width, H = height, pixels = W * H;
    if (pixels > std::numeric_limits<int32_t>::max()) {
        write_error(error, error_length, "map too large");
        return 1;
    }
    if (keypoints < kReranked) {
        write_error(error, error_length, "keypoint selection needs at least 256 keypoints");
        return 1;
    }
    const int64_t K = keypoints, pool = K + kWindow - kReranked, window_start = K - kReranked;
    if (pool > pixels) {
        write_error(error, error_length, "too many keypoints for the map size");
        return 1;
    }
    try {
        // 1. Local maxima per band of rows, in parallel. Each band writes into its own slice of one
        // buffer large enough for every pixel; only the touched pages are ever backed by memory.
        const int64_t band_height = (H + 31) / 32;
        const int64_t band_count = (H + band_height - 1) / band_height;
        std::unique_ptr<Candidate[]> storage(new Candidate[pixels]);
        std::unique_ptr<float[]> scratch(new float[band_count * 3 * W]);
        std::vector<int64_t> counts(band_count, 0);
        Candidate *slices = storage.get();
        float *rows = scratch.get();
        int64_t *band_counts = counts.data();
        dispatch_apply(static_cast<size_t>(band_count), DISPATCH_APPLY_AUTO, ^(size_t band) {
            const int64_t first = static_cast<int64_t>(band) * band_height, last = std::min(H, first + band_height);
            Candidate *slice = slices + first * W;
            int64_t count = local_maxima(logits, W, H, first, last, rows + static_cast<int64_t>(band) * 3 * W, slice);
            if (count > pool) {
                std::nth_element(slice, slice + pool, slice + count, before);
                count = pool;
            }
            band_counts[band] = count;
        });

        // 2. The exact global top `pool`, best first.
        std::vector<Candidate> top;
        top.reserve(static_cast<size_t>(std::min(pixels, band_count * pool)));
        for (int64_t band = 0; band < band_count; ++band) {
            const Candidate *slice = slices + band * band_height * W;
            top.insert(top.end(), slice, slice + counts[band]);
        }
        if (static_cast<int64_t>(top.size()) > pool) {
            std::nth_element(top.begin(), top.begin() + pool, top.end(), before);
            top.resize(static_cast<size_t>(pool));
        }
        std::sort(top.begin(), top.end(), before);
        if (static_cast<int64_t>(top.size()) < pool) {
            std::vector<int32_t> taken;
            taken.reserve(top.size());
            for (const Candidate &candidate : top) taken.push_back(candidate.index);
            std::sort(taken.begin(), taken.end());
            size_t next_taken = 0;
            for (int32_t index = 0; static_cast<int64_t>(top.size()) < pool; ++index) {
                if (next_taken < taken.size() && taken[next_taken] == index) {
                    ++next_taken;
                } else {
                    top.push_back({-kInfinity, index});
                }
            }
        }

        // 3. Integer positions plus sub-pixel offsets.
        std::vector<float> points(static_cast<size_t>(pool) * 2);
        for (int64_t n = 0; n < pool; ++n) {
            const int64_t x = top[n].index % W, y = top[n].index / W;
            float dx, dy;
            subpixel_offset(logits, W, H, x, y, &dx, &dy);
            points[2 * n] = static_cast<float>(x) + dx;
            points[2 * n + 1] = static_cast<float>(y) + dy;
        }

        // 4. Re-rank the window around the cut by the ranker.
        std::vector<Scored> window(static_cast<size_t>(kWindow));
        for (int64_t w = 0; w < kWindow; ++w) {
            const float *point = &points[2 * (window_start + w)];
            window[w] = {sample_ranker(ranker, W, H, point[0], point[1]), static_cast<int32_t>(w)};
        }
        std::partial_sort(window.begin(), window.begin() + kReranked, window.end(), ranks_before);

        // 5. The detector order up to the window, then the re-ranked points, shifted by half a pixel.
        for (int64_t n = 0; n < K; ++n) {
            const int64_t source = n < window_start ? n : window_start + window[n - window_start].index;
            out[2 * n] = points[2 * source] + 0.5f;
            out[2 * n + 1] = points[2 * source + 1] + 0.5f;
        }
        return 0;
    } catch (const std::bad_alloc &) {
        write_error(error, error_length, "out of memory");
        return 1;
    }
}
