// ALIKED's sparse deformable descriptor head, evaluated only at the pixels it reads.
//
// The network's dense part would upsample four feature levels to full resolution, concatenate
// them (128 channels) and L2-normalise every pixel: 400 MB per photo of which a few hundred
// thousand values are used. Here each needed pixel is rebuilt on the fly with the same bilinear
// (align_corners) upsampling and normalisation, and the head's convolutions run as GEMMs.

#include "stitchcore.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <dispatch/dispatch.h>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include "common.hpp"

using stitchcore::write_error;

struct sc_descriptor_head {
    int channels = 0;    // C, concatenated feature channels (128)
    int positions = 0;   // P, deformable sampling positions (16)
    int kernel = 0;      // patch size (3)
    int dimensions = 0;  // D, descriptor length (128)
    std::vector<float> offset_weight;  // [2P, C * kernel * kernel]
    std::vector<float> offset_bias;    // [2P]
    std::vector<float> second_weight;  // [2P, 2P]
    std::vector<float> second_bias;    // [2P]
    std::vector<float> sample_weight;  // [C, C]
    std::vector<float> aggregation;    // [P * C, D]
};

namespace {

constexpr float kSeluAlpha = 1.6732632423543772f;
constexpr float kSeluScale = 1.0507009873554805f;

inline float selu(float x) { return kSeluScale * (x > 0 ? x : kSeluAlpha * std::expm1(x)); }

struct Level {
    const float *data;
    int width;
    int height;
};

// Feature vector of the full-resolution map at integer pixel (x, y): every level upsampled
// bilinearly with align_corners = true, concatenated, then L2-normalised.
struct FeatureMap {
    std::vector<Level> levels;
    int level_channels;
    int width;
    int height;

    void at(int x, int y, float *out) const {
        float *cursor = out;
        for (const Level &level : levels) {
            const size_t plane = static_cast<size_t>(level.width) * level.height;
            if (level.width == width && level.height == height) {
                const float *base = level.data + static_cast<size_t>(y) * width + x;
                for (int c = 0; c < level_channels; ++c) cursor[c] = base[c * plane];
            } else {
                const float sx = static_cast<float>(level.width - 1) / static_cast<float>(width - 1);
                const float sy = static_cast<float>(level.height - 1) / static_cast<float>(height - 1);
                const float fx = sx * static_cast<float>(x), fy = sy * static_cast<float>(y);
                const int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
                const int x1 = x0 < level.width - 1 ? x0 + 1 : x0;
                const int y1 = y0 < level.height - 1 ? y0 + 1 : y0;
                const float ax = fx - static_cast<float>(x0), ay = fy - static_cast<float>(y0);
                const float w00 = (1 - ay) * (1 - ax), w01 = (1 - ay) * ax, w10 = ay * (1 - ax), w11 = ay * ax;
                const size_t i00 = static_cast<size_t>(y0) * level.width + x0;
                const size_t i01 = static_cast<size_t>(y0) * level.width + x1;
                const size_t i10 = static_cast<size_t>(y1) * level.width + x0;
                const size_t i11 = static_cast<size_t>(y1) * level.width + x1;
                for (int c = 0; c < level_channels; ++c) {
                    const float *p = level.data + c * plane;
                    cursor[c] = w00 * p[i00] + w01 * p[i01] + w10 * p[i10] + w11 * p[i11];
                }
            }
            cursor += level_channels;
        }
        const int total = level_channels * static_cast<int>(levels.size());
        float norm = 0;
        // std::fma: the fused product -Os gives; vectorised at -O2 and above, the loop split it into a
        // product and an ordered sum, which changed the last bit of the descriptors.
        for (int c = 0; c < total; ++c) norm = std::fma(out[c], out[c], norm);
        const float inverse = 1.0f / std::max(std::sqrt(norm), 1e-12f);
        for (int c = 0; c < total; ++c) out[c] *= inverse;
    }
};

bool read_floats(std::ifstream &file, std::vector<float> &target, size_t count) {
    target.resize(count);
    file.read(reinterpret_cast<char *>(target.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return static_cast<size_t>(file.gcount()) == count * sizeof(float);
}

// C = A * B^T with A [m, k] and B [n, k], row-major.
void gemm_transposed(const float *a, const float *b, float *c, int m, int n, int k) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a, k, b, k, 0.0f, c, n);
}

}  // namespace

extern "C" sc_descriptor_head *sc_descriptor_head_load(const char *path, char *error, size_t error_length) {
    if (path == nullptr) {
        write_error(error, error_length, "missing path");
        return nullptr;
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        write_error(error, error_length, "cannot open the descriptor head file");
        return nullptr;
    }
    const std::streamoff file_size = file.tellg();
    file.seekg(0);
    char magic[4] = {};
    int32_t sizes[4] = {};
    if (!file.read(magic, 4) || std::string(magic, 4) != "ALKH" ||
        !file.read(reinterpret_cast<char *>(sizes), sizeof(sizes))) {
        write_error(error, error_length, "not a descriptor head file");
        return nullptr;
    }
    // Bounds well above the real network (128, 16, 3, 128) but small enough to rule out garbage headers.
    if (sizes[0] < 1 || sizes[0] > 4096 || sizes[1] < 1 || sizes[1] > 256 || sizes[2] < 1 || sizes[2] > 15 ||
        sizes[2] % 2 == 0 || sizes[3] < 1 || sizes[3] > 4096) {
        write_error(error, error_length, "descriptor head file has an invalid header");
        return nullptr;
    }
    {
        const int64_t c = sizes[0], p = sizes[1], k = sizes[2], d = sizes[3];
        const int64_t floats = 2 * p * c * k * k + 2 * p + 4 * p * p + 2 * p + c * c + p * c * d;  // the six tensors below, in order
        if (file_size != static_cast<std::streamoff>(4 + 16 + 4 * floats)) {
            write_error(error, error_length, "descriptor head file has the wrong size");
            return nullptr;
        }
    }
    sc_descriptor_head *head = nullptr;
    try {
        head = new sc_descriptor_head();
    } catch (const std::exception &) {
        write_error(error, error_length, "out of memory");
        return nullptr;
    }
    head->channels = sizes[0];
    head->positions = sizes[1];
    head->kernel = sizes[2];
    head->dimensions = sizes[3];
    const size_t offsets = 2 * static_cast<size_t>(head->positions);
    const size_t patch = static_cast<size_t>(head->channels) * head->kernel * head->kernel;
    bool ok = false;
    try {
        ok = read_floats(file, head->offset_weight, offsets * patch) &&
             read_floats(file, head->offset_bias, offsets) &&
             read_floats(file, head->second_weight, offsets * offsets) &&
             read_floats(file, head->second_bias, offsets) &&
             read_floats(file, head->sample_weight, static_cast<size_t>(head->channels) * head->channels) &&
             read_floats(file, head->aggregation,
                         static_cast<size_t>(head->positions) * head->channels * head->dimensions);
    } catch (const std::exception &) {
        ok = false;
    }
    if (!ok) {
        delete head;
        write_error(error, error_length, "descriptor head file is truncated");
        return nullptr;
    }
    return head;
}

extern "C" void sc_descriptor_head_free(sc_descriptor_head *head) { delete head; }

extern "C" int32_t sc_descriptor_head_dimensions(const sc_descriptor_head *head) {
    return head ? head->dimensions : 0;
}

extern "C" int32_t sc_describe(const sc_descriptor_head *head, const float *const *levels, const int32_t *level_widths,
                               const int32_t *level_heights, int32_t level_count, int32_t level_channels,
                               int32_t width, int32_t height, const float *keypoints, int32_t count,
                               float *descriptors, char *error, size_t error_length) {
    if (head == nullptr || levels == nullptr || level_widths == nullptr || level_heights == nullptr ||
        descriptors == nullptr || (count > 0 && keypoints == nullptr) || count < 0 ||
        level_count < 1 || level_channels < 1 || level_count * level_channels != head->channels || width <= head->kernel || height <= head->kernel) {
        write_error(error, error_length, "descriptor head does not match the feature levels");
        return 1;
    }
    FeatureMap map;
    map.level_channels = level_channels;
    map.width = width;
    map.height = height;
    for (int i = 0; i < level_count; ++i) {
        if (levels[i] == nullptr || level_widths[i] < 1 || level_heights[i] < 1 || level_widths[i] > width ||
            level_heights[i] > height) {
            write_error(error, error_length, "invalid feature level");
            return 1;
        }
        map.levels.push_back({levels[i], level_widths[i], level_heights[i]});
    }
    if (count == 0) {
        return 0;
    }

    const int C = head->channels, P = head->positions, K = head->kernel, D = head->dimensions;
    const int patch_size = C * K * K, offsets_size = 2 * P;
    const float scale_x = static_cast<float>(width - 1), scale_y = static_cast<float>(height - 1);

    try {
        // Pixel positions as the network computes them: normalise to [-1, 1] and back.
        std::vector<float> pixel(static_cast<size_t>(count) * 2);
        for (int n = 0; n < count; ++n) {
            // Keypoints come from the network inside the image; anything else is clamped so lookups stay in bounds.
            float kx = keypoints[2 * n], ky = keypoints[2 * n + 1];
            if (!std::isfinite(kx)) kx = 0;
            if (!std::isfinite(ky)) ky = 0;
            kx = std::clamp(kx, 0.0f, scale_x);
            ky = std::clamp(ky, 0.0f, scale_y);
            const float nx = 2 * kx / scale_x - 1, ny = 2 * ky / scale_y - 1;
            pixel[2 * n] = (nx / 2 + 0.5f) * scale_x;
            pixel[2 * n + 1] = (ny / 2 + 0.5f) * scale_y;
        }

        // Blocks capture C++ objects as const copies, so the parallel loops only see raw pointers. Each keypoint
        // has its own row of `scratch` for a feature vector: nothing is allocated inside the blocks.
        const FeatureMap *features = &map;
        const float *points = pixel.data();
        std::vector<float> scratch(static_cast<size_t>(count) * C);
        float *scratch_rows = scratch.data();

        // 1. Patches around each keypoint: [count, C * K * K] in (channel, row, column) order.
        std::vector<float> patches(static_cast<size_t>(count) * patch_size);
        float *patch_rows = patches.data();
        dispatch_apply(static_cast<size_t>(count), DISPATCH_APPLY_AUTO, ^(size_t n) {
            float *feature = scratch_rows + n * C;
            const int64_t px = static_cast<int64_t>(points[2 * n]), py = static_cast<int64_t>(points[2 * n + 1]);
            const int64_t corner_x = static_cast<int64_t>(static_cast<float>(px) - K / 2.0f + 1);
            const int64_t corner_y = static_cast<int64_t>(static_cast<float>(py) - K / 2.0f + 1);
            const int x0 = static_cast<int>(std::clamp<int64_t>(corner_x, 0, width - 1 - K));
            const int y0 = static_cast<int>(std::clamp<int64_t>(corner_y, 0, height - 1 - K));
            float *row = patch_rows + n * patch_size;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    features->at(x0 + kx, y0 + ky, feature);
                    for (int c = 0; c < C; ++c) row[c * K * K + ky * K + kx] = feature[c];
                }
            }
        });

        // 2. Offsets: two layers on the patches, then clamped.
        std::vector<float> hidden(static_cast<size_t>(count) * offsets_size);
        gemm_transposed(patches.data(), head->offset_weight.data(), hidden.data(), count, offsets_size, patch_size);
        for (int n = 0; n < count; ++n) {
            for (int j = 0; j < offsets_size; ++j) {
                float &value = hidden[static_cast<size_t>(n) * offsets_size + j];
                value = selu(value + head->offset_bias[j]);
            }
        }
        std::vector<float> offsets(static_cast<size_t>(count) * offsets_size);
        gemm_transposed(hidden.data(), head->second_weight.data(), offsets.data(), count, offsets_size, offsets_size);
        const float maximum = static_cast<float>(std::max(width, height)) / 4.0f;
        for (int n = 0; n < count; ++n) {
            for (int j = 0; j < offsets_size; ++j) {
                float &value = offsets[static_cast<size_t>(n) * offsets_size + j];
                value = std::clamp(value + head->second_bias[j], -maximum, maximum);
            }
        }

        // 3. Deformable samples: bilinear on the normalised map, zero outside (grid_sample, align_corners).
        std::vector<float> samples(static_cast<size_t>(count) * P * C, 0.0f);
        const float *offset_rows = offsets.data();
        float *sample_rows = samples.data();
        dispatch_apply(static_cast<size_t>(count), DISPATCH_APPLY_AUTO, ^(size_t n) {
            float *feature = scratch_rows + n * C;
            for (int p = 0; p < P; ++p) {
                const float ox = offset_rows[n * offsets_size + p], oy = offset_rows[n * offsets_size + P + p];
                const float gx = 2 * (points[2 * n] + ox) / scale_x - 1, gy = 2 * (points[2 * n + 1] + oy) / scale_y - 1;
                const float x = (gx + 1) / 2 * scale_x, y = (gy + 1) / 2 * scale_y;
                if (!std::isfinite(x) || !std::isfinite(y)) continue;
                const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
                const float ax = x - static_cast<float>(x0), ay = y - static_cast<float>(y0);
                float *target = sample_rows + (n * P + p) * C;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int xx = x0 + dx, yy = y0 + dy;
                        if (xx < 0 || yy < 0 || xx >= width || yy >= height) continue;
                        const float weight = (dx ? ax : 1 - ax) * (dy ? ay : 1 - ay);
                        features->at(xx, yy, feature);
                        for (int c = 0; c < C; ++c) target[c] += weight * feature[c];
                    }
                }
            }
        });

        // 4. Per-sample projection, SELU, aggregation over positions, L2 normalisation.
        std::vector<float> projected(samples.size());
        gemm_transposed(samples.data(), head->sample_weight.data(), projected.data(), count * P, C, C);
        for (float &value : projected) value = selu(value);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, count, D, P * C, 1.0f, projected.data(), P * C,
                    head->aggregation.data(), D, 0.0f, descriptors, D);
        for (int n = 0; n < count; ++n) {
            float *row = descriptors + static_cast<size_t>(n) * D;
            const float norm = cblas_snrm2(D, row, 1);
            const float inverse = 1.0f / std::max(norm, 1e-12f);
            for (int d = 0; d < D; ++d) row[d] *= inverse;
        }
    } catch (const std::exception &e) {
        // Allocation failures: nothing is thrown across the C API.
        write_error(error, error_length, e.what());
        return 1;
    }
    return 0;
}
