#include "stitchcore.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <opencv2/core.hpp>
#include <opencv2/features.hpp>

#include "common.hpp"

struct sc_features {
    std::vector<sc_keypoint> keypoints;
    cv::Mat descriptors;  // CV_32F, one row per keypoint
};

using stitchcore::write_error;

extern "C" void sc_free(void *pointer) { std::free(pointer); }

extern "C" sc_features *sc_sift_extract(const uint8_t *gray, int32_t width, int32_t height, int32_t bytes_per_row,
                                        int32_t max_features, int32_t root_sift, char *error, size_t error_length) {
    if (gray == nullptr || width < 1 || height < 1 || bytes_per_row < width) {
        write_error(error, error_length, "invalid image buffer");
        return nullptr;
    }
    try {
        cv::Mat image(height, width, CV_8UC1, const_cast<uint8_t *>(gray), static_cast<size_t>(bytes_per_row));
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(max_features);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        if (root_sift && !descriptors.empty()) {
            for (int row = 0; row < descriptors.rows; ++row) {
                float *values = descriptors.ptr<float>(row);
                double sum = 0;
                for (int col = 0; col < descriptors.cols; ++col) {
                    sum += std::fabs(values[col]);
                }
                const float scale = sum > 0 ? static_cast<float>(1.0 / sum) : 0.0f;
                for (int col = 0; col < descriptors.cols; ++col) {
                    values[col] = std::sqrt(std::fabs(values[col]) * scale);
                }
            }
        }

        auto *result = new sc_features();
        result->keypoints.reserve(keypoints.size());
        for (const cv::KeyPoint &k : keypoints) {
            result->keypoints.push_back({k.pt.x, k.pt.y, k.size, k.angle, k.response});
        }
        result->descriptors = descriptors;
        return result;
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return nullptr;
    }
}

extern "C" int32_t sc_features_count(const sc_features *features) {
    return features ? static_cast<int32_t>(features->keypoints.size()) : 0;
}

extern "C" const sc_keypoint *sc_features_keypoints(const sc_features *features) {
    return (features && !features->keypoints.empty()) ? features->keypoints.data() : nullptr;
}

extern "C" void sc_features_free(sc_features *features) { delete features; }

extern "C" const float *sc_features_descriptors(const sc_features *features, int32_t *size) {
    if (features == nullptr || features->descriptors.empty()) {
        if (size) *size = 0;
        return nullptr;
    }
    *size = features->descriptors.cols;
    return features->descriptors.ptr<float>(0);
}

namespace {

// Exact nearest neighbours from S = A * B^T. With squared norms |a|^2 and |b|^2,
// |a - b|^2 = |a|^2 + |b|^2 - 2 a.b. A is processed in row blocks to bound memory.
std::vector<sc_match> nearest_neighbours(const float *a, int count_a, const float *b, int count_b, int size,
                                         float ratio, bool mutual) {
    std::vector<sc_match> kept;
    if (count_a < 1 || count_b < 2 || size < 1) {
        return kept;
    }
    // A descriptor with a NaN or infinite component gets an infinite norm, so it never matches.
    const float infinity = std::numeric_limits<float>::infinity();
    std::vector<float> norm_a(count_a), norm_b(count_b);
    for (int i = 0; i < count_a; ++i) {
        norm_a[i] = cblas_sdot(size, a + static_cast<size_t>(i) * size, 1, a + static_cast<size_t>(i) * size, 1);
        if (!std::isfinite(norm_a[i])) norm_a[i] = infinity;
    }
    for (int j = 0; j < count_b; ++j) {
        norm_b[j] = cblas_sdot(size, b + static_cast<size_t>(j) * size, 1, b + static_cast<size_t>(j) * size, 1);
        if (!std::isfinite(norm_b[j])) norm_b[j] = infinity;
    }

    const int block = 512;
    std::vector<float> products(static_cast<size_t>(block) * count_b);
    std::vector<int> best(count_a, -1);
    std::vector<float> best_distance(count_a), second_distance(count_a);
    std::vector<int> column_best(count_b, -1);
    std::vector<float> column_distance(count_b, std::numeric_limits<float>::infinity());

    for (int start = 0; start < count_a; start += block) {
        const int rows = std::min(block, count_a - start);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, count_b, size, 1.0f,
                    a + static_cast<size_t>(start) * size, size, b, size, 0.0f, products.data(), count_b);
        for (int r = 0; r < rows; ++r) {
            const int i = start + r;
            const float *row = products.data() + static_cast<size_t>(r) * count_b;
            float d1 = std::numeric_limits<float>::infinity(), d2 = d1;
            int j1 = -1;
            for (int j = 0; j < count_b; ++j) {
                const float raw = norm_a[i] + norm_b[j] - 2 * row[j];
                const float d = std::isfinite(raw) ? std::max(0.0f, raw) : infinity;
                if (d < d1) {
                    d2 = d1;
                    d1 = d;
                    j1 = j;
                } else if (d < d2) {
                    d2 = d;
                }
                if (d < column_distance[j]) {
                    column_distance[j] = d;
                    column_best[j] = i;
                }
            }
            best[i] = j1;
            best_distance[i] = std::sqrt(d1);
            second_distance[i] = std::sqrt(d2);
        }
    }

    kept.reserve(count_a / 4);
    for (int i = 0; i < count_a; ++i) {
        const int j = best[i];
        if (j < 0 || !std::isfinite(best_distance[i]) || second_distance[i] <= 0 ||
            best_distance[i] >= ratio * second_distance[i]) continue;
        if (mutual && column_best[j] != i) continue;
        kept.push_back({i, j, 1.0f - best_distance[i] / second_distance[i]});
    }
    return kept;
}

int32_t publish(const std::vector<sc_match> &kept, sc_match **matches) {
    *matches = stitchcore::allocate_array<sc_match>(kept.size());
    if (!kept.empty()) {
        std::memcpy(*matches, kept.data(), kept.size() * sizeof(sc_match));
    }
    return static_cast<int32_t>(kept.size());
}

}  // namespace

extern "C" int32_t sc_match_descriptors(const sc_features *a, const sc_features *b, float ratio, int32_t mutual,
                                        sc_match **matches, char *error, size_t error_length) {
    *matches = nullptr;
    try {
        if (a == nullptr || b == nullptr || a->descriptors.empty() || b->descriptors.empty() ||
            a->descriptors.cols != b->descriptors.cols) {
            return publish({}, matches);
        }
        return publish(nearest_neighbours(a->descriptors.ptr<float>(0), a->descriptors.rows,
                                          b->descriptors.ptr<float>(0), b->descriptors.rows,
                                          a->descriptors.cols, ratio, mutual != 0),
                       matches);
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return -1;
    }
}

extern "C" int32_t sc_match_raw(const float *a, int32_t count_a, const float *b, int32_t count_b, int32_t size,
                                float ratio, int32_t mutual, sc_match **matches, char *error, size_t error_length) {
    *matches = nullptr;
    try {
        if (a == nullptr || b == nullptr) {
            return publish({}, matches);
        }
        return publish(nearest_neighbours(a, count_a, b, count_b, size, ratio, mutual != 0), matches);
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return -1;
    }
}
