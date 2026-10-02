#include "stitchcore.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>

namespace {

int minimum_sample(sc_model model) {
    switch (model) {
        case SC_MODEL_TRANSLATION: return 1;
        case SC_MODEL_SIMILARITY: return 2;
        case SC_MODEL_AFFINE: return 3;
        case SC_MODEL_HOMOGRAPHY: return 4;
    }
    return 4;
}

cv::UsacParams magsac_params(double threshold, int max_iterations, double confidence, int seed) {
    cv::UsacParams params;
    params.threshold = threshold;
    params.confidence = confidence;
    params.maxIterations = max_iterations;
    params.randomGeneratorState = seed;
    params.isParallel = false;
    params.sampler = cv::SAMPLING_UNIFORM;
    params.score = cv::SCORE_METHOD_MAGSAC;
    params.loMethod = cv::LOCAL_OPTIM_SIGMA;
    params.loIterations = 10;
    params.loSampleSize = 75;
    params.final_polisher = cv::MAGSAC;
    params.final_polisher_iterations = 10;
    params.neighborsSearch = cv::NEIGH_GRID;
    return params;
}

// Stores a 2x3 or 3x3 CV_64F matrix as a row-major 3x3 array.
void store_transform(const cv::Mat &m, double out[9]) {
    for (int i = 0; i < 9; ++i) {
        out[i] = (i == 0 || i == 4 || i == 8) ? 1.0 : 0.0;
    }
    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[r * 3 + c] = m.at<double>(r, c);
        }
    }
}

}  // namespace

extern "C" sc_fit sc_fit_model(const float *points_a, const float *points_b, int32_t count, sc_model model,
                               double threshold, int32_t max_iterations, double confidence, int32_t seed,
                               uint8_t *inlier_mask) {
    sc_fit fit{};
    if (points_a == nullptr || points_b == nullptr || inlier_mask == nullptr || count < 1) {
        return fit;
    }
    std::fill(inlier_mask, inlier_mask + count, uint8_t{0});
    if (count < minimum_sample(model) || !(threshold > 0) || max_iterations < 1 || !(confidence > 0 && confidence < 1)) {
        return fit;
    }
    try {
        // Non-finite correspondences are dropped before the fit; `index` maps back to the caller's order.
        std::vector<cv::Point2f> a, b;
        std::vector<int> index;
        a.reserve(count);
        b.reserve(count);
        index.reserve(count);
        for (int i = 0; i < count; ++i) {
            const float ax = points_a[2 * i], ay = points_a[2 * i + 1], bx = points_b[2 * i], by = points_b[2 * i + 1];
            if (!std::isfinite(ax) || !std::isfinite(ay) || !std::isfinite(bx) || !std::isfinite(by)) continue;
            a.push_back({ax, ay});
            b.push_back({bx, by});
            index.push_back(i);
        }
        const int valid = static_cast<int>(a.size());
        if (valid < minimum_sample(model)) {
            return fit;
        }
        cv::setRNGSeed(seed);
        std::vector<uint8_t> mask;
        cv::Mat transform;

        switch (model) {
            case SC_MODEL_TRANSLATION: {
                cv::Vec2d t = cv::estimateTranslation2D(a, b, mask, cv::RANSAC, threshold,
                                                        static_cast<size_t>(max_iterations), confidence, 0);
                if (std::isfinite(t[0]) && std::isfinite(t[1]) && !mask.empty()) {
                    transform = cv::Mat::eye(2, 3, CV_64F);
                    transform.at<double>(0, 2) = t[0];
                    transform.at<double>(1, 2) = t[1];
                }
                break;
            }
            case SC_MODEL_SIMILARITY:
                transform = cv::estimateAffinePartial2D(a, b, mask, cv::RANSAC, threshold,
                                                        static_cast<size_t>(max_iterations), confidence, 10);
                break;
            case SC_MODEL_AFFINE:
                transform = cv::estimateAffine2D(a, b, mask, magsac_params(threshold, max_iterations, confidence, seed));
                break;
            case SC_MODEL_HOMOGRAPHY:
                transform = cv::findHomography(a, b, mask, magsac_params(threshold, max_iterations, confidence, seed));
                break;
        }
        if (transform.empty() || mask.size() != static_cast<size_t>(valid)) {
            return fit;
        }
        transform.convertTo(transform, CV_64F);
        store_transform(transform, fit.transform);
        for (double value : fit.transform) {
            if (!std::isfinite(value)) {
                return sc_fit{};
            }
        }

        // Residuals over inliers, measured with the final model.
        std::vector<double> errors;
        errors.reserve(valid);
        const double *h = fit.transform;
        for (int i = 0; i < valid; ++i) {
            if (!mask[i]) {
                continue;
            }
            const double x = a[i].x, y = a[i].y;
            const double w = h[6] * x + h[7] * y + h[8];
            if (std::fabs(w) < 1e-12) {
                continue;
            }
            const double px = (h[0] * x + h[1] * y + h[2]) / w;
            const double py = (h[3] * x + h[4] * y + h[5]) / w;
            const double error = std::hypot(px - b[i].x, py - b[i].y);
            if (!std::isfinite(error)) {
                continue;
            }
            errors.push_back(error);
            inlier_mask[index[i]] = 1;
        }
        if (errors.empty()) {
            std::fill(inlier_mask, inlier_mask + count, uint8_t{0});
            return sc_fit{};
        }
        double sum_squares = 0;
        for (double e : errors) {
            sum_squares += e * e;
        }
        std::nth_element(errors.begin(), errors.begin() + errors.size() / 2, errors.end());
        fit.ok = 1;
        fit.inlier_count = static_cast<int32_t>(errors.size());
        fit.rmse = std::sqrt(sum_squares / errors.size());
        fit.median_error = errors[errors.size() / 2];
        return fit;
    } catch (const std::exception &) {
        std::fill(inlier_mask, inlier_mask + count, uint8_t{0});
        return sc_fit{};
    }
}
