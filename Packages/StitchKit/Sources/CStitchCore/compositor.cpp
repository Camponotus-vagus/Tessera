// Composites aligned photos into one image: warp, exposure gains, seams, blending, crop.
//
// Swift drives it one photo at a time so memory stays bounded: first a small 8-bit copy of every photo
// (exposure and seams are estimated on those), then each photo at the output scale, 16 bits per channel.
// OpenCV's blenders work in CV_16S and wrap around on overflow, so values are scaled to at most
// 14 bits before feeding and back afterwards.

#include "stitchcore.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/warpers.hpp>

#include "common.hpp"

using stitchcore::write_error;
namespace cd = cv::detail;

namespace stitchcore {
bool largest_rectangle(const uint8_t *mask, int width, int height, size_t stride, int32_t rect[4]);
}

struct sc_compositor {
    sc_compose_options options{};
    int count = 0;
    std::vector<cv::Size> full;
    std::vector<cv::Matx33d> transform;  // planar: image -> mosaic (full-res); rotation: R
    std::vector<double> focal;
    double warped_scale = 1;              // rotation: projection radius at full resolution
    cv::Point origin;                     // canvas top-left in warped coordinates at the output scale
    cv::Size canvas;

    std::vector<cv::Point> seam_corners;
    std::vector<cv::UMat> seam_images;    // CV_8UC3
    std::vector<cv::UMat> seam_masks;     // CV_8U, 0 or 255, updated by the seam finder
    std::vector<bool> seam_added;

    std::vector<cv::Point> corners;       // per photo, at the output scale
    std::vector<cv::Size> sizes;
    std::vector<cv::Mat> gains;
    double value_scale = 0.25;            // 16-bit value -> blender value (at most 14 bits after gains)
    cv::Ptr<cd::Blender> blender;
    cv::Mat hard, hard_mask;              // SC_BLEND_NONE: direct composite, CV_16UC3 and CV_8U
    std::vector<bool> added;
    bool prepared = false;
    std::atomic<bool> cancelled{false};
};

namespace {

bool planar(const sc_compositor &c) { return c.options.projection == SC_PROJECTION_PLANE; }

int interpolation(const sc_compositor &c) {
    switch (c.options.interpolation) {
        case 0: return cv::INTER_NEAREST;
        case 1: return cv::INTER_LINEAR;
        default: return cv::INTER_CUBIC;
    }
}

cv::Ptr<cd::RotationWarper> rotation_warper(const sc_compositor &c, double scale) {
    const float s = float(c.warped_scale * scale);
    switch (c.options.projection) {
        case SC_PROJECTION_RECTILINEAR: return cv::makePtr<cd::PlaneWarper>(s);
        case SC_PROJECTION_CYLINDRICAL: return cv::makePtr<cd::CylindricalWarper>(s);
        default: return cv::makePtr<cd::SphericalWarper>(s);
    }
}

cv::Mat camera_K(const sc_compositor &c, int i, double scale) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = float(c.focal[i] * scale);
    K.at<float>(1, 1) = float(c.focal[i] * scale);
    K.at<float>(0, 2) = float(c.full[i].width * 0.5 * scale);
    K.at<float>(1, 2) = float(c.full[i].height * 0.5 * scale);
    return K;
}

cv::Mat camera_R(const sc_compositor &c, int i) {
    cv::Mat R(3, 3, CV_32F);
    for (int k = 0; k < 9; ++k) R.at<float>(k / 3, k % 3) = float(c.transform[i].val[k]);
    return R;
}

// Planar map at `scale`: from the photo resized by `scale` to the mosaic resized by `scale`.
cv::Matx33d planar_at(const sc_compositor &c, int i, double scale) {
    const cv::Matx33d S(scale, 0, 0, 0, scale, 0, 0, 0, 1);
    return S * c.transform[i] * S.inv();
}

cv::Size scaled(const cv::Size &size, double scale) {
    return cv::Size(std::max(1, int(std::lround(size.width * scale))), std::max(1, int(std::lround(size.height * scale))));
}

// Pixels covered by a photo of `size` mapped by a planar map (the bounding box of its corner pixel centres);
// throws when a corner falls behind the plane.
cv::Rect planar_roi(const cv::Matx33d &M, const cv::Size &size) {
    double minx = INFINITY, miny = INFINITY, maxx = -INFINITY, maxy = -INFINITY;
    const cv::Point2d corners[4] = {{0, 0}, {size.width - 1.0, 0}, {size.width - 1.0, size.height - 1.0},
                                    {0, size.height - 1.0}};
    for (const cv::Point2d &p : corners) {
        const cv::Vec3d q = M * cv::Vec3d(p.x, p.y, 1);
        if (!(q[2] > 1e-9)) throw std::runtime_error("a photo maps behind the projection plane");
        minx = std::min(minx, q[0] / q[2]);
        maxx = std::max(maxx, q[0] / q[2]);
        miny = std::min(miny, q[1] / q[2]);
        maxy = std::max(maxy, q[1] / q[2]);
    }
    // A small tolerance keeps exact integer positions from picking up a neighbouring pixel.
    const cv::Point tl(int(std::floor(minx + 1e-6)), int(std::floor(miny + 1e-6)));
    const cv::Point br(int(std::ceil(maxx - 1e-6)) + 1, int(std::ceil(maxy - 1e-6)) + 1);
    return cv::Rect(tl, br);
}

cv::Rect warp_roi(const sc_compositor &c, int i, double scale) {
    const cv::Size size = scaled(c.full[i], scale);
    if (planar(c)) return planar_roi(planar_at(c, i, scale), size);
    return rotation_warper(c, scale)->warpRoi(size, camera_K(c, i, scale), camera_R(c, i));
}

// Warps `source` (any depth, 1 or 3 channels) of photo i at `scale`; returns the top-left corner.
cv::Point warp(const sc_compositor &c, int i, double scale, const cv::Mat &source, int mode, int border,
               cv::Mat &result) {
    if (!planar(c)) {
        return rotation_warper(c, scale)->warp(source, camera_K(c, i, scale), camera_R(c, i), mode, border, result);
    }
    const cv::Matx33d M = planar_at(c, i, scale);
    const cv::Rect roi = planar_roi(M, source.size());
    const cv::Matx33d shifted = cv::Matx33d(1, 0, -roi.x, 0, 1, -roi.y, 0, 0, 1) * M;
    cv::warpPerspective(source, result, cv::Mat(shifted), roi.size(), mode, border);
    return roi.tl();
}

// The valid area of photo i, warped: nearest-neighbour warp of a full mask. The colour is warped with
// reflected borders, so border pixels need no erosion.
cv::Point warp_mask(const sc_compositor &c, int i, double scale, const cv::Size &size, cv::Mat &mask) {
    const cv::Mat ones(size, CV_8U, cv::Scalar(255));
    return warp(c, i, scale, ones, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask);
}

// Applies an EXIF orientation (1-8) to a photo stored as encoded, giving what a viewer shows.
cv::Mat oriented(const cv::Mat &stored, int orientation) {
    cv::Mat result;
    switch (orientation) {
        case 2: cv::flip(stored, result, 1); break;
        case 3: cv::rotate(stored, result, cv::ROTATE_180); break;
        case 4: cv::flip(stored, result, 0); break;
        case 5: cv::transpose(stored, result); break;
        case 6: cv::rotate(stored, result, cv::ROTATE_90_CLOCKWISE); break;
        case 7: cv::transpose(stored, result); cv::flip(result, result, -1); break;
        case 8: cv::rotate(stored, result, cv::ROTATE_90_COUNTERCLOCKWISE); break;
        default: result = stored; break;
    }
    return result;
}

// An RGBA16 buffer as stored, oriented and without alpha; checks it has the expected oriented size.
cv::Mat rgb_from(const uint16_t *rgba, int width, int height, int bytes_per_row, int orientation, const cv::Size &expected) {
    if (width < 1 || height < 1 || bytes_per_row < width * 8) throw std::runtime_error("invalid photo buffer");
    const cv::Mat stored(height, width, CV_16UC4, const_cast<uint16_t *>(rgba), size_t(bytes_per_row));
    cv::Mat rgb;
    cv::cvtColor(stored, rgb, cv::COLOR_RGBA2RGB);
    rgb = oriented(rgb, orientation);
    if (rgb.size() != expected) throw std::runtime_error("photo size does not match the alignment");
    return rgb;
}

void check(const sc_compositor &c) {
    if (c.cancelled.load()) throw std::runtime_error("cancelled");
}

// Largest rectangle of fully valid pixels. Exact on masks up to 60 MP; above that it is searched on a
// conservative downscale (a cell counts only when every pixel around its sample is valid), mapped back
// inside the valid area and grown at full resolution.
cv::Rect inscribed_rectangle(const cv::Mat &mask) {
    cv::Mat valid = mask == 255;
    int32_t r[4];
    if (valid.total() <= 60'000'000) {
        if (!stitchcore::largest_rectangle(valid.data, valid.cols, valid.rows, valid.step, r)) return cv::Rect();
        return cv::Rect(r[0], r[1], r[2], r[3]);
    }
    const double factor = 4096.0 / std::max(mask.cols, mask.rows);
    const int k = int(std::ceil(1 / factor)) + 1;
    cv::Mat eroded, small;
    cv::erode(valid, eroded, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k)));
    cv::resize(eroded, small, cv::Size(), factor, factor, cv::INTER_NEAREST);
    if (!stitchcore::largest_rectangle(small.data, small.cols, small.rows, small.step, r)) return cv::Rect();
    // INTER_NEAREST samples pixel floor(i / factor) for cell i, and every sample of a valid cell has a valid
    // neighbourhood wider than the spacing between samples: the span between the first and last samples
    // is valid.
    auto sample = [&](int cell, int limit) { return std::min(limit - 1, int(std::floor(cell / factor))); };
    const int x0 = sample(r[0], mask.cols), x1 = sample(r[0] + r[2] - 1, mask.cols);
    const int y0 = sample(r[1], mask.rows), y1 = sample(r[1] + r[3] - 1, mask.rows);
    cv::Rect rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    if (cv::countNonZero(valid(rect)) != int(rect.area())) {
        if (!stitchcore::largest_rectangle(valid.data, valid.cols, valid.rows, valid.step, r)) return cv::Rect();
        return cv::Rect(r[0], r[1], r[2], r[3]);
    }
    auto full_row = [&](int y, int a, int b) { return cv::countNonZero(valid(cv::Range(y, y + 1), cv::Range(a, b))) == b - a; };
    auto full_column = [&](int x, int a, int b) { return cv::countNonZero(valid(cv::Range(a, b), cv::Range(x, x + 1))) == b - a; };
    for (bool grew = true; grew;) {
        grew = false;
        if (rect.y > 0 && full_row(rect.y - 1, rect.x, rect.x + rect.width)) { --rect.y; ++rect.height; grew = true; }
        if (rect.y + rect.height < mask.rows && full_row(rect.y + rect.height, rect.x, rect.x + rect.width)) { ++rect.height; grew = true; }
        if (rect.x > 0 && full_column(rect.x - 1, rect.y, rect.y + rect.height)) { --rect.x; ++rect.width; grew = true; }
        if (rect.x + rect.width < mask.cols && full_column(rect.x + rect.width, rect.y, rect.y + rect.height)) { ++rect.width; grew = true; }
    }
    return rect;
}

// Gains of photo i at the size of its warped image, as one CV_32FC3 value or a CV_32FC3 map. The layout
// of getMatGains depends on the compensator: 1x1 for Gain, 4x1 CV_64F for Channels (in the order of the
// fed channels), a block grid for the block compensators.
cv::Mat gain_map(const sc_compositor &c, int i, const cv::Size &size) {
    if (c.gains.empty() || c.gains[i].empty()) return cv::Mat();
    const cv::Mat &g = c.gains[i];
    switch (c.options.exposure) {
        case SC_EXPOSURE_GAIN:
            return cv::Mat(1, 1, CV_32FC3, cv::Scalar::all(g.at<double>(0)));
        case SC_EXPOSURE_CHANNELS: {
            const double *v = g.ptr<double>();
            return cv::Mat(1, 1, CV_32FC3, cv::Scalar(v[0], v[1], v[2]));
        }
        case SC_EXPOSURE_BLOCKS: {
            cv::Mat map;
            cv::resize(g, map, size, 0, 0, cv::INTER_LINEAR);
            if (map.channels() == 1) cv::merge(std::vector<cv::Mat>{map, map, map}, map);
            map.convertTo(map, CV_32FC3);
            return map;
        }
        default:
            return cv::Mat();
    }
}

// cv::multiply takes a per-channel constant only as a Scalar.
cv::Scalar as_scalar(const cv::Mat &single) {
    const cv::Vec3f g = single.at<cv::Vec3f>(0, 0);
    return cv::Scalar(g[0], g[1], g[2]);
}

}  // namespace

extern "C" sc_compositor *sc_compositor_create(const sc_compose_image *images, int32_t count,
                                               const sc_compose_options *options, char *error,
                                               size_t error_length) {
    if (images == nullptr || options == nullptr || count < 1 || !(options->scale > 0) || options->scale > 1 ||
        !(options->seam_scale > 0) || options->seam_scale > 1) {
        write_error(error, error_length, "invalid compositor input");
        return nullptr;
    }
    try {
        auto c = std::make_unique<sc_compositor>();
        c->options = *options;
        c->count = count;
        for (int i = 0; i < count; ++i) {
            if (images[i].width < 1 || images[i].height < 1) throw std::runtime_error("invalid photo size");
            c->full.emplace_back(images[i].width, images[i].height);
            const double *t = images[i].transform;
            c->transform.emplace_back(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]);
            c->focal.push_back(images[i].focal);
            if (!planar(*c) && !(images[i].focal > 0)) throw std::runtime_error("missing focal length");
        }
        if (!planar(*c)) {
            std::vector<double> f = c->focal;
            std::nth_element(f.begin(), f.begin() + f.size() / 2, f.end());
            c->warped_scale = f[f.size() / 2];
        }
        std::vector<cv::Point> tls;
        std::vector<cv::Size> sizes;
        for (int i = 0; i < count; ++i) {
            const cv::Rect roi = warp_roi(*c, i, options->scale);
            tls.push_back(roi.tl());
            sizes.push_back(roi.size());
        }
        const cv::Rect canvas = cd::resultRoi(tls, sizes);
        c->origin = canvas.tl();
        c->canvas = canvas.size();
        if (c->canvas.width < 1 || c->canvas.height < 1 || c->canvas.width > 200000 || c->canvas.height > 200000) {
            throw std::runtime_error("the panorama would be " + std::to_string(c->canvas.width) + " x " +
                                     std::to_string(c->canvas.height) + " pixels");
        }
        c->seam_corners.resize(count);
        c->seam_images.resize(count);
        c->seam_masks.resize(count);
        c->seam_added.assign(count, false);
        c->corners.resize(count);
        c->sizes.resize(count);
        c->added.assign(count, false);
        return c.release();
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return nullptr;
    }
}

extern "C" void sc_compositor_free(sc_compositor *compositor) { delete compositor; }

extern "C" void sc_compositor_cancel(sc_compositor *compositor) {
    if (compositor) compositor->cancelled.store(true);
}

extern "C" void sc_compositor_canvas_size(const sc_compositor *c, int32_t *width, int32_t *height) {
    if (width) *width = c ? c->canvas.width : 0;
    if (height) *height = c ? c->canvas.height : 0;
}

extern "C" int32_t sc_compositor_outline(const sc_compositor *c, int32_t index, float *points, int32_t capacity) {
    if (c == nullptr || index < 0 || index >= c->count || points == nullptr || capacity < 4) return 0;
    try {
        const int per_edge = std::max(1, capacity / 4);
        const double s = c->options.scale;
        const cv::Size size = scaled(c->full[index], s);
        cv::Ptr<cd::RotationWarper> warper = planar(*c) ? nullptr : rotation_warper(*c, s);
        const cv::Mat K = planar(*c) ? cv::Mat() : camera_K(*c, index, s), R = planar(*c) ? cv::Mat() : camera_R(*c, index);
        const cv::Matx33d M = planar(*c) ? planar_at(*c, index, s) : cv::Matx33d::eye();
        int written = 0;
        const double w = size.width - 0.5, h = size.height - 0.5;
        const cv::Point2d corners[5] = {{-0.5, -0.5}, {w, -0.5}, {w, h}, {-0.5, h}, {-0.5, -0.5}};
        for (int edge = 0; edge < 4; ++edge) {
            for (int k = 0; k < per_edge && written < capacity; ++k) {
                const double t = double(k) / per_edge;
                const cv::Point2d p = corners[edge] + (corners[edge + 1] - corners[edge]) * t;
                cv::Point2f q;
                if (warper) {
                    q = warper->warpPoint(cv::Point2f(float(p.x), float(p.y)), K, R);
                } else {
                    const cv::Vec3d v = M * cv::Vec3d(p.x, p.y, 1);
                    q = cv::Point2f(float(v[0] / v[2]), float(v[1] / v[2]));
                }
                points[2 * written] = q.x - float(c->origin.x);
                points[2 * written + 1] = q.y - float(c->origin.y);
                ++written;
            }
        }
        return written;
    } catch (const std::exception &) {
        return 0;
    }
}

extern "C" void sc_compositor_seam_size(const sc_compositor *c, int32_t index, int32_t *width, int32_t *height) {
    cv::Size size;
    if (c && index >= 0 && index < c->count) size = scaled(c->full[index], c->options.seam_scale);
    if (width) *width = size.width;
    if (height) *height = size.height;
}

extern "C" void sc_compositor_image_size(const sc_compositor *c, int32_t index, int32_t *width, int32_t *height) {
    cv::Size size;
    if (c && index >= 0 && index < c->count) size = scaled(c->full[index], c->options.scale);
    if (width) *width = size.width;
    if (height) *height = size.height;
}

extern "C" int32_t sc_compositor_add_seam_image(sc_compositor *c, int32_t index, const uint16_t *rgba,
                                                int32_t width, int32_t height, int32_t bytes_per_row,
                                                int32_t orientation, char *error, size_t error_length) {
    if (c == nullptr || rgba == nullptr || index < 0 || index >= c->count || c->prepared) {
        write_error(error, error_length, "invalid seam image");
        return 1;
    }
    try {
        check(*c);
        const cv::Size size = scaled(c->full[index], c->options.seam_scale);
        // Exposure compensation and graph cut expect 0..255 values.
        cv::Mat rgb, warped, mask;
        rgb_from(rgba, width, height, bytes_per_row, orientation, size).convertTo(rgb, CV_8U, 1.0 / 257.0);
        c->seam_corners[index] = warp(*c, index, c->options.seam_scale, rgb, cv::INTER_LINEAR, cv::BORDER_REFLECT, warped);
        warp_mask(*c, index, c->options.seam_scale, size, mask);
        if (mask.size() != warped.size()) throw std::runtime_error("seam mask and image differ in size");
        warped.copyTo(c->seam_images[index]);
        mask.copyTo(c->seam_masks[index]);
        c->seam_added[index] = true;
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return std::string(e.what()) == "cancelled" ? 2 : 1;
    }
}

extern "C" int32_t sc_compositor_prepare(sc_compositor *c, char *error, size_t error_length) {
    if (c == nullptr || c->prepared ||
        !std::all_of(c->seam_added.begin(), c->seam_added.end(), [](bool v) { return v; })) {
        write_error(error, error_length, "every seam image must be added once before preparing");
        return 1;
    }
    try {
        check(*c);
        // Exposure: gains estimated on the 8-bit seam copies, applied by us at full precision.
        double gmax = 1;
        if (c->options.exposure != SC_EXPOSURE_NONE) {
            cv::Ptr<cd::ExposureCompensator> compensator;
            switch (c->options.exposure) {
                case SC_EXPOSURE_GAIN: {
                    auto gain = cv::makePtr<cd::GainCompensator>(1);
                    gain->setSimilarityThreshold(c->options.similarity);
                    compensator = gain;
                    break;
                }
                case SC_EXPOSURE_BLOCKS: compensator = cv::makePtr<cd::BlocksChannelsCompensator>(32, 32, 1); break;
                default: {
                    // Pixels that differ between photos (glare, parallax) are left out of the estimate.
                    auto channels = cv::makePtr<cd::ChannelsCompensator>(1);
                    channels->setSimilarityThreshold(c->options.similarity);
                    compensator = channels;
                    break;
                }
            }
            compensator->feed(c->seam_corners, c->seam_images, c->seam_masks);
            compensator->getMatGains(c->gains);
            for (const cv::Mat &g : c->gains) {
                double high = 1;
                cv::minMaxLoc(g.reshape(1), nullptr, &high);
                if (std::isfinite(high)) gmax = std::max(gmax, high);
            }
        }
        check(*c);
        // The blender sums in 16-bit integers: 16-bit values times the largest gain must stay below 2^14,
        // which leaves room for the two dilated masks that meet at a seam. Without seams every photo can
        // overlap every other.
        c->value_scale = 16383.0 / (65535.0 * gmax);
        if (c->options.seam == SC_SEAM_NONE && c->options.blend != SC_BLEND_NONE) c->value_scale /= std::max(1, c->count / 2);

        if (c->options.seam != SC_SEAM_NONE && c->count > 1) {
            std::vector<cv::UMat> images(c->count);
            for (int i = 0; i < c->count; ++i) c->seam_images[i].convertTo(images[i], CV_32F);
            cv::Ptr<cd::SeamFinder> finder;
            switch (c->options.seam) {
                case SC_SEAM_VORONOI: finder = cv::makePtr<cd::VoronoiSeamFinder>(); break;
                case SC_SEAM_DP: finder = cv::makePtr<cd::DpSeamFinder>(cd::DpSeamFinder::COLOR_GRAD); break;
                default: finder = cv::makePtr<cd::GraphCutSeamFinder>(cd::GraphCutSeamFinderBase::COST_COLOR_GRAD); break;
            }
            finder->find(images, c->seam_corners, c->seam_masks);
        }
        c->seam_images.clear();
        check(*c);

        for (int i = 0; i < c->count; ++i) {
            const cv::Rect roi = warp_roi(*c, i, c->options.scale);
            c->corners[i] = roi.tl();
            c->sizes[i] = roi.size();
        }
        if (c->options.blend == SC_BLEND_NONE) {
            c->hard = cv::Mat::zeros(c->canvas, CV_16UC3);
            c->hard_mask = cv::Mat::zeros(c->canvas, CV_8U);
        } else {
            const float width = std::sqrt(float(c->canvas.area())) * 5.f / 100.f;
            const int bands = std::clamp(int(std::ceil(std::log2(std::max(width, 2.f)))) - 1, 1, 10);
            c->blender = cv::makePtr<cd::MultiBandBlender>(false, bands, CV_32F);
            c->blender->prepare(cv::Rect(c->origin, c->canvas));
        }
        c->prepared = true;
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return std::string(e.what()) == "cancelled" ? 2 : 1;
    }
}

extern "C" int32_t sc_compositor_add_image(sc_compositor *c, int32_t index, const uint16_t *rgba, int32_t width,
                                           int32_t height, int32_t bytes_per_row, int32_t orientation, char *error,
                                           size_t error_length) {
    if (c == nullptr || rgba == nullptr || index < 0 || index >= c->count || !c->prepared || c->added[index]) {
        write_error(error, error_length, "photos must be added once, after preparing");
        return 1;
    }
    try {
        check(*c);
        const cv::Size size = scaled(c->full[index], c->options.scale);
        cv::Mat rgb = rgb_from(rgba, width, height, bytes_per_row, orientation, size), warped, mask;
        const cv::Point tl = warp(*c, index, c->options.scale, rgb, interpolation(*c), cv::BORDER_REFLECT, warped);
        rgb.release();
        warp_mask(*c, index, c->options.scale, size, mask);
        if (tl != c->corners[index] || warped.size() != c->sizes[index] || mask.size() != warped.size()) {
            throw std::runtime_error("warped photo does not match its predicted area");
        }
        // The seam mask, dilated and scaled up, limits the photo to its side of the seams.
        if (c->count > 1 && c->options.seam != SC_SEAM_NONE) {
            cv::Mat seam, dilated;
            c->seam_masks[index].copyTo(seam);
            cv::dilate(seam, dilated, cv::Mat());
            cv::resize(dilated, seam, mask.size(), 0, 0, cv::INTER_LINEAR_EXACT);
            cv::bitwise_and(mask, seam, mask);
        }
        check(*c);

        const cv::Mat gains = gain_map(*c, index, warped.size());
        if (c->options.blend == SC_BLEND_NONE) {
            cv::Mat values = warped;
            if (!gains.empty()) {
                cv::Mat f;
                warped.convertTo(f, CV_32F);
                if (gains.total() == 1) cv::multiply(f, as_scalar(gains), f);
                else cv::multiply(f, gains, f);
                f.convertTo(values, CV_16U);
            }
            const cv::Rect target(tl - c->origin, warped.size());
            values.copyTo(c->hard(target), mask);
            cv::Mat region = c->hard_mask(target);
            cv::bitwise_or(region, mask, region);
        } else {
            cv::Mat feed;
            if (gains.empty()) {
                warped.convertTo(feed, CV_16S, c->value_scale);
            } else {
                cv::Mat f;
                warped.convertTo(f, CV_32F, c->value_scale);
                if (gains.total() == 1) cv::multiply(f, as_scalar(gains), f);
                else cv::multiply(f, gains, f);
                f.convertTo(feed, CV_16S);
            }
            warped.release();
            c->blender->feed(feed, mask, tl);
        }
        c->added[index] = true;
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return std::string(e.what()) == "cancelled" ? 2 : 1;
    }
}

extern "C" int32_t sc_compositor_finish(sc_compositor *c, sc_panorama *out, char *error, size_t error_length) {
    if (out) *out = sc_panorama{};
    if (c == nullptr || out == nullptr || !c->prepared ||
        !std::all_of(c->added.begin(), c->added.end(), [](bool v) { return v; })) {
        write_error(error, error_length, "every photo must be added before finishing");
        return 1;
    }
    try {
        check(*c);
        cv::Mat values, mask;
        if (c->options.blend == SC_BLEND_NONE) {
            values = c->hard;
            mask = c->hard_mask;
        } else {
            cv::Mat blended;
            c->blender->blend(blended, mask);  // CV_16SC3, values times value_scale
            c->blender.release();
            blended.convertTo(values, CV_16U, 1.0 / c->value_scale);
        }
        check(*c);
        // The whole covered area with alpha = coverage; the largest rectangle without empty pixels is
        // returned alongside, so cropping can be switched without stitching again.
        const cv::Rect inscribed = inscribed_rectangle(mask);
        uint16_t *pixels = stitchcore::allocate_array<uint16_t>(size_t(values.total()) * 4);
        if (pixels == nullptr) throw std::runtime_error("not enough memory for the panorama");
        cv::Mat rgba(values.size(), CV_16UC4, pixels);
        cv::Mat alpha16;
        mask.convertTo(alpha16, CV_16U, 65535.0 / 255.0);
        const cv::Mat planes[2] = {values, alpha16};
        const int from_to[8] = {0, 0, 1, 1, 2, 2, 3, 3};
        cv::mixChannels(planes, 2, &rgba, 1, from_to, 4);
        // Straight alpha: no colour under fully transparent pixels.
        rgba.setTo(cv::Scalar::all(0), mask == 0);
        out->width = values.cols;
        out->height = values.rows;
        out->crop[0] = inscribed.x;
        out->crop[1] = inscribed.y;
        out->crop[2] = inscribed.width;
        out->crop[3] = inscribed.height;
        out->opaque = cv::countNonZero(mask) == int(mask.total()) ? 1 : 0;
        out->pixels = pixels;
        c->hard.release();
        c->hard_mask.release();
        return 0;
    } catch (const std::exception &e) {
        write_error(error, error_length, e.what());
        return std::string(e.what()) == "cancelled" ? 2 : 1;
    }
}

extern "C" void sc_panorama_free(sc_panorama *panorama) {
    if (panorama == nullptr) return;
    std::free(panorama->pixels);
    *panorama = sc_panorama{};
}

extern "C" int32_t sc_inscribed_rectangle(const uint8_t *mask, int32_t width, int32_t height, int32_t bytes_per_row,
                                          int32_t *rect) {
    if (mask == nullptr || rect == nullptr || width < 1 || height < 1 || bytes_per_row < width) return 0;
    try {
        const cv::Mat m(height, width, CV_8U, const_cast<uint8_t *>(mask), size_t(bytes_per_row));
        const cv::Rect r = inscribed_rectangle(m);
        rect[0] = r.x;
        rect[1] = r.y;
        rect[2] = r.width;
        rect[3] = r.height;
        return r.empty() ? 0 : 1;
    } catch (const std::exception &) {
        return 0;
    }
}
