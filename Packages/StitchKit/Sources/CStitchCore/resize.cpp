// Area-averaging reduction of the analysis images, with OpenCV so that it runs optimised in every build.

#include "stitchcore.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

extern "C" int32_t sc_resize_area_rgba8(const uint8_t *pixels, int32_t width, int32_t height, int32_t bytes_per_row,
                                        uint8_t *out, int32_t out_width, int32_t out_height) {
    if (pixels == nullptr || out == nullptr || width < 1 || height < 1 || bytes_per_row < width * 4 || out_width < 1 ||
        out_height < 1 || out_width > width || out_height > height) {
        return 0;
    }
    try {
        const cv::Mat input(height, width, CV_8UC4, const_cast<uint8_t *>(pixels), size_t(bytes_per_row));
        cv::Mat output(out_height, out_width, CV_8UC4, out);
        cv::resize(input, output, output.size(), 0, 0, cv::INTER_AREA);
        // resize writes into the caller's buffer only when the destination already has the right size and type.
        return output.data == out ? 1 : 0;
    } catch (const std::exception &) {
        return 0;
    }
}
