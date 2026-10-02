// Largest axis-aligned rectangle of valid pixels, used to crop the ragged border of a panorama.

#include "stitchcore.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace stitchcore {

// Maximal rectangle in a binary mask: for each row the column heights of consecutive valid pixels
// form a histogram whose largest rectangle is found with a stack, O(width * height) in total.
// Returns false when the mask has no valid pixel.
bool largest_rectangle(const uint8_t *mask, int width, int height, size_t stride, int32_t rect[4]) {
    std::vector<int> heights(static_cast<size_t>(width) + 1, 0);  // sentinel column of height 0
    std::vector<int> stack;
    stack.reserve(static_cast<size_t>(width) + 1);
    int64_t best = 0;
    for (int y = 0; y < height; ++y) {
        const uint8_t *row = mask + static_cast<size_t>(y) * stride;
        for (int x = 0; x < width; ++x) heights[x] = row[x] ? heights[x] + 1 : 0;
        stack.clear();
        for (int x = 0; x <= width; ++x) {
            while (!stack.empty() && heights[stack.back()] >= heights[x]) {
                const int h = heights[stack.back()];
                stack.pop_back();
                const int left = stack.empty() ? 0 : stack.back() + 1;
                const int64_t area = static_cast<int64_t>(h) * (x - left);
                if (area > best) {
                    best = area;
                    rect[0] = left;
                    rect[1] = y - h + 1;
                    rect[2] = x - left;
                    rect[3] = h;
                }
            }
            stack.push_back(x);
        }
    }
    return best > 0;
}

}  // namespace stitchcore

extern "C" int32_t sc_largest_rectangle(const uint8_t *mask, int32_t width, int32_t height, int32_t bytes_per_row,
                                        int32_t *rect) {
    if (mask == nullptr || rect == nullptr || width < 1 || height < 1 || bytes_per_row < width) return 0;
    rect[0] = rect[1] = rect[2] = rect[3] = 0;
    return stitchcore::largest_rectangle(mask, width, height, static_cast<size_t>(bytes_per_row), rect) ? 1 : 0;
}
