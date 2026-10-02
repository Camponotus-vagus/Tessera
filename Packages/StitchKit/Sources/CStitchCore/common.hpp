#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace stitchcore {

inline void write_error(char *buffer, size_t length, const std::string &message) {
    if (buffer == nullptr || length == 0) {
        return;
    }
    std::snprintf(buffer, length, "%s", message.c_str());
}

template <typename T>
T *allocate_array(size_t count) {
    if (count == 0) {
        count = 1;
    }
    return static_cast<T *>(std::malloc(count * sizeof(T)));
}

}  // namespace stitchcore
