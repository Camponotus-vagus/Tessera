// Plain C interface to the C++ engine (OpenCV 5 + ONNX Runtime).
// This is the only header Swift sees; nothing here exposes C++ types.
#ifndef STITCHCORE_H
#define STITCHCORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// MARK: - Shared

typedef struct {
    float x;
    float y;
    float size;
    float angle;
    float response;
} sc_keypoint;

typedef struct {
    int32_t index_a;
    int32_t index_b;
    /// Higher is better, in [0, 1].
    float score;
} sc_match;

/// Frees buffers returned by this library.
void sc_free(void *pointer);

// MARK: - SIFT

typedef struct sc_features sc_features;

/// Detects SIFT keypoints on an 8-bit gray image and computes descriptors.
/// `max_features` = 0 means unlimited. With `root_sift` != 0 descriptors become RootSIFT
/// (L1 normalisation followed by an element-wise square root).
/// Returns NULL on failure and writes a message into `error` when provided.
sc_features *sc_sift_extract(const uint8_t *gray, int32_t width, int32_t height, int32_t bytes_per_row,
                             int32_t max_features, int32_t root_sift, char *error, size_t error_length);
int32_t sc_features_count(const sc_features *features);
const sc_keypoint *sc_features_keypoints(const sc_features *features);
/// Row-major descriptors, one row per keypoint; writes the row length into `*size`.
const float *sc_features_descriptors(const sc_features *features, int32_t *size);
void sc_features_free(sc_features *features);

/// Exact nearest-neighbour matching with Lowe's ratio test and an optional mutual check.
/// Distances come from a single matrix product (Accelerate), which equals brute force.
/// Writes a newly allocated array into `*matches` (free with sc_free) and returns its length,
/// or -1 on failure.
int32_t sc_match_descriptors(const sc_features *a, const sc_features *b, float ratio, int32_t mutual,
                             sc_match **matches, char *error, size_t error_length);

/// Same matching on raw row-major float descriptors (`count_a` x `size` and `count_b` x `size`).
int32_t sc_match_raw(const float *a, int32_t count_a, const float *b, int32_t count_b, int32_t size, float ratio,
                     int32_t mutual, sc_match **matches, char *error, size_t error_length);

// MARK: - Geometric verification

typedef enum {
    SC_MODEL_TRANSLATION = 0,
    SC_MODEL_SIMILARITY = 1,
    SC_MODEL_AFFINE = 2,
    SC_MODEL_HOMOGRAPHY = 3,
} sc_model;

typedef struct {
    int32_t ok;
    /// Row-major 3x3 matrix mapping points of image A onto image B.
    double transform[9];
    int32_t inlier_count;
    /// Reprojection error over inliers, in the units of the input points.
    double rmse;
    double median_error;
} sc_fit;

/// Robustly fits `model` to point pairs (interleaved x,y; `count` pairs).
/// Translation and similarity use RANSAC, affine and homography use USAC MAGSAC++.
/// `inlier_mask` must hold `count` bytes and receives 1 for inliers.
/// `seed` makes the result reproducible.
sc_fit sc_fit_model(const float *points_a, const float *points_b, int32_t count, sc_model model,
                    double threshold, int32_t max_iterations, double confidence, int32_t seed,
                    uint8_t *inlier_mask);

// MARK: - Learned features (RaCo-ALIKED extractor and LightGlue matcher, ONNX Runtime)

typedef enum {
    SC_EXECUTION_CPU = 0,
    SC_EXECUTION_COREML_CPU_ONLY = 1,
    SC_EXECUTION_COREML_CPU_AND_GPU = 2,
    SC_EXECUTION_COREML_ALL = 3,
} sc_execution;

/// An ONNX Runtime session for one of the split models.
typedef struct sc_onnx_model sc_onnx_model;

/// 1 when the library was built with ONNX Runtime; otherwise every ONNX entry point fails.
int32_t sc_onnx_available(void);

/// Loads a model. `cache_directory` may be NULL. `low_memory` != 0 disables the CPU arena.
sc_onnx_model *sc_onnx_model_create(const char *model_path, sc_execution execution, const char *cache_directory,
                                    int32_t intra_op_threads, int32_t low_memory, char *error,
                                    size_t error_length);
void sc_onnx_model_free(sc_onnx_model *model);

typedef struct {
    int32_t keypoint_count;
    int32_t descriptor_size;
    /// keypoint_count * 2 floats, pixels of the network input.
    float *keypoints;
    /// keypoint_count * descriptor_size floats.
    float *descriptors;
} sc_extraction;

/// Runs the extractor on one RGB planar float image in [0, 1] whose size matches the model.
int32_t sc_extract(sc_onnx_model *model, const float *chw, int32_t width, int32_t height, sc_extraction *result,
                   char *error, size_t error_length);
void sc_extraction_free(sc_extraction *result);

/// Runs the sparse half of the split extractor on the dense maps computed by Core ML:
/// `logits` and `ranker` [1, 1, H, W], `features` [1, C, H, W], all contiguous float32.
int32_t sc_extract_sparse(sc_onnx_model *model, const float *logits, const float *ranker, const float *features,
                          int32_t width, int32_t height, int32_t channels, sc_extraction *result, char *error,
                          size_t error_length);

/// Keypoint selection of the faster split: `logits` and `ranker` [1, 1, H, W] -> keypoints [K, 2].
/// Fills only the keypoints of `result` (descriptor_size is 0).
int32_t sc_select_keypoints(sc_onnx_model *model, const float *logits, const float *ranker, int32_t width,
                            int32_t height, sc_extraction *result, char *error, size_t error_length);

// MARK: - ALIKED descriptor head (C++, Accelerate)

typedef struct sc_descriptor_head sc_descriptor_head;

/// Loads the weights written by tools/export/split_extractor.py head.
sc_descriptor_head *sc_descriptor_head_load(const char *path, char *error, size_t error_length);
void sc_descriptor_head_free(sc_descriptor_head *head);
int32_t sc_descriptor_head_dimensions(const sc_descriptor_head *head);

/// Descriptors for `count` keypoints (pixels of a `width` x `height` image) from ALIKED's feature
/// levels, each [level_channels, level_heights[i], level_widths[i]] contiguous float32.
/// `descriptors` receives count * dimensions floats.
int32_t sc_describe(const sc_descriptor_head *head, const float *const *levels, const int32_t *level_widths,
                    const int32_t *level_heights, int32_t level_count, int32_t level_channels, int32_t width,
                    int32_t height, const float *keypoints, int32_t count, float *descriptors, char *error,
                    size_t error_length);

/// Runs the matcher. `keypoints` holds both images ([2, K, 2], normalised by each image's long edge),
/// `descriptors` [2, K, D]. Writes, for every keypoint of the first image, the index of its partner
/// in the second image and a confidence that is 0 when the pair is not mutual.
int32_t sc_match_learned(sc_onnx_model *model, const float *keypoints, const float *descriptors,
                         int32_t keypoint_count, int32_t descriptor_size, int32_t *partner, float *confidence,
                         char *error, size_t error_length);

// MARK: - Panorama helpers

/// Largest axis-aligned rectangle of non-zero pixels in `mask` (one byte per pixel). Writes x, y,
/// width, height into `rect` and returns 1, or returns 0 when the mask has no non-zero pixel.
int32_t sc_largest_rectangle(const uint8_t *mask, int32_t width, int32_t height, int32_t bytes_per_row,
                             int32_t *rect);

#ifdef __cplusplus
}
#endif

#endif
