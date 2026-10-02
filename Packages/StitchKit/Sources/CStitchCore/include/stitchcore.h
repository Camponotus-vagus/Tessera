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

// MARK: - RaCo keypoint selection (C++)

/// The same selection as sc_select_keypoints without ONNX Runtime: `logits` and `ranker` [H, W]
/// -> `keypoints` x 2 floats in `out`, canvas pixels. It reproduces RaCo only for
/// 1024 <= keypoints <= 2560, where RaCo re-ranks the boundary window; other counts from 256 to
/// W * H - 256 run the same steps (the tests use them on small maps), but RaCo selects differently there.
/// Same keypoints in the same order as the ONNX model except among exactly equal logits; ties and
/// non-finite values are resolved deterministically (see select.cpp).
int32_t sc_select_keypoints_native(const float *logits, const float *ranker, int32_t width, int32_t height,
                                   int32_t keypoints, float *out, char *error, size_t error_length);

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

// MARK: - Global alignment

typedef enum {
    SC_ALIGN_TRANSLATION = 0,
    SC_ALIGN_SIMILARITY = 1,
    SC_ALIGN_AFFINE = 2,
    SC_ALIGN_HOMOGRAPHY = 3,
    /// A camera rotating about its centre: a rotation and a focal length per photo.
    SC_ALIGN_ROTATION = 4,
} sc_align_model;

typedef struct {
    int32_t width;   // oriented full-resolution size
    int32_t height;
    double focal;    // rotation: focal length prior in pixels, 0 when unknown
} sc_align_image;

typedef struct {
    int32_t a;
    int32_t b;
    int32_t count;
    const float *points;   // count x (ax, ay, bx, by), full-resolution pixels, pixel centres at integers
    const float *sigma;    // count standard deviations in pixels, or NULL for 1
    double homography[9];  // a -> b, row-major, used to initialise
} sc_align_pair;

typedef struct {
    int32_t ok;
    /// RMS transfer error of all correspondences from a to b, full-resolution pixels of b.
    double rms;
    /// Planar models: solver iterations. Rotation: 0 ray bundle adjustment, 1 focal fixed at the prior.
    int32_t iterations;
} sc_align_result;

/// Aligns photos joined by `pairs` (one connected group). Planar models write, for every photo, the
/// row-major 3x3 map from its pixels to the mosaic, which is the anchor's pixel frame. Rotation writes
/// cv::detail's camera rotation R (world ray = R K^-1 p, K with the principal point at the photo centre)
/// and the focal length in `focals`. `wave`: -1 none, 0 horizontal, 1 vertical, 2 automatic.
/// `pair_rms` (optional) receives each pair's RMS transfer error.
int32_t sc_align(sc_align_model model, const sc_align_image *images, int32_t image_count,
                 const sc_align_pair *pairs, int32_t pair_count, int32_t anchor, int32_t wave, double *transforms,
                 double *focals, double *pair_rms, sc_align_result *result, char *error, size_t error_length);

/// Worst corner stretch (max of s and 1/s, s = sqrt|det J|) of planar transforms re-anchored on `anchor`;
/// infinity when a corner falls behind the plane. `sizes` holds width, height per photo.
double sc_alignment_stretch(const double *transforms, const int32_t *sizes, int32_t image_count, int32_t anchor);

// MARK: - Compositing

typedef enum {
    /// Planar maps (image -> mosaic pixels) for tiles and documents.
    SC_PROJECTION_PLANE = 0,
    /// Rotating camera, projected onto a plane, a cylinder or a sphere.
    SC_PROJECTION_RECTILINEAR = 1,
    SC_PROJECTION_CYLINDRICAL = 2,
    SC_PROJECTION_SPHERICAL = 3,
} sc_projection;

typedef enum { SC_SEAM_NONE = 0, SC_SEAM_VORONOI = 1, SC_SEAM_GRAPHCUT = 2, SC_SEAM_DP = 3 } sc_seam;
typedef enum { SC_EXPOSURE_NONE = 0, SC_EXPOSURE_GAIN = 1, SC_EXPOSURE_CHANNELS = 2, SC_EXPOSURE_BLOCKS = 3 } sc_exposure;
/// SC_BLEND_NONE copies each pixel from one photo, keeping its original values (no gains when exposure is NONE).
typedef enum { SC_BLEND_NONE = 0, SC_BLEND_MULTIBAND = 2 } sc_blend;

typedef struct {
    sc_projection projection;
    double scale;          // output pixels per full-resolution pixel, (0, 1]
    double seam_scale;     // seam and exposure copies, pixels per full-resolution pixel
    sc_seam seam;
    sc_exposure exposure;
    double similarity;     // exposure: colour difference above which pixels are ignored (0..1), 1 to use all
    sc_blend blend;
    int32_t interpolation; // 0 nearest, 1 linear, 2 cubic
} sc_compose_options;

typedef struct {
    int32_t width;          // oriented full-resolution size
    int32_t height;
    double transform[9];    // SC_PROJECTION_PLANE: image -> mosaic; otherwise the rotation R of sc_align
    double focal;           // rotation only
} sc_compose_image;

typedef struct {
    int32_t width;
    int32_t height;
    /// Largest rectangle (x, y, width, height) with no empty pixel; zero size when none fits.
    int32_t crop[4];
    /// 1 when every pixel is covered.
    int32_t opaque;
    /// RGBA, 16 bits per channel, straight alpha (0 or 65535), rows top to bottom; free with sc_panorama_free.
    uint16_t *pixels;
} sc_panorama;

typedef struct sc_compositor sc_compositor;

/// Plans the panorama: output size and photo outlines are known before any pixel is added.
sc_compositor *sc_compositor_create(const sc_compose_image *images, int32_t count, const sc_compose_options *options,
                                    char *error, size_t error_length);
void sc_compositor_free(sc_compositor *compositor);
/// Makes the current or next call return 2 (cancelled). Safe from any thread.
void sc_compositor_cancel(sc_compositor *compositor);
void sc_compositor_canvas_size(const sc_compositor *compositor, int32_t *width, int32_t *height);
/// Outline of photo `index` in panorama pixels, sampled along its edges: up to `capacity` points
/// (x, y pairs in `points`). Returns the number written.
int32_t sc_compositor_outline(const sc_compositor *compositor, int32_t index, float *points, int32_t capacity);
/// Sizes the photos must have when added: for the seam copies and for compositing.
void sc_compositor_seam_size(const sc_compositor *compositor, int32_t index, int32_t *width, int32_t *height);
void sc_compositor_image_size(const sc_compositor *compositor, int32_t index, int32_t *width, int32_t *height);
/// Every photo once, oriented RGBA 16-bit at sc_compositor_seam_size. Returns 0, 1 (error) or 2 (cancelled).
int32_t sc_compositor_add_seam_image(sc_compositor *compositor, int32_t index, const uint16_t *rgba,
                                     int32_t bytes_per_row, char *error, size_t error_length);
/// Exposure gains and seams, after every seam image.
int32_t sc_compositor_prepare(sc_compositor *compositor, char *error, size_t error_length);
/// Every photo once, oriented RGBA 16-bit at sc_compositor_image_size, after preparing.
int32_t sc_compositor_add_image(sc_compositor *compositor, int32_t index, const uint16_t *rgba, int32_t bytes_per_row,
                                char *error, size_t error_length);
int32_t sc_compositor_finish(sc_compositor *compositor, sc_panorama *panorama, char *error, size_t error_length);
void sc_panorama_free(sc_panorama *panorama);

// MARK: - Panorama helpers

/// Largest axis-aligned rectangle of non-zero pixels in `mask` (one byte per pixel). Writes x, y,
/// width, height into `rect` and returns 1, or returns 0 when the mask has no non-zero pixel.
int32_t sc_largest_rectangle(const uint8_t *mask, int32_t width, int32_t height, int32_t bytes_per_row,
                             int32_t *rect);

#ifdef __cplusplus
}
#endif

#endif
