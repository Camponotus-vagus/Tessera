# Architecture

## Layout

| Path | Contents |
|---|---|
| `App/` | SwiftUI app: session state, graph, pair and panorama views, sidebar, settings inspector, export panel, model download |
| `Packages/StitchKit/Sources/StitchKit` | the engine in Swift: pipeline, verification, graph, report, alignment, compositing, file export, model installer |
| `Packages/StitchKit/Sources/CStitchCore` | C++ with a plain C API (`include/stitchcore.h`): OpenCV, Accelerate and, optionally, ONNX Runtime |
| `Packages/StitchKit/Sources/stitchbench` | command-line driver |
| `Packages/StitchKit/Tests` | Swift Testing suites (synthetic mosaics, matching, geometry, extractor equivalence) |
| `tools/export` | scripts that turn the PyTorch weights into the Core ML and ONNX models |
| `tools/build-opencv.sh` | builds the static OpenCV in `Vendor/opencv` (core, imgproc, features, flann, geometry, stitching) |

Swift never sees C++ types. Everything crosses the boundary as flat float arrays and small C structs, and every native handle (SIFT features, ONNX sessions, the descriptor head) is owned by a Swift class that frees it in `deinit`. `StitchEngine` is an actor; the per-photo and per-pair work runs in task groups and detached tasks, and Core ML predictions on the dense extractor are serialised by a lock so the GPU runs one at a time while the CPU stages of other photos proceed.

## Pipeline

`StitchEngine.analyze(urls:configuration:excluded:progress:)` returns a `MatchReport`.

### 1. Features

Both matchers work on reduced copies made by `ImageLoader.analysisImage`: the photo is decoded at a power-of-two reduction, which ImageIO does exactly on the pixel-centre map for JPEG, HEIC, PNG and TIFF, reduced to the working size by area averaging, as LightGlue's own image loading does, and turned upright by its EXIF orientation. ImageIO's thumbnails, used before, do not follow that map: JPEG thumbnails are shrunk by up to 5e-4 at 1024 pixels, which put the two edges of an overlap more than half a pixel apart in full-resolution terms.

- **RootSIFT**: OpenCV SIFT on a downscaled gray image (1.5 MP by default, up to 6000 keypoints), descriptors L1-normalised and square-rooted. Coordinates are scaled back to the original image through pixel centres, (x + 0.5) / s - 0.5.
- **RaCo-ALIKED**: each photo is fitted, without distortion, into a fixed 1024 x 768 (or 768 x 1024) canvas, and up to three photos are in flight at once. RaCo upsamples its coarser feature levels with `align_corners`, which stretches the keypoints by about 0.42 canvas pixels from edge to edge (measured on photos shifted by known amounts: 0.37 to 0.47 depending on the scene, the same along both axes of the canvas); the conversion back to the original image pulls them towards the canvas centre by that much. The network is split in three:
  - Core ML (GPU, fp32): RaCo's score map and ranker map, and ALIKED's four feature levels at their own resolutions (1, 1/2, 1/8, 1/32).
  - C++ (`select.cpp`, CPU): non-maximum suppression, top-k, sub-pixel refinement and the boundary ranker, giving 2048 keypoints in under a millisecond. It reproduces the exported ONNX select model for 1024 to 2560 keypoints, where RaCo ranks the boundary window: on the test photos, letterboxed too, the same keypoints inside the photo within 1e-4 px, in the same order except among exactly equal logits. The ONNX model remains available for comparison in builds with ONNX Runtime.
  - C++ (`descriptor_head.cpp`): ALIKED's sparse deformable descriptor head. It rebuilds the upsampled, L2-normalised feature vector only at the 9 patch pixels and 4 x 16 deformable sample corners of each keypoint, then runs the head's layers as matrix products. Building the full-resolution 128-channel map instead would cost 400 MB per photo.

### 2. Candidate pairs

With four photos or fewer, every pair is matched. Otherwise the affinity of a pair is the number of mutual nearest neighbours passing Lowe's ratio test among the best 512 descriptors of each photo (learned descriptors when available). The candidates are the union of consecutive shots (by capture time, then file name), the two partners with the highest affinity for each photo, and a maximum spanning tree of the affinity, so that no photo is left without a path to the others.

### 3. Matching

- **RootSIFT**: exact nearest neighbours from one Accelerate matrix product per block of rows (the descriptors have unit norm, so distances follow from dot products), Lowe's ratio test at 0.8 and a mutual check.
- **LightGlue** on Core ML (GPU, fp16 by default). Keypoints are centred on the fixed canvas and divided by half its long edge (512 px), as LightGlue's `normalize_keypoints` does. The exported matcher returns, for every keypoint of the first photo, its partner and a confidence that is zero when the pair is not mutual; matches above 0.1 are kept. Consecutive shots are matched while extraction is still running.

### 4. Verification

For each pair and each candidate motion model of the chosen mode:

| Model | Estimator |
|---|---|
| translation | `estimateTranslation2D`, RANSAC |
| similarity | `estimateAffinePartial2D`, RANSAC |
| affine | `estimateAffine2D`, USAC MAGSAC++ |
| homography | `findHomography`, USAC MAGSAC++ |

The inlier threshold is 3 px at 1 MP, scaled to the resolution of the second photo, where the residuals are measured. The simplest model that keeps at least 92% of the best inlier count, with an RMS error no larger than 1.5 times the best model's plus a tenth of the threshold, is chosen. A homography is stored with the sign that puts its inliers in front of the camera, and the overlap is the part of the first photo's rectangle where the projective map is in front of the camera and lands inside the second photo (five linear half-planes), so it stays correct for wide rotations. The pair is then:

- **perspective change** in plane mode, if a homography fitted alongside explains at least 1.5 times as many matches as the chosen model: the photos are not tiles of a plane, and document or automatic mode is the right choice;
- **implausible** if the transform folds the image, or, for planar scenes, scales by more than 3x (relative to the ratio of the two resolutions) or shears with an anisotropy above 2;
- **degenerate** if the convex hull of the inliers, after removing its outer layer of points, covers less than 5% of the overlap or 0.5% of the image (typical of repeated labels or a line of text);
- **not significant** if the a-contrario number of false alarms, NFA = (n - s) C(n, k) C(k, s) alpha^(k - s), is not below 1;
- **verified** otherwise.

The Brown-Lowe confidence n_inliers / (8 + 0.3 n_overlap) is reported too, but not used for the decision: reflections and parallax produce matches that are consistent with another motion and make it reject correct pairs.

### 5. Groups and bridges

Verified pairs define connected components. If more than one remains, up to three rounds try the two untried pairs with the highest affinity between every two groups, keeping only pairs with some affinity and at most as many pairs per round as there are photos.

### 6. Graph and report

`MatchGraphBuilder` assigns each photo to a component and gives the photos outside the largest one a reason (a lone photo is never shown as joined). Photos that cannot be opened or decoded are reported as unreadable instead of stopping the run. `GraphLayout` places each component by chaining pairwise transforms along a maximum spanning tree of inlier counts, grown from the centre of the graph; homographies are replaced by the similarity that matches them at the centre of the overlap, so long rotation chains stay in place. Results are sorted, so the same input gives the same report, and an analysis can be cancelled between stages. The `MatchReport` (versioned JSON) contains the images, features, candidate pairs with the reason each was chosen, every pair's matches, fits, inliers, overlap polygons and verdict, and the graph.

## Stitching

`StitchEngine.stitch(_:request:progress:)` turns a `MatchReport` into a `Panorama`: the largest group of photos, aligned, warped, blended and kept in memory at 16 bits per channel until it is exported.

### 7. Global alignment

`AlignmentProblem.build` collects, for every verified pair of the group, the inliers of each matcher that verified it, removes duplicates closer than half the inlier threshold, refits a homography to the union (keeping the best matcher's inliers alone when the refit keeps fewer points than 90% of them) and keeps at most 300 matches per pair, sampled over an 8 x 8 grid so that a dense patch does not outweigh the rest of the overlap. `Aligner` and `align.cpp` then solve one of these models over all photos at once:

| Model | Solver |
|---|---|
| translation, similarity, affine | weighted linear least squares with Huber reweighting, one photo fixed |
| homography | Levenberg-Marquardt with an analytic Jacobian; the reference photo is the one that keeps the largest stretch of the others smallest |
| rotation | OpenCV `HomographyBasedEstimator` for the first rotations, with the focal-length prior described below, then `BundleAdjusterRay`, or `BundleAdjusterReproj` with the focal length fixed when that fails |

For rotation, the focal length starts from the EXIF data when every photo has it (35 mm equivalent, f = f35 * hypot(w, h) / 43.27), else from the pairwise homographies, else from a 72-degree field of view across the long side. The ray adjuster's result is rejected when, with EXIF, the photos' focal lengths differ by more than 5%, or when its median focal length leaves 0.67 to 1.5 times a prior taken from EXIF or the homographies (the 72-degree fallback is not checked). The reprojection adjuster then refines the rotations alone, and its result must pass the same test. The ray adjuster weighs every match alike, so whatever moved between shots (water, people) pulls the whole panorama. When it succeeds, it runs once more, for at most 20 iterations, from its own result without the matches more than three times the median error (and at least 2 px) off it. A pair that would keep fewer than 8 of its matches (or fewer than all, when it has fewer than 8) is left out of that second run. There is no second run when no match is beyond the limit, when more than 30% of them are, when the pairs left would no longer join every photo, or after the reprojection adjuster, whose errors mostly measure a wrong focal length. The second result is turned so that the first one's reference photo keeps its orientation. On eleven synthetic scenes of a turning camera with a band that drifts 3 px per shot, the worst error away from the band drops from 0.49-0.99 px to 0.11-0.26 px. The reported errors still count every match. Wave correction levels the horizon when there are at least three photos spanning 30 degrees or more.

Plane mode picks the simplest of translation, similarity and affine whose RMS error is within 1.25 times the best plus half a pixel, and notes when a homography would earn its perspective (see below) over the best of the three and lower its error by more than a pixel. Document mode uses the homography. Rotation mode falls back to the homography, with a note, if the photos do not fit a turning camera. Automatic mode solves all of them and takes the first that fits: translation, then similarity, each when its RMS error is within 1.25 times the best plus half a pixel; then rotation, when its error is within that tolerance or within 1.5 times the homography's plus one pixel, or when the homography fails or stretches a photo more than four times; then affine, when it is within the tolerance or when the homography does not earn its perspective; then the homography. The tolerance is never below half a pixel, nor above 1.25 times the best planar fit plus half a pixel, so a translation within half a pixel, or a similarity within half a pixel when the translation is outside that upper bound, is taken without solving the homographies and the rotation. A homography earns its perspective over a planar fit when it stretches no photo more than four times and halves the planar fit's error. Parallax (specimens on pins, a camera held by hand) lets a chain of homographies lower the error a little while the perspective builds up along the chain: on the drawer the homographies reach 4.4 px against 6.5 px for the affine, but bend the row of photos into a fan and stretch the first ones almost twice, so the affine is kept. How far they stretch depends on small differences in the matches (1.2 to 1.95 times over the original order and shuffled copies of the same photos), so the rule does not count on it. Four synthetic tiles of a flat page at different tilts give 0.7 px against 21 px. After solving, pairs are dropped one at a time, worst first, while there are at least as many pairs as photos, the worst error is above three times the median and the inlier threshold, and the group stays connected without the pair: up to two pairs, or a quarter of the pairs beyond a spanning tree when that is more. Each time the same model is solved again. If it fits worse without the pair, the pair was what made it look plausible (on one shuffled copy of the drawer, a rotation that ended 3553 px off), and the model is chosen again at once; otherwise it is chosen again after the last drop. A note follows when the pairs' errors, each divided by its own inlier threshold, have an RMS above one: every pair was verified to within its threshold.

### 8. Compositing

`Compositor` (`compositor.cpp`) reads the photos twice: small copies first, to estimate exposure and place the seams, then each photo at the output resolution (full, half or a quarter), warped and handed to the blender one at a time, so that only the photo in hand is held at that size.

- Projection: rotation panoramas use OpenCV's plane (rectilinear), cylindrical and spherical warpers, with the yaw recentred on the middle of the panorama. Automatic projection picks rectilinear when the horizontal and vertical fields of view are both within 100 degrees and no photo is more than 60 degrees off axis, cylindrical when only the vertical field is within 100 degrees, and spherical otherwise. Panoramas of 330 degrees or more are refused for now. Planar models are warped with `warpPerspective` onto the reference photo's plane. Each copy of a photo has a whole number of pixels, and it is placed by its actual size, with pixel centres mapped as (x + 0.5) * r - 0.5, so that photos keep their full-resolution distances at any scale. `ImageLoader.stored` makes the copies with vImage's high-quality scaling, which follows that map; CoreGraphics, when it draws an image smaller, maps the corner pixels onto each other instead: at a tenth of the size, the copy's edge pixels land about half a pixel of the copy (4.5 full-resolution pixels) from where the pixel-centre map puts them. The seam copies are never larger than the output ones. Colour is interpolated with bicubic weights and reflected borders; masks use nearest neighbours, so no photo bleeds past its edge.
- Exposure: gains are estimated on 8-bit copies at seam resolution and applied to the 16-bit pixels at output size. By default there is one gain per photo and colour channel, estimated only on overlap pixels whose colours differ by at most 0.15 (normalised RGB distance), so that glare and parallax do not bias it; the alternative is a grid of gains per photo, for vignetting.
- Seams: a graph cut on colour and gradient differences (`COST_COLOR_GRAD`) at about 0.1 megapixels per photo, dynamic programming above 30 photos, or the Voronoi split halfway between photos. The finder runs one pair of photos at a time, so that Stop takes effect between pairs; the exposure estimate, the seam of one pair and the final blend run to their end.
- Blending: multi-band blending on 16-bit signed pyramids. The pyramids add up the weighted Laplacians of every photo, each at most the largest value, with weights that add up to at most the number of photos meeting at a pixel; values are scaled so that the largest gain times the largest such number in the panorama fits, which keeps about 13 bits when four photos meet somewhere. With Original values the seams are hard: every pixel is copied from one warped photo, and the alpha stays 0 or 65535.

The panorama is uncropped RGBA at 16 bits per channel, unpremultiplied, in the photos' own colour space when they share one (Display P3 for iPhone photos) and in Display P3 otherwise. Its largest axis-aligned rectangle without transparent pixels is found exactly with the histogram method up to 60 megapixels, and on a reduced mask with a conservative mapping back above that. If the panorama would need more than half the Mac's memory, or be more than 200,000 pixels wide or tall, it is made at a lower resolution and the result says so; a canvas more than four times that size at the requested scale is refused as a broken alignment. `PanoramaWriter` writes PNG, JPEG, TIFF (LZW at 8 bits, uncompressed at 16, where ImageIO's LZW makes 16-bit photos larger and Deflate saves little for much more time) or HEIC through ImageIO, flattening the transparent area onto white for the formats without alpha, and replaces an existing file only once the new one is complete.

## Models

The engine looks for the learned models in `TESSERA_MODELS`, then `Contents/Resources/Models` inside the app, then `~/Library/Application Support/Tessera/Models`, then `Models/` in the working directory or one of its parents (a source checkout). Core ML packages are compiled once into `~/Library/Caches/Tessera/CoreML`, keyed by a digest of the package's files; a damaged entry is compiled again. The report records which extractor and matcher actually ran. When the learned matcher cannot run the chosen backends and precisions (`LearnedModelSet.supports`: models missing from the set, or ONNX Runtime missing from the build) and RootSIFT is selected too, RootSIFT runs alone and the report says why (`learnedProblem`); the app's inspector offers only the choices that the set and the build can run.

The app does not ship the models. `ModelManifest` reads the release asset (URL, SHA-256 and size) from `models.json` in the app's resources (a copy of `tools/models.json`) or, in a source checkout, from `tools/models.json` itself. `ModelInstaller` streams the archive with a URLSession data task into a hidden folder next to `Application Support/Tessera/Models`, hashing it with CryptoKit as it arrives (an HTTP error status stops it before the body), unpacks it with `ditto` and checks that the result is a usable model set, all inside that folder. It then puts the new set in place with a single rename that swaps it with the old one (`RENAME_SWAP`), so an installed set stays usable until its replacement is complete. A failure or a cancelled task deletes the hidden folder and leaves the installed set as it was; the app cancels a download when it quits and waits for that. A hidden folder that a crash leaves behind is deleted by the next installation once nothing has written to it for ten minutes. Removing the models also deletes their compiled entries in the Core ML cache. The app and `stitchbench --download-models` use the same installer.
