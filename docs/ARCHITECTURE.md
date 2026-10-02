# Architecture

## Layout

| Path | Contents |
|---|---|
| `App/` | SwiftUI app: session state, graph view, pair view, sidebar, settings inspector |
| `Packages/StitchKit/Sources/StitchKit` | the engine in Swift: pipeline, verification, graph, report |
| `Packages/StitchKit/Sources/CStitchCore` | C++ with a plain C API (`include/stitchcore.h`): OpenCV, ONNX Runtime, Accelerate |
| `Packages/StitchKit/Sources/stitchbench` | command-line driver |
| `Packages/StitchKit/Tests` | Swift Testing suites (synthetic mosaics, matching, geometry, extractor equivalence) |
| `tools/export` | scripts that turn the PyTorch weights into the Core ML and ONNX models |

Swift never sees C++ types. Everything crosses the boundary as flat float arrays and small C structs, and every native handle (SIFT features, ONNX sessions, the descriptor head) is owned by a Swift class that frees it in `deinit`. `StitchEngine` is an actor; the per-photo and per-pair work runs in task groups and detached tasks, and Core ML predictions on the dense extractor are serialised by a lock so the GPU runs one at a time while the CPU stages of other photos proceed.

## Pipeline

`StitchEngine.analyze(urls:configuration:excluded:)` returns a `MatchReport`.

### 1. Features

- **RootSIFT.** OpenCV SIFT on a downscaled gray image (1.5 MP by default, up to 6000 keypoints), descriptors L1-normalised and square-rooted. Coordinates are scaled back to the original image.
- **RaCo-ALIKED.** Each photo is fitted, without distortion, into a fixed 1024 x 768 (or 768 x 1024) canvas. The network is split in three:
  - Core ML (GPU, fp32): RaCo's score map and ranker map, and ALIKED's four feature levels at their own resolutions (1, 1/2, 1/8, 1/32).
  - ONNX Runtime (CPU): non-maximum suppression, top-k, sub-pixel refinement and the boundary ranker, giving 2048 keypoints.
  - C++ (`descriptor_head.cpp`): ALIKED's sparse deformable descriptor head. It rebuilds the upsampled, L2-normalised feature vector only at the 9 patch pixels and 4 x 16 deformable sample corners of each keypoint, then runs the head's layers as matrix products. Building the full-resolution 128-channel map instead would cost 400 MB per photo.
  Up to three photos are in flight at once.

### 2. Candidate pairs

With four photos or fewer, every pair is matched. Otherwise the affinity of a pair is the number of mutual nearest neighbours passing Lowe's ratio test among the best 512 descriptors of each photo (learned descriptors when available). The candidates are the union of consecutive shots (by capture time, then file name), the two most affine partners of each photo, and a maximum spanning tree of the affinity, so that no photo is left without a path to the others.

### 3. Matching

- **RootSIFT**: exact nearest neighbours from one Accelerate matrix product per block of rows (the descriptors have unit norm, so distances follow from dot products), Lowe's ratio test at 0.8 and a mutual check.
- **LightGlue** on Core ML (GPU, fp16 by default). Each photo's keypoints are normalised by its own long edge. The exported matcher returns, for every keypoint of the first photo, its partner and a confidence that is zero when the pair is not mutual; matches above 0.1 are kept. Consecutive shots are matched while extraction is still running.

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

## Models

The engine looks for the learned models in `TESSERA_MODELS`, then `Contents/Resources/Models` inside the app, then `~/Library/Application Support/Tessera/Models`, then `Models/` in the working directory or one of its parents (a source checkout). Core ML packages are compiled once into `~/Library/Caches/Tessera/CoreML`, keyed by a digest of the package's files; a damaged entry is compiled again. The report records which extractor and matcher actually ran.

The app does not ship the models. `ModelManifest` reads the release asset (URL and SHA-256) from `models.json` in the app's resources (a copy of `tools/models.json`) or, in a source checkout, from `tools/models.json` itself. `ModelInstaller` downloads the archive with URLSession, checks its SHA-256 with CryptoKit, unpacks it with `ditto` and checks that the result is a usable model set, all inside a hidden folder next to `Application Support/Tessera/Models`. It then puts the new set in place with a single rename that swaps it with the old one (`RENAME_SWAP`), so an installed set stays usable until its replacement is complete. A failure or a cancelled task deletes the hidden folder and leaves the installed set as it was. The app and `stitchbench --download-models` use the same installer.
