# Benchmarks

Measured on a MacBook Air M5 (32 GB, macOS 27), release build of `stitchbench`. The MacBook Air is passively cooled: under long sustained load the same runs can be up to 1.5x slower, so treat the numbers as indicative.

## Test sets

- **Drawer**: six 4032 x 3024 photos of an insect drawer, shot handheld with an iPhone. Nearly planar subject, many nearly identical labels, glossy background with reflections that move with the camera, specimens on pins (parallax).
- **Grid**: 24 overlapping 1200 x 900 crops of one drawer photo, in a 6 x 4 grid (276 possible pairs).

## End to end

| Case | Pairs matched | Time |
|---|---|---|
| Drawer, RootSIFT + LightGlue, candidate pairs | 9 of 15 | 0.9 s |
| Drawer, LightGlue only, all pairs | 15 | 0.9 s |
| Grid, RootSIFT + LightGlue, candidate pairs | 31 of 276 | 2.9 s |
| Grid, RootSIFT + LightGlue, all pairs | 276 | 13.7 s (before the latest extractor changes) |

All photos end up in one group in every case. On the drawer the verified pairs are exactly the five consecutive shots; a false link between two rows of identical labels is rejected by the plausibility and coverage checks. With the capture order removed (files renamed in shuffled order, metadata stripped), the affinity's spanning tree alone recovers the same five pairs.

## Learned extractor, per photo (1024 x 768)

| Variant | Time | Same matches as reference |
|---|---|---|
| ONNX Runtime, whole extractor on the CPU | 1.25-1.95 s | reference |
| Core ML dense part (GPU) + ONNX sparse part | 0.16 s | yes (IoU 1.000) |
| + pipeline of three photos | 0.085 s | yes |
| + descriptor head in C++ on feature levels | 0.062 s | keypoints identical, descriptors within 5e-6 |

Tried and dropped: fp16 for the dense part on the GPU (faster, but 9% fewer inliers and a costly output conversion), Core ML output backings (slower), low-precision accumulation on the GPU (no effect), ONNX Runtime's CoreML execution provider (splits the graphs into 46-67 partitions and stays slower than native Core ML).

## Matcher, per pair

| Variant | Time | Same matches as reference |
|---|---|---|
| ONNX Runtime, CPU | 1.08 s | reference |
| Core ML fp32, GPU | 0.35 s | IoU 1.000 |
| Core ML fp16, GPU | 0.03 s from Swift (0.12 s from Python) | IoU 0.984 |
| Core ML fp16, Neural Engine | 0.28 s | IoU 0.96 |

## RootSIFT matching

Brute-force OpenCV matching took about 80 ms per pair with 6000 keypoints; the Accelerate matrix-product version takes about 20 ms and returns the same matches (checked against brute force in the test suite).

## Quality on the drawer

Inliers on the chosen model for the five consecutive pairs:

| Pair | RootSIFT | RaCo-ALIKED + LightGlue |
|---|---|---|
| 1-2 | 33 | 69 |
| 2-3 | 71 | 73 |
| 3-4 | 90 | 104 |
| 4-5 | 23 (rejected: inliers on a strip) | 47 |
| 5-6 | 62 | 165 |

Only 20-25% of the tentative matches are inliers even on correct pairs: reflections of the ceiling lights move with the camera and the pinned specimens show parallax, so they do not follow a single plane-to-plane transform.
