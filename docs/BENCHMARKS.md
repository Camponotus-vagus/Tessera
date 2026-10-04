# Benchmarks

Measured on a MacBook Air M5 (32 GB, macOS 27), release build of `stitchbench`. The MacBook Air is passively cooled: under long sustained load the same runs can be up to 1.5x slower, so treat the numbers as indicative.

## Test sets

The sets of hundreds of photos are described under Large sets.

- **Drawer**: six 4032 x 3024 photos of an insect drawer, shot handheld with an iPhone. Nearly planar subject, many nearly identical labels, glossy background with reflections that move with the camera, specimens on pins (parallax).
- **Grid**: 24 overlapping 1200 x 900 crops of one drawer photo, in a 6 x 4 grid (276 possible pairs).
- **Box**: four 3024 x 4032 HEIC photos of a box of pinned ants, handheld, hundreds of nearly identical labels.
- **Newspaper** and **Boat**: OpenCV's stitching test images, four 818 x 1125 pieces of a newspaper page and six 3888 x 2592 photos taken by turning the camera.

## End to end

| Case | Pairs matched | Time |
|---|---|---|
| Drawer, RootSIFT + LightGlue, candidate pairs | 13 of 15 | 1.0 s |
| Drawer, LightGlue only, all pairs | 15 | 0.8 s |
| Grid, RootSIFT + LightGlue, candidate pairs | 88 of 276 | 5.0 s |
| Grid, RootSIFT + LightGlue, all pairs | 276 | 10.7 s |

These runs had other work going on in the background (load average about 3). On the drawer four of the five pairs of consecutive shots are verified, and so is 4-6. On 4-5 both matchers find consistent matches only along the top row of labels, because the large specimens in the bottom row show too much parallax. LightGlue's inliers cover just under 5% of the overlap (RootSIFT's 1%), the share below which the coverage check rejects inliers on a strip. The bridge rounds then try the pairs between the two groups, and RootSIFT joins them through 4-6. With LightGlue alone the drawer stays in two groups, 1-4 and 5-6; the grid and the other sets end up in one group in every case. A false link between two rows of identical labels is rejected by the plausibility and coverage checks. With the capture order removed (files renamed in shuffled order, metadata stripped), the spanning tree and the bridge rounds still join all six photos, with five or six verified pairs: whether 4-5, 4-6 or both get through depends on small differences in the matches, since 4-5 sits at the threshold. In two of the nine shuffled copies tried, 4-5 was verified 28 and 91 px off the other pairs; both times the global alignment left it out and kept the affine fit. With all pairs compared, the grid gets four false links between rows of identical labels, 864 to 1526 px off the other pairs; the alignment leaves all four out and joins the tiles to 0.43 px. With the candidate pairs, Automatic mode aligns the grid in 93 ms, to 0.42 px: a translation within half a pixel is taken without solving the rotation, which takes about a second on these tiles. The grid has more than eight tiles, so its candidates include the pairs a first layout predicts (57 of the 88), which close the loops between rows; without them it was 31 pairs in 2.8 s.

## Stitching

The app's default settings, which `stitchbench` gets with `--source both`: both matchers, automatic mode and projection, full resolution, seams around objects, multi-band blending. Analysis covers features, matching and verification; stitching covers alignment, warping, exposure, seams and blending, without writing the file.

| Set | Model and projection | Panorama | Alignment error | Analysis | Stitching | Peak memory |
|---|---|---|---|---|---|---|
| Drawer | affine, flat | 13401 x 3340 (45 MP) | 6.5 px | 1.2 s | 1.7 s | 4.9 GB |
| Box | affine, flat | 7982 x 4230 (34 MP) | 7.5 px | 0.9 s | 1.4 s | 3.9 GB |
| Newspaper | similarity, flat | 1793 x 1138 | 0.40 px | 0.7 s | 0.8 s | 1.1 GB |
| Boat | rotation, cylindrical | 10748 x 2759, cropped to 10710 x 2289 | 5.4 px | 0.8 s | 1.9 s | 3.7 GB |

The load average was about 3 here too. The analysis column is a single run, while the first table gives the third of three runs, which is about 0.15 s faster. On the drawer the stitching time splits into 0.02 s of alignment, 0.50 s for the seam copies and seam finding, 0.84 s of warping and exposure and 0.27 s of blending; writing a JPEG takes another 0.25 s. The seam copies are decoded at full size and resampled with vImage, which costs about 0.15 s per set and 0.4 to 0.7 GB more peak memory on these sets compared with drawing them smaller directly, and keeps them where the compositor expects them. The alignment errors of the drawer and the box come from parallax: the specimens stand on pins above the bottom of the drawer and the photos were taken by hand, so no single plane fits every match. The seams go around the specimens, which hides most of it. The newspaper, a flat page, aligns to under half a pixel.

## Large sets

Public sets, used only locally: NIST's 2015 microscopy stitching tiles (Level 1: 100 phase-contrast tiles in a 10 x 10 grid; Level 3: 552 Cy5 fluorescence tiles; both 1392 x 1040 at 16 bits), two OpenDroneMap drone surveys released under CC0 (Seneca: 167 near-infrared photos of farm fields, 3600 x 2700, taken by a fixed-wing drone without a gimbal; Zoo: 524 photos of a zoo and its surroundings, 4896 x 3672) and PTGui's Paris example (79 frames of 2000 x 3000 from a robotic head). Default settings, both matchers, automatic mode; the drone surveys were stitched at a reduced scale. Peak memory is the larger of the two runs.

| Set | Joined | Model | Panorama | Alignment error | Analysis | Stitching (of which alignment) | Peak memory |
|---|---|---|---|---|---|---|---|
| NIST Level 1 | 100 of 100 | translation | 12713 x 9526, full size | 0.78 px | 18 s | 9.2 s (0.6 s) | 6.5 GB |
| Paris | 51 of 79 | rotation, cylindrical | 13548 x 7856 after cropping, full size | 1.35 px | 31 s | 18.5 s (6.4 s) | 11.7 GB |
| Seneca | 162 of 167 | homography | 3715 x 4160 at a quarter | 7.6 px | 123 s | 66 s (46 s) | 4.6 GB |
| Zoo | 511 of 524 | homography | 10989 x 8138 at an eighth | 15.4 px | 394 s | 309 s (119 s) | 11.7 GB |

The NIST tiles span 3600 to 5500 of their 65535 levels: before they were stretched for analysis, RootSIFT found no keypoints and all 100 stayed apart. Level 3 still falls apart into 405 groups, the largest of 15 tiles, because most Cy5 tiles show a few cells on an empty background; that needs phase correlation and the grid layout, which Tessera does not use yet. The 28 Paris frames left out show only sky. On Zoo, the layout-predicted pairs (3868 tried, 2397 verified) take the analysis from 107 to 394 s, and they are what keeps the mosaic flat: without them the homographies bend a long branch of the survey towards the horizon, at 11.6 px, since the loops are missing that would show the error. Halving them to three new pairs per photo saved a fifth of the time and squeezed part of the mosaic. The alignment of Zoo took hours before the sparse solver and the rotation trial on 40 photos; the 20 photos it leaves out as misplaced are mostly treetops, where the matches repeat and the trees show parallax. Seneca's analysis grows from 33 to 123 s for the same reason.

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

Inliers on the chosen model for the five consecutive pairs and for 4-6:

| Pair | RootSIFT | RaCo-ALIKED + LightGlue |
|---|---|---|
| 1-2 | 38 | 77 |
| 2-3 | 86 | 75 |
| 3-4 | 88 | 104 |
| 4-5 | 23 (rejected: inliers on a strip) | 44 (rejected: inliers on a strip) |
| 4-6 | 37 | 29 (rejected: inliers on a strip) |
| 5-6 | 76 | 180 |

Only a fifth to a third of the tentative matches are inliers even on correct pairs: reflections of the ceiling lights move with the camera and the pinned specimens show parallax, so they do not follow a single plane-to-plane transform.
