# Third-party notices

Tessera uses the following software and models. Their licenses apply to the corresponding components.

| Component | Use in Tessera | License |
|---|---|---|
| [OpenCV](https://opencv.org) 5.0.0 | SIFT, robust model fitting (USAC / MAGSAC++), bundle adjustment, warping, exposure compensation, seam finding and multi-band blending; linked statically into the app | Apache-2.0 |
| Carotene, OpenCV's ARM NEON HAL (NVIDIA) | optimised image routines inside OpenCV | BSD-3-Clause |
| OpenCV's bundle adjusters (Intel, Willow Garage) | a modified copy of their optimisation loop and Jacobians in `interruptible_adjuster.hpp`, so that the alignment reports its progress and stops when asked | BSD-3-Clause |
| [ONNX Runtime](https://onnxruntime.ai) | optional, only in builds with the `ONNXRuntime` trait: ONNX keypoint selection and CPU backends for the learned models; not in the released app | MIT |
| [RaCo](https://github.com/cvg/RaCo) (ETH Zurich, Computer Vision and Geometry group) | keypoint detector and ranker weights | Apache-2.0 |
| [ALIKED](https://github.com/Shiaoming/ALIKED) | descriptor network weights | BSD-3-Clause |
| [LightGlue](https://github.com/cvg/LightGlue) (ETH Zurich, Computer Vision and Geometry group) | matcher architecture and weights for RaCo-ALIKED | Apache-2.0 |
| [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) | export-friendly model code used by `tools/export` | Apache-2.0 |
| Apple Core ML, Accelerate | on-device inference (on the GPU through Core ML), linear algebra and image resampling (vImage) | system frameworks |

The models that Tessera downloads (`tessera-models-*.zip` in the releases) are converted from the RaCo, ALIKED and LightGlue weights with the scripts in `tools/export`. The conversion splits the networks and rewrites a few layers as equivalent matrix products; it does not retrain the weights. [Licenses/models.md](Licenses/models.md) lists which file comes from which weights; the archive contains it as `LICENSES.md`, with the Apache License 2.0.

The app contains the license texts in `Tessera.app/Contents/Resources/Licenses`: Tessera's MIT License, this file, the Apache License 2.0, [Licenses/opencv.md](Licenses/opencv.md) (OpenCV's copyright notice, Carotene's license and that of the copied bundle adjusters) and the licenses of the third-party code inside the OpenCV modules, as OpenCV's build installs them. The other license texts are in the linked repositories. Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0. BSD-3-Clause: https://opensource.org/license/bsd-3-clause. MIT: https://opensource.org/license/mit.

## References

- Brown, M. and Lowe, D. G. *Automatic Panoramic Image Stitching using Invariant Features.* IJCV 2007.
- Moisan, L., Moulon, P. and Monasse, P. *Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers.* IPOL 2012.
- Barath, D., Noskova, J., Ivashechkin, M. and Matas, J. *MAGSAC++, a fast, reliable and accurate robust estimator.* CVPR 2020.
- Arandjelović, R. and Zisserman, A. *Three things everyone should know to improve object retrieval* (RootSIFT). CVPR 2012.
- Lindenberger, P., Sarlin, P.-E. and Pollefeys, M. *LightGlue: Local Feature Matching at Light Speed.* ICCV 2023.
- Zhao, X. et al. *ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation.* IEEE TIM 2023.
- Burt, P. J. and Adelson, E. H. *A Multiresolution Spline With Application to Image Mosaics.* ACM Transactions on Graphics 1983.
- Kwatra, V., Schödl, A., Essa, I., Turk, G. and Bobick, A. *Graphcut Textures: Image and Video Synthesis Using Graph Cuts.* SIGGRAPH 2003.
- Uyttendaele, M., Eden, A. and Szeliski, R. *Eliminating Ghosting and Exposure Artifacts in Image Mosaics.* CVPR 2001.
