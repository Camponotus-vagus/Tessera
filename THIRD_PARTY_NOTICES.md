# Third-party notices

Tessera uses the following software and models. Their licenses apply to the corresponding components.

| Component | Use in Tessera | License |
|---|---|---|
| [OpenCV](https://opencv.org) 5 | SIFT, robust model fitting (USAC / MAGSAC++), image utilities | Apache-2.0 |
| [ONNX Runtime](https://onnxruntime.ai) | keypoint selection, CPU fallbacks for the learned models | MIT |
| [RaCo](https://github.com/cvg/RaCo) (ETH Zurich, Computer Vision and Geometry group) | keypoint detector and ranker weights | Apache-2.0 |
| [ALIKED](https://github.com/Shiaoming/ALIKED) | descriptor network weights | BSD-3-Clause |
| [LightGlue](https://github.com/cvg/LightGlue) (ETH Zurich, Computer Vision and Geometry group) | matcher architecture and weights for RaCo-ALIKED | Apache-2.0 |
| [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) | export-friendly model code used by `tools/export` | Apache-2.0 |
| Apple Core ML, Accelerate, Metal | on-device inference and linear algebra | system frameworks |

The models shipped with Tessera (`Models/`) are converted from the RaCo, ALIKED and LightGlue weights with the scripts in `tools/export`. The conversion splits the networks and rewrites a few layers as equivalent matrix products; it does not retrain or alter the weights.

The full license texts are available in the linked repositories. Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0. BSD-3-Clause: https://opensource.org/license/bsd-3-clause. MIT: https://opensource.org/license/mit.

## References

- Brown, M. and Lowe, D. G. *Automatic Panoramic Image Stitching using Invariant Features.* IJCV 2007.
- Moisan, L., Moulon, P. and Monasse, P. *Automatic Homographic Registration of a Pair of Images, with A Contrario Elimination of Outliers.* IPOL 2012.
- Barath, D., Noskova, J., Ivashechkin, M. and Matas, J. *MAGSAC++, a fast, reliable and accurate robust estimator.* CVPR 2020.
- Arandjelović, R. and Zisserman, A. *Three things everyone should know to improve object retrieval* (RootSIFT). CVPR 2012.
- Lindenberger, P., Sarlin, P.-E. and Pollefeys, M. *LightGlue: Local Feature Matching at Light Speed.* ICCV 2023.
- Zhao, X. et al. *ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation.* IEEE TIM 2023.
