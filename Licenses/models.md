# Licenses of the learned models

Tessera's learned models are converted from published pretrained weights with the scripts in `tools/export` of [Tessera](https://github.com/Camponotus-vagus/Tessera). These files were changed from the originals: the networks were converted to Core ML and ONNX, split into parts, and a few layers were rewritten as equivalent operations. The weights were not retrained.

| Files | Converted from | License |
|---|---|---|
| `raco_aliked_levels_*.mlpackage` (score and ranker maps), `raco_select_*.onnx` | [RaCo](https://github.com/cvg/RaCo) weights (`raco.pth`), Copyright 2023 ETH Zurich | Apache License 2.0 |
| `raco_aliked_levels_*.mlpackage` (feature levels), `aliked_descriptor_head.bin` | [ALIKED](https://github.com/Shiaoming/ALIKED) weights (`aliked-n16.pth`), Copyright (c) 2022, Zhao Xiaoming | BSD 3-Clause |
| `lightglue_raco_aliked_k2048_fp16.mlpackage` | [LightGlue](https://github.com/cvg/LightGlue) weights for RaCo-ALIKED (`raco_aliked_lightglue.pth`), Copyright 2023 ETH Zurich | Apache License 2.0 |

The text of the Apache License 2.0 is in `Apache-2.0.txt`.

## ALIKED

BSD 3-Clause License

Copyright (c) 2022, Zhao Xiaoming
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
