# OpenCV

The Tessera app contains parts of [OpenCV](https://opencv.org) 5.0.0, linked statically: the core, imgproc, features, flann, geometry and stitching modules, built by `tools/build-opencv.sh` without changes to the source. OpenCV is licensed under the Apache License 2.0, whose text is in `Apache-2.0.txt`. The other files in the `opencv` folder are the licenses of third-party code inside these modules, as installed by OpenCV's build.

```
Copyright (C) 2000-2022, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2023, OpenCV Foundation, all rights reserved.
Copyright (C) 2008-2016, Itseez Inc., all rights reserved.
Copyright (C) 2019-2023, Xperience AI, all rights reserved.
Copyright (C) 2019-2022, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
Copyright (C) 2022-2023, Southern University of Science And Technology, all rights reserved.
Copyright (C) 2023-2025, OpenCV AI, all rights reserved.
Copyright (C) 2024, Bigvision LLC, all rights reserved.

Third party copyrights are property of their respective owners.
```

## Carotene

The ARM NEON routines of OpenCV's Carotene HAL are included under this license:

```
License Agreement For libcarotene (3-clause BSD License)

Copyright (C) 2014-2015, NVIDIA Corporation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
```
