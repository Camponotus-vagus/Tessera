#!/bin/zsh
# Builds the minimal static OpenCV that Tessera links: core, imgproc, features, flann, geometry and
# stitching, for Apple Silicon and macOS 15, with Accelerate for LAPACK and no third-party downloads.
# Result: Vendor/opencv/{include,lib}. Needs cmake and ninja (brew install cmake ninja).
set -euo pipefail
root="${0:A:h:h}"
version=5.0.0
digest=b0528f5a1d379d59d4701cb28c36e22214cc51cf64594e5b56f2d3e6c0233095
deployment=15.0
src="$root/Vendor/src"
archive="$src/opencv-$version.tar.gz"
build="$root/Vendor/build/opencv-$version"
prefix="$root/Vendor/opencv"

mkdir -p "$src"
if [ ! -f "$archive" ] || [ "$(shasum -a 256 "$archive" | cut -d' ' -f1)" != "$digest" ]; then
  echo "downloading OpenCV $version source"
  curl -fL --progress-bar -o "$archive" "https://github.com/opencv/opencv/archive/refs/tags/$version.tar.gz"
fi
if [ "$(shasum -a 256 "$archive" | cut -d' ' -f1)" != "$digest" ]; then
  echo "checksum mismatch for $archive" >&2
  exit 1
fi
[ -d "$src/opencv-$version" ] || tar -xzf "$archive" -C "$src"

cmake -S "$src/opencv-$version" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$deployment" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_LIST=core,imgproc,features,flann,geometry,stitching \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF \
  -DBUILD_JAVA=OFF -DBUILD_OBJC=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python_bindings_generator=OFF \
  -DBUILD_opencv_js=OFF -DBUILD_ZLIB=OFF \
  -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_ITT=OFF -DWITH_EIGEN=OFF \
  -DWITH_PROTOBUF=OFF -DWITH_ADE=OFF -DWITH_KLEIDICV=OFF -DWITH_FASTCV=OFF -DWITH_OPENVINO=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_AVFOUNDATION=OFF -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_TIFF=OFF \
  -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF -DWITH_AVIF=OFF -DWITH_SPNG=OFF \
  -DWITH_IMGCODEC_HDR=OFF -DWITH_IMGCODEC_SUNRASTER=OFF -DWITH_IMGCODEC_PXM=OFF -DWITH_IMGCODEC_PFM=OFF \
  -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_VTK=OFF -DWITH_UNIFONT=OFF -DWITH_LAPACK=ON -DWITH_PTHREADS_PF=ON \
  -DOPENCV_GENERATE_PKGCONFIG=OFF -DINSTALL_CREATE_DISTRIB=ON
cmake --build "$build"
rm -rf "$prefix"
cmake --install "$build" > /dev/null
echo "OpenCV $version installed in $prefix"
