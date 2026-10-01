# MIT License
#
# Copyright (c) 2024 Tiziano Guadagnino, Benedikt Mersch, Ignacio Vizzo, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# Silence timestamp warning
if(CMAKE_VERSION VERSION_GREATER 3.24)
  cmake_policy(SET CMP0135 OLD)
endif()

include(FetchContent)
# ALICE M2: build offline from the tarballs vendored in ../3rdparty (the robot
# has no guaranteed internet access). The first FetchContent_Declare() of a
# name wins, so Sophus and robin-map are declared here, ahead of the
# declarations inside kiss_icp's own 3rdparty scripts, with the same version,
# options and patch those scripts use. Eigen3 and TBB come from the system.
set(KINEMATIC_ICP_VENDOR_DIR ${CMAKE_CURRENT_LIST_DIR}/../3rdparty)
set(SOPHUS_USE_BASIC_LOGGING ON CACHE BOOL "Don't use fmt for Sophus libraru")
set(BUILD_SOPHUS_TESTS OFF CACHE BOOL "Don't build Sophus tests")
set(BUILD_SOPHUS_EXAMPLES OFF CACHE BOOL "Don't build Sophus Examples")
FetchContent_Declare(sophus SYSTEM URL ${KINEMATIC_ICP_VENDOR_DIR}/sophus-1.22.10.tar.gz
                     PATCH_COMMAND patch -p1 < ${KINEMATIC_ICP_VENDOR_DIR}/sophus.patch UPDATE_DISCONNECTED 1)
FetchContent_Declare(tessil SYSTEM URL ${KINEMATIC_ICP_VENDOR_DIR}/robin-map-1.2.1.tar.gz)
FetchContent_Declare(kiss_icp URL ${KINEMATIC_ICP_VENDOR_DIR}/kiss-icp-1.2.0.tar.gz SOURCE_SUBDIR cpp/kiss_icp)
FetchContent_MakeAvailable(kiss_icp)
