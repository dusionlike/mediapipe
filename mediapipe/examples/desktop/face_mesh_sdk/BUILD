# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_library(
    name = "face_mesh_sdk",
    srcs = [
        "face_mesh_detect.cpp",
        "face_mesh_sdk.cpp",
    ],
    hdrs = [
        "face_mesh_detect.h",
        "face_mesh_sdk.h",
        "types.h",
    ],
    data = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
    ],
    alwayslink = True,
)

cc_binary(
    name = "sdk_test",
    srcs = [
        "sdk_test.cpp",
    ],
    deps = [
        ":face_mesh_sdk",
    ],
)
