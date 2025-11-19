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
    name = "pose_tracking_sdk",
    srcs = [
        "pose_tracking_detect.cpp",
        "pose_tracking_sdk.cpp",
    ],
    hdrs = [
        "pose_tracking_detect.h",
        "pose_tracking_sdk.h",
        "types.h",
    ],
    data = [
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/pose_tracking:pose_tracking_cpu_deps",
    ],
    alwayslink = True,
)

cc_binary(
    name = "sdk_test",
    srcs = [
        "sdk_test.cpp",
    ],
    deps = [
        ":pose_tracking_sdk",
    ],
)
