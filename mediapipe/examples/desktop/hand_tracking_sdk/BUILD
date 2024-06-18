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

cc_binary(
    name = "hand_tracking_sdk",
    srcs = [
        "hand_tracking_detect.cpp",
        "hand_tracking_detect.h",
        "hand_tracking_sdk.cpp",
        "hand_tracking_sdk.h",
        "hand_tracking_sdk_c.cpp",
        "hand_tracking_sdk_c.h",
        "types.h",
    ],
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    linkshared = True,
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:packet_inner_join_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_renderer_cpu",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "sdk_test",
    srcs = [
        "hand_tracking_detect.cpp",
        "hand_tracking_detect.h",
        "hand_tracking_sdk.cpp",
        "hand_tracking_sdk.h",
        "sdk_test.cpp",
        "types.h",
    ],
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:packet_inner_join_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "//mediapipe/graphs/hand_tracking/subgraphs:hand_renderer_cpu",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)

cc_binary(
    name = "hand_tracking_test",
    srcs = [
        "test.cpp",
    ],
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
    ],
)
