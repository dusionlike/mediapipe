#pragma once

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "types.h"

class PoseTrackingMMPGraph {
 private:
  mediapipe::CalculatorGraph graph;
  std::unique_ptr<mediapipe::OutputStreamPoller> pose_landmarks_poller_;
  std::unique_ptr<mediapipe::OutputStreamPoller> segmentation_mask_poller_;

  std::string calculator_graph_config_contents = R"(
# MediaPipe graph that performs pose tracking with TensorFlow Lite on CPU.

# Input image. (ImageFrame)
input_stream: "input_video"

# Pose landmarks. (NormalizedLandmarkList)
output_stream: "pose_landmarks"
# Segmentation mask. (ImageFrame)
output_stream: "segmentation_mask"

# Generates side packet to enable segmentation.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:enable_segmentation"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { bool_value: $ENABLE_SEGMENTATION }
    }
  }
}

# Throttles the images flowing downstream for flow control. It passes through
# the very first incoming image unaltered, and waits for downstream nodes
# (calculators and subgraphs) in the graph to finish their tasks before it
# passes through another image. All images that come in while waiting are
# dropped, limiting the number of in-flight images in most part of the graph to
# 1. This prevents the downstream nodes from queuing up incoming images and data
# excessively, which leads to increased latency and memory usage, unwanted in
# real-time mobile applications. It also eliminates unnecessarily computation,
# e.g., the output produced by a node may get dropped downstream if the
# subsequent nodes are still busy processing previous inputs.
node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:pose_landmarks"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

# Subgraph that detects poses and corresponding landmarks.
node {
  calculator: "PoseLandmarkCpu"
  input_side_packet: "ENABLE_SEGMENTATION:enable_segmentation"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "LANDMARKS:pose_landmarks"
  output_stream: "SEGMENTATION_MASK:segmentation_mask"
  output_stream: "DETECTION:pose_detection"
  output_stream: "ROI_FROM_LANDMARKS:roi_from_landmarks"
}
)";

 public:
  PoseTrackingMMPGraph();
  ~PoseTrackingMMPGraph();

  absl::Status InitMPPGraph(bool enable_segmentation);
  absl::Status InitMPPGraph(std::vector<std::string> model_paths, 
                            bool enable_segmentation);

  absl::Status ReleaseMPPGraph();

  absl::Status RunMPPGraph(const cv::Mat &img, std::vector<PoseInfo> &poses, 
                           bool is_image_mode = false);
  absl::Status RunMPPGraphByImageMode(const cv::Mat &img,
                                      std::vector<PoseInfo> &poses);
};
