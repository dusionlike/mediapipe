#pragma once

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
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

class HandTrackingMMPGraph {
 private:
  mediapipe::CalculatorGraph graph;
  std::unique_ptr<mediapipe::OutputStreamPoller> landmarks_poller_;
  std::unique_ptr<mediapipe::OutputStreamPoller> handedness_poller_;

  std::string calculator_graph_config_contents = R"(
# MediaPipe graph that performs hands tracking on desktop with TensorFlow
# Lite on CPU.
# Used in the example in
# mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu.

# CPU image. (ImageFrame)
input_stream: "input_video"

# Generates side packet cotaining max number of hands to detect/track.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:num_hands"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { int_value: 2 }
    }
  }
}

# Detects/tracks hand landmarks.
node {
  calculator: "HandLandmarkTrackingCpu"
  input_stream: "IMAGE:input_video"
  input_side_packet: "NUM_HANDS:num_hands"
  output_stream: "LANDMARKS:landmarks"
  output_stream: "HANDEDNESS:handedness"
  output_stream: "PALM_DETECTIONS:multi_palm_detections"
  output_stream: "HAND_ROIS_FROM_LANDMARKS:multi_hand_rects"
  output_stream: "HAND_ROIS_FROM_PALM_DETECTIONS:multi_palm_rects"
}
)";

 public:
  HandTrackingMMPGraph();
  ~HandTrackingMMPGraph();

  absl::Status InitMPPGraph();
  absl::Status InitMPPGraph(std::vector<std::string> model_paths);

  absl::Status ReleaseMPPGraph();

  absl::Status RunMPPGraph(const cv::Mat &img, std::vector<HandInfo> &hands);
  absl::Status RunMPPGraphByImageMode(const cv::Mat &img, std::vector<HandInfo> &hands);
};
