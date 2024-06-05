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

class MMPGraph {
 private:
  mediapipe::CalculatorGraph graph;
  std::unique_ptr<mediapipe::OutputStreamPoller> poller;

 public:
  MMPGraph();
  ~MMPGraph();

  absl::Status InitMPPGraph(int face_max_num);

  absl::Status ReleaseMPPGraph();

  absl::Status RunMPPGraph(const cv::Mat &img, std::vector<FaceInfo> &faces);
};
