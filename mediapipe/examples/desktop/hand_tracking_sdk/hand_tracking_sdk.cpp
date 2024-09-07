

#include "hand_tracking_sdk.h"

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "hand_tracking_detect.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

static HandTrackingMMPGraph *graph = nullptr;

static std::string last_error_message;

std::string getHandTrackingErrorMessages() { return last_error_message; }

int initHandTracking() { initHandTracking({}); }

int initHandTracking(std::vector<std::string> model_paths) {
  graph = new HandTrackingMMPGraph();
  absl::Status run_status;
  if (model_paths.size() == 0) {
    run_status = graph->InitMPPGraph();
  } else {
    run_status = graph->InitMPPGraph(model_paths);
  }
  if (!run_status.ok()) {
    last_error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int releaseHandTracking() {
  absl::Status run_status = graph->ReleaseMPPGraph();
  delete graph;
  graph = nullptr;
  if (!run_status.ok()) {
    last_error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int getHandLandmark(const cv::Mat &img, std::vector<HandInfo> &hands) {
  absl::Status run_status = graph->RunMPPGraph(img, hands);
  if (!run_status.ok()) {
    last_error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int getHandLandmarkByImageMode(const cv::Mat &img,
                               std::vector<HandInfo> &hands) {
  absl::Status run_status = graph->RunMPPGraphByImageMode(img, hands);
  if (!run_status.ok()) {
    last_error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}
