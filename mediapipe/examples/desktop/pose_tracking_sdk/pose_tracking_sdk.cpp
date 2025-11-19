#include "pose_tracking_sdk.h"

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "pose_tracking_detect.h"
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

static PoseTrackingMMPGraph *graph = nullptr;

static std::string error_message;

std::string getPoseTrackingErrorMessages() { return error_message; }

int initPoseTracking(bool enable_segmentation) {
  return initPoseTracking({}, enable_segmentation);
}

int initPoseTracking(std::vector<std::string> model_paths, 
                     bool enable_segmentation) {
  graph = new PoseTrackingMMPGraph();
  absl::Status run_status;
  if (model_paths.size() == 0) {
    run_status = graph->InitMPPGraph(enable_segmentation);
  } else {
    run_status = graph->InitMPPGraph(model_paths, enable_segmentation);
  }
  if (!run_status.ok()) {
    error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int releasePoseTracking() {
  absl::Status run_status = graph->ReleaseMPPGraph();
  delete graph;
  graph = nullptr;
  if (!run_status.ok()) {
    error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int getPoseTracking(const cv::Mat &img, std::vector<PoseInfo> &poses) {
  absl::Status run_status = graph->RunMPPGraph(img, poses);
  if (!run_status.ok()) {
    error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int getPoseTrackingByImageMode(const cv::Mat &img,
                               std::vector<PoseInfo> &poses) {
  absl::Status run_status = graph->RunMPPGraphByImageMode(img, poses);
  if (!run_status.ok()) {
    error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}
