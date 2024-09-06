#include "face_mesh_sdk.h"

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "face_mesh_detect.h"
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

static MMPGraph *graph = nullptr;

static std::string error_message;

std::string getFaceLandmarkErrorMessages() { return error_message; }

int initFaceLandmark(int num_faces, bool with_attention) {
  return initFaceLandmark({}, num_faces, with_attention);
}

int initFaceLandmark(std::vector<std::string> model_paths, int num_faces,
                     bool with_attention) {
  google::InitGoogleLogging("FaceMeshDetectSDK");
  graph = new MMPGraph();
  absl::Status run_status;
  if (model_paths.size() == 0) {
    run_status = graph->InitMPPGraph(num_faces, with_attention);
  } else {
    run_status = graph->InitMPPGraph(model_paths, num_faces, with_attention);
  }
  if (!run_status.ok()) {
    error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int releaseFaceLandmark() {
  google::ShutdownGoogleLogging();
  absl::Status run_status = graph->ReleaseMPPGraph();
  delete graph;
  graph = nullptr;
  if (!run_status.ok()) {
    error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int getFaceLandmark(const cv::Mat &img, std::vector<FaceInfo> &faces) {
  absl::Status run_status = graph->RunMPPGraph(img, faces);
  if (!run_status.ok()) {
    error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int getFaceLandmarkByImageMode(const cv::Mat &img,
                               std::vector<FaceInfo> &faces) {
  absl::Status run_status = graph->RunMPPGraphByImageMode(img, faces);
  if (!run_status.ok()) {
    error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}
