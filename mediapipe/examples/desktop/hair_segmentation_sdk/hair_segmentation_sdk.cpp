

#include "hair_segmentation_sdk.h"

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "hair_segmentation.h"
#include "mediapipe/framework/port/status.h"

static MMPGraph *graph = nullptr;

static std::string last_error_message;

std::string getHairSegErrorMessages() { return last_error_message; }

int initHairSeg() { return initHairSeg({}); }

int initHairSeg(std::vector<std::string> model_paths) {
  google::InitGoogleLogging("HairSegSDK");
  graph = new MMPGraph();
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

int releaseHairSeg() {
  google::ShutdownGoogleLogging();
  absl::Status run_status = graph->ReleaseMPPGraph();
  delete graph;
  graph = nullptr;
  if (!run_status.ok()) {
    last_error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}

int getHairMask(const cv::Mat &img, cv::Mat &mask) {
  absl::Status run_status = graph->RunMPPGraph(img, mask);
  if (!run_status.ok()) {
    last_error_message = run_status.ToString();
    return run_status.raw_code();
  }
  return 0;
}
