

#include "hand_tracking_sdk_c.h"

#include "hand_tracking_sdk.h"

std::string error_message;

int getHandLandmarkErrorMessages_C(char *message, int message_size) {
  if (error_message.size() < message_size) {
    strcpy(message, error_message.c_str());
  }
  return error_message.size();
}

int initHandTracking_C() { return initHandTrackingWithModelPaths_C(nullptr); }

int initHandTrackingWithModelPaths_C(const char **model_paths) {
  try {
    if (model_paths == nullptr) {
      initHandTracking();
    } else {
      std::vector<std::string> model_paths_cpp;
      for (int i = 0; model_paths[i] != nullptr; i++) {
        model_paths_cpp.push_back(model_paths[i]);
      }
      initHandTracking(model_paths_cpp);
    }
  } catch (const std::exception &e) {
    error_message = e.what();
    return -1;
  }
  return 0;
}

int releaseHandTracking_C() {
  try {
    releaseHandTracking();
  } catch (const std::exception &e) {
    error_message = e.what();
    return -1;
  }
  return 0;
}

int getHandLandmark_C(const unsigned char *data, int data_size,
                      HandInfo_C *hands, int *output_hand_num) {
  try {
    // 读取data为cv::Mat
    cv::Mat img = cv::imdecode(cv::Mat(1, data_size, CV_8UC1, (void *)data),
                               cv::IMREAD_COLOR);

    std::vector<HandInfo> hands_cpp;
    getHandLandmark(img, hands_cpp);
    *output_hand_num = hands_cpp.size();
    for (int i = 0; i < hands_cpp.size(); i++) {
      hands[i].score = hands_cpp[i].score;
      hands[i].handedness = hands_cpp[i].handedness == "Left" ? 0 : 1;
      for (int j = 0; j < hands_cpp[i].keypoints.size(); j++) {
        hands[i].keypoints[j * 2] = hands_cpp[i].keypoints[j].x;
        hands[i].keypoints[j * 2 + 1] = hands_cpp[i].keypoints[j].y;
      }
    }
  } catch (const std::exception &e) {
    error_message = e.what();
    return -1;
  }
  return 0;
}
