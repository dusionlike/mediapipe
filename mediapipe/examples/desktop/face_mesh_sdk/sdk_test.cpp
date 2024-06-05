#include <iostream>

#include "face_mesh_sdk.h"

int main() {
  int ret = initFaceLandmark(1);
  if (ret != 0) {
    std::cerr << "Failed to initialize face landmark." << std::endl;
    return -1;
  }

  for (size_t i = 0; i < 4; i++) {
    cv::Mat img = cv::imread("data/" + std::to_string(i) + ".jpg");
    std::vector<FaceInfo> faces;
    auto start = cv::getTickCount();
    ret = getFaceLandmarkFormMat(img, faces);
    auto end = cv::getTickCount();
    std::cout << "Time: " << (end - start) / cv::getTickFrequency() << "s"
              << std::endl;

    if (ret != 0) {
      std::cout << "Failed to get face landmark from Mat." << std::endl;
      std::cout << getFaceLandmarkErrorMessagesPP() << std::endl;
      return -1;
    }

    std::cout << "Face num: " << faces.size() << std::endl;

    for (auto &face : faces) {
      for (auto &landmark : face.landmarks68) {
        cv::circle(img, landmark, 2, cv::Scalar(0, 255, 0), -1);
      }
    }
    
    cv::imwrite("data/res_" + std::to_string(i) + ".jpg", img);
  }

  releaseFaceLandmark();
  return 0;
}