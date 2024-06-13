#include <iostream>

#include "face_mesh_sdk.h"

int main() {
  int ret = initFaceLandmark(1);
  if (ret != 0) {
    std::cerr << "Failed to initialize face landmark." << std::endl;
    return -1;
  }

  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cout << "Error: cannot open camera" << std::endl;
    return -1;
  }
  cv::Mat frame;
  while (true) {
    cap >> frame;
    if (frame.empty()) {
      std::cout << "Error: frame is empty" << std::endl;
      break;
    }
    std::vector<FaceInfo> faces;
    auto start = cv::getTickCount();
    ret = getFaceLandmarkFormMat(frame, faces);
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
        cv::circle(frame, landmark, 2, cv::Scalar(0, 255, 0), -1);
      }
    }

    cv::imshow("frame", frame);

    if (cv::waitKey(1) == 27) {
      break;
    }
  }

  releaseFaceLandmark();
  return 0;
}