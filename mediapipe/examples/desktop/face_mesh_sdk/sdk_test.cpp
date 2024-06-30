#include <iostream>

#include "face_mesh_sdk.h"

int main() {
  try {
    initFaceLandmark(1);

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
      getFaceLandmark(frame, faces);
      auto end = cv::getTickCount();
      std::cout << "Time: " << 1000 * (end - start) / cv::getTickFrequency()
                << "ms" << std::endl;

      std::cout << "Face num: " << faces.size() << std::endl;

      for (auto &face : faces) {
        for (auto &landmark : face.landmarks68) {
          cv::circle(frame, landmark, 2, cv::Scalar(0, 255, 0), -1);
        }

        for (auto &landmark : face.left_iris_landmarks) {
          cv::circle(frame, landmark, 2, cv::Scalar(255, 0, 0), -1);
        }

        for (auto &landmark : face.right_iris_landmarks) {
          cv::circle(frame, landmark, 2, cv::Scalar(255, 0, 0), -1);
        }
      }

      cv::imshow("frame", frame);

      if (cv::waitKey(1) == 27) {
        break;
      }
    }

    releaseFaceLandmark();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  return 0;
}