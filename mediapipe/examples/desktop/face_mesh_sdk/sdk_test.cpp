#include <iostream>

#include "face_mesh_sdk.h"

#define USE_CAMERA 1

int main() {
  try {
    initFaceLandmark(1, true);

#ifdef USE_CAMERA
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
      std::cout << "Error: cannot open camera" << std::endl;
      return -1;
    }
    cv::Mat frame;
    while (true) {
      cap >> frame;
#else
    for (size_t i = 0; i <= 7; i++) {
      std::string filename = "data/" + std::to_string(i) + ".jpg";
      cv::Mat frame = cv::imread(filename);
#endif
      if (frame.empty()) {
        std::cout << "Error: frame is empty" << std::endl;
        break;
      }
      std::vector<FaceInfo> faces;
      auto start = cv::getTickCount();
#ifdef USE_CAMERA
      int code = getFaceLandmark(frame, faces);
#else
      int code = getFaceLandmarkByImageMode(frame, faces);
#endif

      auto end = cv::getTickCount();
      std::cout << "Time: " << 1000 * (end - start) / cv::getTickFrequency()
                << "ms" << std::endl;

      if (code != 0) {
        std::cout << "Error: " << getFaceLandmarkErrorMessages() << std::endl;
        break;
      }

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
#ifdef USE_CAMERA
      cv::imshow("frame", frame);

      if (cv::waitKey(1) == 27) {
        break;
      }
#else
      cv::imwrite("data/res_" + std::to_string(i) + ".jpg", frame);
#endif
    }

    releaseFaceLandmark();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  return 0;
}