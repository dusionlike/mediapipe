#include <iostream>

#include "hair_segmentation_sdk.h"

// #define USE_CAMERA 1

int main() {
  try {
    initHairSeg();

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
      cv::Mat mask;
      auto start = cv::getTickCount();
#ifdef USE_CAMERA
      int code = getHairMask(frame, faces);
#else
      int code = getHairMaskByImageMode(frame, mask);
#endif

      auto end = cv::getTickCount();
      std::cout << "Time: " << 1000 * (end - start) / cv::getTickFrequency()
                << "ms" << std::endl;

      if (code != 0) {
        std::cout << "Error: " << getHairSegErrorMessages() << std::endl;
        break;
      }

#ifdef USE_CAMERA
      cv::imshow("frame", frame);

      if (cv::waitKey(1) == 27) {
        break;
      }
#else
      cv::imwrite("data/mask_" + std::to_string(i) + ".jpg", mask);
#endif
    }

    releaseHairSeg();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  return 0;
}