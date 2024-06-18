#include <iostream>

#include "hand_tracking_sdk.h"

int main() {
  try {
    initHandTracking();

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
      cv::flip(frame, frame, /*flipcode=HORIZONTAL*/ 1);

      std::vector<HandInfo> hands;
      auto start = cv::getTickCount();
      getHandLandmark(frame, hands);
      auto end = cv::getTickCount();
      std::cout << "Time: " << 1000 * (end - start) / cv::getTickFrequency()
                << "ms" << std::endl;

      std::cout << "Hand num: " << hands.size() << std::endl;

      for (auto &hand : hands) {
        for (auto &landmark : hand.keypoints) {
          cv::circle(frame, landmark, 2, cv::Scalar(0, 255, 0), -1);
        }
        cv::Rect rect = cv::boundingRect(hand.keypoints);
        cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
        cv::putText(frame, hand.handedness + " " + std::to_string(hand.score),
                    cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(255, 0, 0));
      }

      cv::imshow("frame", frame);

      if (cv::waitKey(1) == 27) {
        break;
      }
    }

    releaseHandTracking();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  return 0;
}