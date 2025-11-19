#include <iostream>

#include "pose_tracking_sdk.h"

int main() {
  try {
    std::cout << "Initializing pose tracking..." << std::endl;
    int ret = initPoseTracking(true);
    if (ret != 0) {
      std::cerr << "Failed to initialize pose tracking: " 
                << getPoseTrackingErrorMessages() << std::endl;
      return -1;
    }
    std::cout << "Pose tracking initialized successfully!" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
      std::cout << "Error: cannot open camera" << std::endl;
      releasePoseTracking();
      return -1;
    }
    
    std::cout << "Camera opened successfully. Press ESC to exit." << std::endl;
    cv::Mat frame;
    while (true) {
      cap >> frame;
      if (frame.empty()) {
        std::cout << "Error: frame is empty" << std::endl;
        break;
      }
      cv::flip(frame, frame, /*flipcode=HORIZONTAL*/ 1);

      std::vector<PoseInfo> poses;
      auto start = cv::getTickCount();
      ret = getPoseTracking(frame, poses);
      auto end = cv::getTickCount();
      
      if (ret != 0) {
        std::cerr << "Failed to get pose tracking: " 
                  << getPoseTrackingErrorMessages() << std::endl;
        break;
      }
      std::cout << "Time: " << 1000 * (end - start) / cv::getTickFrequency()
                << "ms" << std::endl;

      std::cout << "Pose num: " << poses.size() << std::endl;

      for (auto &pose : poses) {
        // Apply segmentation mask if available
        if (!pose.segmentation_mask.empty()) {
          cv::Mat mask_resized;
          cv::resize(pose.segmentation_mask, mask_resized, frame.size());
          
          // Convert mask to 3 channels if needed
          cv::Mat mask_3ch;
          if (mask_resized.channels() == 1) {
            cv::cvtColor(mask_resized, mask_3ch, cv::COLOR_GRAY2BGR);
          } else {
            mask_3ch = mask_resized;
          }
          
          // Create colored overlay (green tint)
          cv::Mat overlay = frame.clone();
          cv::Mat colored_mask = cv::Mat::zeros(frame.size(), frame.type());
          colored_mask.setTo(cv::Scalar(0, 255, 0));
          
          // Apply mask to get only person region
          cv::Mat mask_binary;
          cv::threshold(mask_resized, mask_binary, 0.5, 1.0, cv::THRESH_BINARY);
          if (mask_binary.type() != CV_8UC1) {
            mask_binary.convertTo(mask_binary, CV_8UC1, 255);
          }
          
          // Blend the colored mask with original frame
          colored_mask.copyTo(overlay, mask_binary);
          cv::addWeighted(frame, 0.7, overlay, 0.3, 0, frame);
        }
        
        // Draw pose landmarks
        for (size_t i = 0; i < pose.landmarks.size(); ++i) {
          cv::Point3f pt = pose.landmarks[i];
          cv::circle(frame, cv::Point(pt.x, pt.y), 3, cv::Scalar(0, 255, 0), -1);
        }

        // Draw skeleton connections
        // Define pose connections based on MediaPipe Pose landmarks
        std::vector<std::pair<int, int>> connections = {
            // Face
            {0, 1}, {1, 2}, {2, 3}, {3, 7}, {0, 4}, {4, 5}, {5, 6}, {6, 8},
            // Torso
            {9, 10}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16},
            {11, 23}, {12, 24}, {23, 24},
            // Left arm
            {11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21}, {17, 19},
            // Right arm
            {12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22}, {18, 20},
            // Left leg
            {23, 25}, {25, 27}, {27, 29}, {29, 31}, {27, 31},
            // Right leg
            {24, 26}, {26, 28}, {28, 30}, {30, 32}, {28, 32}
        };

        for (auto &conn : connections) {
          if (conn.first < pose.landmarks.size() && conn.second < pose.landmarks.size()) {
            cv::Point3f pt1 = pose.landmarks[conn.first];
            cv::Point3f pt2 = pose.landmarks[conn.second];
            cv::line(frame, cv::Point(pt1.x, pt1.y), cv::Point(pt2.x, pt2.y), 
                     cv::Scalar(255, 0, 0), 2);
          }
        }
      }

      cv::imshow("Pose Tracking", frame);

      if (cv::waitKey(1) == 27) {
        break;
      }
    }

    releasePoseTracking();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }
  return 0;
}
