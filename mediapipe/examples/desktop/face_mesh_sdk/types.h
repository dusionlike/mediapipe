#pragma once

#ifndef FACELANDMARK_TYPE_H
#define FACELANDMARK_TYPE_H

#include <vector>

#include "opencv2/opencv.hpp"

struct FaceInfo {
  cv::Rect roi;
  float score;
  std::vector<cv::Point> landmarks;
  std::vector<cv::Point> landmarks68;
  std::vector<cv::Point> left_iris_landmarks;
  std::vector<cv::Point> right_iris_landmarks;

  FaceInfo()
      : score(0.0),
        landmarks(478),
        landmarks68(68),
        left_iris_landmarks(5),
        right_iris_landmarks(5) {}
};

#endif  // FACELANDMARK_TYPE_H