#pragma once

#ifndef HANDLANDMARK_TYPE_H
#define HANDLANDMARK_TYPE_H

#include <vector>

#include "opencv2/opencv.hpp"

struct HandInfo {
  std::vector<cv::Point> keypoints;
  // 'Left' | 'Right'
  std::string handedness;
  float score;
};

struct HandInfo_C {
  int keypoints[42];
  // 0: 'Left' , 1: 'Right'
  int handedness;
  float score;
};

#endif  // HANDLANDMARK_TYPE_H