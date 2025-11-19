#pragma once

#ifndef POSETRACKING_TYPE_H
#define POSETRACKING_TYPE_H

#include <vector>

#include "opencv2/opencv.hpp"

struct PoseInfo {
  cv::Rect roi;
  std::vector<cv::Point3f> landmarks;
  cv::Mat segmentation_mask;

  PoseInfo()
      : landmarks(33) {}
};

#endif  // POSETRACKING_TYPE_H
