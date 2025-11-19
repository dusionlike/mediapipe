#pragma once

#include "opencv2/opencv.hpp"
#include "types.h"

#ifdef _WIN32
#define EXPORT_SDK extern __declspec(dllexport)
#else
#define EXPORT_SDK extern
#endif

EXPORT_SDK std::string getPoseTrackingErrorMessages();

EXPORT_SDK int initPoseTracking(bool enable_segmentation);
EXPORT_SDK int initPoseTracking(std::vector<std::string> model_paths,
                                 bool enable_segmentation);

EXPORT_SDK int releasePoseTracking();

EXPORT_SDK int getPoseTracking(const cv::Mat &img,
                                std::vector<PoseInfo> &poses);

EXPORT_SDK int getPoseTrackingByImageMode(const cv::Mat &img,
                                           std::vector<PoseInfo> &poses);

#undef EXPORT_SDK
