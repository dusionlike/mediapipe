#pragma once

#include "opencv2/opencv.hpp"
#include "types.h"

#ifdef _WIN32
#define EXPORT_SDK extern __declspec(dllexport)
#else
#define EXPORT_SDK extern
#endif

EXPORT_SDK std::string getHandTrackingErrorMessages();

EXPORT_SDK int initHandTracking();
EXPORT_SDK int initHandTracking(std::vector<std::string> model_paths);

EXPORT_SDK int releaseHandTracking();

EXPORT_SDK int getHandLandmark(const cv::Mat &img,
                                std::vector<HandInfo> &hands);

EXPORT_SDK int getHandLandmarkByImageMode(const cv::Mat &img,
                                           std::vector<HandInfo> &hands);

#undef EXPORT_SDK
