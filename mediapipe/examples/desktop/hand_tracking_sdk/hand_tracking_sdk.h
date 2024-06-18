#pragma once

#include "opencv2/opencv.hpp"
#include "types.h"

#ifdef _WIN32
#define EXPORT_SDK extern __declspec(dllexport)
#else
#define EXPORT_SDK extern
#endif

EXPORT_SDK void initHandTracking();
EXPORT_SDK void initHandTracking(std::vector<std::string> model_paths);

EXPORT_SDK void releaseHandTracking();

EXPORT_SDK void getHandLandmark(const cv::Mat &img,
                                std::vector<HandInfo> &hands);

#undef EXPORT_SDK
