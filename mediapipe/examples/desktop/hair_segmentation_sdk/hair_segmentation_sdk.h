#pragma once

#include "opencv2/opencv.hpp"

#ifdef _WIN32
#define EXPORT_SDK extern __declspec(dllexport)
#else
#define EXPORT_SDK extern
#endif

EXPORT_SDK void initHairSeg();
EXPORT_SDK void initHairSeg(std::vector<std::string> model_paths);

EXPORT_SDK void releaseHairSeg();

EXPORT_SDK void getHairMask(const cv::Mat &img, cv::Mat &mask);

#undef EXPORT_SDK
