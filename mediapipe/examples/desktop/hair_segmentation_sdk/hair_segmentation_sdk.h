#pragma once

#include "opencv2/opencv.hpp"

#ifdef _WIN32
#define EXPORT_SDK extern __declspec(dllexport)
#else
#define EXPORT_SDK extern
#endif

EXPORT_SDK std::string getHairSegErrorMessages();

EXPORT_SDK int initHairSeg();
EXPORT_SDK int initHairSeg(std::vector<std::string> model_paths);

EXPORT_SDK int releaseHairSeg();

EXPORT_SDK int getHairMask(const cv::Mat &img, cv::Mat &mask);
EXPORT_SDK int getHairMaskByImageMode(const cv::Mat &img, cv::Mat &mask);

#undef EXPORT_SDK
