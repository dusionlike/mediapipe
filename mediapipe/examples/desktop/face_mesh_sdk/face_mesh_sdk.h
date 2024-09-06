#pragma once

#include "opencv2/opencv.hpp"
#include "types.h"

#ifdef _WIN32
#define EXPORT_SDK extern __declspec(dllexport)
#else
#define EXPORT_SDK extern
#endif

EXPORT_SDK std::string getFaceLandmarkErrorMessages();

EXPORT_SDK int initFaceLandmark(int num_faces, bool with_attention);
EXPORT_SDK int initFaceLandmark(std::vector<std::string> model_paths,
                                 int num_faces, bool with_attention);

EXPORT_SDK int releaseFaceLandmark();

EXPORT_SDK int getFaceLandmark(const cv::Mat &img,
                                std::vector<FaceInfo> &faces);

EXPORT_SDK int getFaceLandmarkByImageMode(const cv::Mat &img,
                                std::vector<FaceInfo> &faces);

#undef EXPORT_SDK
