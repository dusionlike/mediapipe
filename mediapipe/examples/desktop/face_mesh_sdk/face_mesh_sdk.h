#pragma once

#include "opencv2/opencv.hpp"
#include "types.h"

#ifdef _WIN32
#define EXPORT_SDK extern __declspec(dllexport)
#else
#define EXPORT_SDK extern
#endif

EXPORT_SDK void initFaceLandmark(int num_faces, bool with_attention);
EXPORT_SDK void initFaceLandmark(std::vector<std::string> model_paths,
                                 int num_faces, bool with_attention);

EXPORT_SDK void releaseFaceLandmark();

EXPORT_SDK void getFaceLandmark(const cv::Mat &img,
                                std::vector<FaceInfo> &faces);

#undef EXPORT_SDK
