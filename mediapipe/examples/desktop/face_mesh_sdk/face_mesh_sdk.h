#pragma once

#ifdef __cplusplus
#include "opencv2/opencv.hpp"
#include "types.h"
#endif

struct FaceInfoC {
  int center_x;
  int center_y;
  int width;
  int height;
  float angle;
  float score;
  int landmarks[478][2];
  int landmarks68[68][2];
  int left_iris_landmarks[5][2];
  int right_iris_landmarks[5][2];
};

#ifdef _WIN32
#define EXPORT_SDK extern "C" __declspec(dllexport)
#define EXPORT_SDK_CPP extern __declspec(dllexport)
#else
#define EXPORT_SDK extern "C"
#define EXPORT_SDK_CPP extern
#endif

EXPORT_SDK int initFaceLandmark(int face_max_num);

EXPORT_SDK int releaseFaceLandmark();

EXPORT_SDK int getFaceLandmark(const unsigned char *data, int data_size,
                               FaceInfoC *faces_c, int *output_face_num);

EXPORT_SDK int getFaceLandmarkErrorMessages(char *message, int message_size);

#ifdef __cplusplus
EXPORT_SDK_CPP int getFaceLandmarkFormMat(const cv::Mat &img,
                                          std::vector<FaceInfo> &faces);
EXPORT_SDK_CPP std::string getFaceLandmarkErrorMessagesPP();
#endif

#undef EXPORT_SDK
#undef EXPORT_SDK_CPP
