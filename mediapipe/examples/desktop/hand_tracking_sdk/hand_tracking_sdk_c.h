#pragma once

#include "opencv2/opencv.hpp"
#include "types.h"

#ifdef _WIN32
#define EXPORT_SDK __declspec(dllexport)
#else
#define EXPORT_SDK
#endif

extern "C" {
EXPORT_SDK int initHandTracking_C();
EXPORT_SDK int initHandTrackingWithModelPaths_C(const char **model_paths);

EXPORT_SDK int releaseHandTracking_C();

EXPORT_SDK int getHandLandmark_C(const unsigned char *data, int data_size,
                                 HandInfo_C *hands, int *output_hand_num);

EXPORT_SDK int getHandLandmarkErrorMessages_C(char *message, int message_size);
}

#undef EXPORT_SDK
