#pragma once

struct FaceInfoC {
  int x;
  int y;
  int width;
  int height;
  float score;
  int landmarks[478][2];
  int landmarks68[68][2];
  int left_iris_landmarks[5][2];
  int right_iris_landmarks[5][2];
};

#ifdef _WIN32
#define EXPORT_SDK extern "C" __declspec(dllexport)
#else
#define EXPORT_SDK extern "C"
#endif

EXPORT_SDK int initFaceLandmark_C(int face_max_num);
EXPORT_SDK int initFaceLandmarkWithModelPaths_C(const char **model_paths,
                                                int face_max_num);

EXPORT_SDK int releaseFaceLandmark_C();

EXPORT_SDK int getFaceLandmark_C(const unsigned char *data, int data_size,
                                 FaceInfoC *faces_c, int *output_face_num);

EXPORT_SDK int getFaceLandmarkErrorMessages_C(char *message, int message_size);

#undef EXPORT_SDK
