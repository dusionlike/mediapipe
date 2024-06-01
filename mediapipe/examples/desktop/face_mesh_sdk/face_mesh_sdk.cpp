#include "face_mesh_sdk.h"

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "face_mesh_detect.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

MMPGraph *graph = nullptr;
std::string error_message;

std::string getFaceLandmarkErrorMessagesPP() { return error_message; }

int getFaceLandmarkErrorMessages(char *message, int message_size) {
  if (error_message.size() > message_size) {
    return -1;
  }
  strcpy(message, error_message.c_str());
  return 0;
}

int initFaceLandmark(int face_max_num) {
  google::InitGoogleLogging("FaceMeshDetectSDK");
  graph = new MMPGraph();
  absl::Status run_status = graph->InitMPPGraph(face_max_num);
  if (!run_status.ok()) {
    error_message = run_status.message();
    return -1;
  }
  return 0;
}

int releaseFaceLandmark() {
  google::ShutdownGoogleLogging();
  absl::Status run_status = graph->ReleaseMPPGraph();
  if (!run_status.ok()) {
    error_message = run_status.message();
    return -1;
  }
  return 0;
}

int getFaceLandmarkFormMat(const cv::Mat &img, std::vector<FaceInfo> &faces) {
  try {
    absl::Status run_status = graph->RunMPPGraph(img, faces);
    if (!run_status.ok()) {
      error_message = run_status.message();
      return -1;
    }
    return 0;
  } catch (const std::exception &e) {
    error_message = e.what();
    return -1;
  }
}

int getFaceLandmark(const unsigned char *data, int data_size,
                    FaceInfoC *faces_c, int *output_face_num) {
  std::vector<uchar> buffer(data, data + data_size);
  cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);

  std::vector<FaceInfo> faces;
  int code = getFaceLandmarkFormMat(img, faces);
  if (code != 0) {
    return code;
  }

  try {
    *output_face_num = faces.size();
    for (int i = 0; i < faces.size(); i++) {
      faces_c[i].x = faces[i].roi.x;
      faces_c[i].y = faces[i].roi.y;
      faces_c[i].width = faces[i].roi.width;
      faces_c[i].height = faces[i].roi.height;
      faces_c[i].score = faces[i].score;

      for (int j = 0; j < faces[i].landmarks.size(); j++) {
        faces_c[i].landmarks[j][0] = faces[i].landmarks[j].x;
        faces_c[i].landmarks[j][1] = faces[i].landmarks[j].y;
      }

      for (int j = 0; j < faces[i].landmarks68.size(); j++) {
        faces_c[i].landmarks68[j][0] = faces[i].landmarks68[j].x;
        faces_c[i].landmarks68[j][1] = faces[i].landmarks68[j].y;
      }

      for (int j = 0; j < faces[i].left_iris_landmarks.size(); j++) {
        faces_c[i].left_iris_landmarks[j][0] =
            faces[i].left_iris_landmarks[j].x;
        faces_c[i].left_iris_landmarks[j][1] =
            faces[i].left_iris_landmarks[j].y;
        faces_c[i].right_iris_landmarks[j][0] =
            faces[i].right_iris_landmarks[j].x;
        faces_c[i].right_iris_landmarks[j][1] =
            faces[i].right_iris_landmarks[j].y;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    error_message = e.what();
    return -1;
  }
}
