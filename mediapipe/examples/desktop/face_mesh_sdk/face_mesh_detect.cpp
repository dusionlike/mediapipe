#include "face_mesh_detect.h"

#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/util/global_model_path_map.h"

constexpr char kInputStream[] = "input_video";

const int map_478_to_68[] = {
    234, 93,  132, 58,  172, 136, 150, 176, 152, 400, 379, 365,
    397, 288, 361, 323, 454,                                     // 脸部边缘
    70,  63,  105, 66,  107, 336, 296, 334, 293, 300,            // 眉毛
    168, 197, 5,   4,                                            // 鼻子中线
    64,  99,  2,   328, 294,                                     // 鼻子下部
    33,  160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  // 眼睛
    61,  0,   40,  37,  267, 270, 291, 321, 405, 17,  181, 91,
    78,  81,  13,  311, 308, 402, 14,  178  // 嘴巴
};

FaceMeshMMPGraph::FaceMeshMMPGraph() {}
FaceMeshMMPGraph::~FaceMeshMMPGraph() {}

absl::Status FaceMeshMMPGraph::InitMPPGraph(int num_faces,
                                            bool with_attention) {
  std::vector<std::string> model_paths = {
      "models/face_detection_short_range.tflite",
      with_attention ? "models/face_landmark_with_attention.tflite"
                     : "models/face_landmark.tflite",
  };
  return InitMPPGraph(model_paths, num_faces, with_attention);
}

absl::Status FaceMeshMMPGraph::InitMPPGraph(
    std::vector<std::string> model_paths, int num_faces, bool with_attention) {
  if (model_paths.size() != 2) {
    return absl::InvalidArgumentError("model_paths should contain 2 elements");
  }
  mediapipe::GlobalModelPathMap::Add(
      "mediapipe/modules/face_detection/face_detection_short_range.tflite",
      model_paths[0]);
  if (with_attention) {
    mediapipe::GlobalModelPathMap::Add(
        "mediapipe/modules/face_landmark/face_landmark_with_attention.tflite",
        model_paths[1]);
  } else {
    mediapipe::GlobalModelPathMap::Add(
        "mediapipe/modules/face_landmark/face_landmark.tflite", model_paths[1]);
  }

  // 修改配置文件中的最大人脸数
  absl::StrReplaceAll({{"$NUM_FACES", std::to_string(num_faces)}},
                      &calculator_graph_config_contents);
  // 修改配置中的是否使用注意力模型
  absl::StrReplaceAll({{"$WITH_ATTENTION", with_attention ? "true" : "false"}},
                      &calculator_graph_config_contents);

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ASSIGN_OR_RETURN(auto face_landmarks_poller,
                   graph.AddOutputStreamPoller("multi_face_landmarks", true));

  face_landmarks_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(face_landmarks_poller));

  MP_RETURN_IF_ERROR(graph.StartRun({}));
  return absl::OkStatus();
}

absl::Status FaceMeshMMPGraph::ReleaseMPPGraph() {
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

absl::Status FaceMeshMMPGraph::RunMPPGraphByImageMode(
    const cv::Mat &img, std::vector<FaceInfo> &faces) {
  auto res = RunMPPGraph(img, faces);

  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());

  return res;
}

absl::Status FaceMeshMMPGraph::RunMPPGraph(const cv::Mat &ori_img,
                                           std::vector<FaceInfo> &faces) {
  if (graph.GraphInputStreamsClosed()) {
    MP_RETURN_IF_ERROR(graph.StartRun({}));
  }

  cv::Mat img;
  cv::cvtColor(ori_img, img, cv::COLOR_BGR2RGB);
  // resize_image(ori_img, img, std::max(ori_img.cols, ori_img.rows));
  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, img.cols, img.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  img.copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us))));

  mediapipe::Packet landmarks_packet;
  if (face_landmarks_poller_->Next(&landmarks_packet) &&
      !landmarks_packet.IsEmpty()) {
    const std::vector<mediapipe::NormalizedLandmarkList> &landmarks =
        landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    std::vector<std::vector<cv::Point>> cv_landmarks;
    std::vector<std::vector<cv::Point3f>> cv_landmarks_3d;

    std::vector<cv::Rect> boxs;
    std::vector<float> scores;

    for (const auto &landmark : landmarks) {
      cv::Rect box;
      float score = 90;

      int landmark_size = landmark.landmark_size();

      std::vector<cv::Point> cv_landmark(landmark_size);
      std::vector<cv::Point3f> cv_landmark_3d(landmark_size);
      for (size_t i = 0; i < landmark_size; i++) {
        const auto &point = landmark.landmark(i);
        cv_landmark[i] = cv::Point(point.x() * img.cols, point.y() * img.rows);
        cv_landmark_3d[i] = cv::Point3f(point.x(), point.y(), point.z());
      }
      cv_landmarks.push_back(cv_landmark);
      cv_landmarks_3d.push_back(cv_landmark_3d);
      box = cv::boundingRect(cv_landmark);
      boxs.push_back(box);
      scores.push_back(score);
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxs, scores, 0.5, 0.45, nms_result);

    for (auto &&i : nms_result) {
      FaceInfo face;
      face.roi = boxs[i];
      face.score = scores[i];
      auto landmark = cv_landmarks[i];
      auto landmark_3d = cv_landmarks_3d[i];
      for (size_t i = 0; i < landmark.size(); i++) {
        face.landmarks[i] = landmark_3d[i];
      }

      // 478 -> 68
      for (int i = 0; i < 68; ++i) {
        face.landmarks68[i] = landmark[map_478_to_68[i]];
      }

      for (size_t i = 468; i < 5; i++) {
        face.left_iris_landmarks[i] = landmark[i];
      }
      for (size_t i = 473; i < 5; i++) {
        face.right_iris_landmarks[i] = landmark[i];
      }

      faces.push_back(face);
    }
  }

  return absl::OkStatus();
}
