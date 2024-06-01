#include "face_mesh_detect.h"

#include "absl/strings/str_replace.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "multi_face_landmarks";

#define FACE_LANDMARKS 478

const int map_478_to_68[] = {
    234, 93,  132, 58,  172, 136, 150, 176, 152, 400, 379, 365,
    397, 288, 361, 323, 454,                                     // 脸部边缘
    70,  63,  105, 66,  107, 336, 296, 334, 293, 300,            // 眉毛
    168, 197, 5,   1,                                            // 鼻子中线
    64,  99,  2,   328, 294,                                     // 鼻子下部
    33,  160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  // 眼睛
    61,  0,   40,  37,  267, 270, 291, 321, 405, 17,  181, 91,
    78,  81,  13,  311, 308, 402, 14,  178  // 嘴巴
};

MMPGraph::MMPGraph() {}
MMPGraph::~MMPGraph() {}

absl::Status MMPGraph::InitMPPGraph(int face_max_num) {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(
      mediapipe::file::GetContents("models/face_mesh_desktop_landmarks.pbtxt",
                                   &calculator_graph_config_contents));

  // 修改配置文件中的最大人脸数
  absl::StrReplaceAll({{"$FACE_MAX_NUM", std::to_string(face_max_num)}},
                      &calculator_graph_config_contents);

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Start running the calculator graph.";
  // ASSIGN_OR_RETURN(poller, graph.AddOutputStreamPoller(kOutputStream));
  auto status_or = graph.AddOutputStreamPoller(kOutputStream);
  if (!status_or.ok()) {
    return status_or.status();
  }
  poller = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(status_or).value());
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  return absl::OkStatus();
}

absl::Status MMPGraph::ReleaseMPPGraph() {
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

absl::Status MMPGraph::RunMPPGraph(const cv::Mat &img,
                                   std::vector<FaceInfo> &faces) {
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
  if (poller->Next(&landmarks_packet)) {
    auto &landmarks =
        landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
    for (const auto &landmark : landmarks) {
      FaceInfo face;
      face.score = 1.0;
      for (size_t i = 0; i < landmark.landmark_size(); i++) {
        const auto &point = landmark.landmark(i);
        face.landmarks[i] =
            cv::Point(point.x() * img.cols, point.y() * img.rows);
      }

      // 478 -> 68
      for (int i = 0; i < 68; ++i) {
        face.landmarks68[i] = face.landmarks[map_478_to_68[i]];
      }

      for (size_t i = 468; i < 5; i++) {
        face.left_iris_landmarks[i] = face.landmarks[i];
      }
      for (size_t i = 473; i < 5; i++) {
        face.right_iris_landmarks[i] = face.landmarks[i];
      }

      faces.push_back(face);
    }
  }

  return absl::OkStatus();
}
