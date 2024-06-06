#include "face_mesh_detect.h"

#include "absl/strings/str_replace.h"

constexpr char kInputStream[] = "input_video";

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

// /**
//  * 将图片转换为指定大小的正方形，并且保持原始图片的长宽比不变形，不足的地方用黑色填充
//  */
// float resize_image(const cv::Mat &img, cv::Mat &output, int size) {
//   cv::Size target_size(size, size);
//   cv::Mat padded_img(target_size, img.type(), cv::Scalar(0));
//   float scale = std::min(static_cast<float>(target_size.width) / img.cols,
//                          static_cast<float>(target_size.height) / img.rows);
//   cv::Size new_size = cv::Size(img.cols * scale, img.rows * scale);
//   cv::Mat resized;
//   cv::resize(img, resized, new_size);
//   cv::Rect roi(0, 0, resized.cols, resized.rows);
//   resized.copyTo(padded_img(roi));
//   output = padded_img;
//   return scale;
// }

extern std::string calculator_graph_config_contents;

MMPGraph::MMPGraph() {}
MMPGraph::~MMPGraph() {}

absl::Status MMPGraph::InitMPPGraph(int face_max_num) {
  // 修改配置文件中的最大人脸数
  absl::StrReplaceAll({{"$FACE_MAX_NUM", std::to_string(face_max_num)}},
                      &calculator_graph_config_contents);

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(auto face_count_poller,
                   graph.AddOutputStreamPoller("face_count"));
  ASSIGN_OR_RETURN(auto face_detections_poller,
                   graph.AddOutputStreamPoller("face_detections"));
  ASSIGN_OR_RETURN(auto face_rects_poller,
                   graph.AddOutputStreamPoller("face_rects_from_detections"));

  ASSIGN_OR_RETURN(auto face_landmarks_poller,
                   graph.AddOutputStreamPoller("multi_face_landmarks"));
  face_count_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(face_count_poller));
  face_detections_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(face_detections_poller));
  face_rects_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(face_rects_poller));

  face_landmarks_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(face_landmarks_poller));

  MP_RETURN_IF_ERROR(graph.StartRun({}));
  return absl::OkStatus();
}

absl::Status MMPGraph::ReleaseMPPGraph() {
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

absl::Status MMPGraph::RunMPPGraph(const cv::Mat &ori_img,
                                   std::vector<FaceInfo> &faces) {
  const cv::Mat &img = ori_img;
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
  mediapipe::Packet face_count_packet;
  if (face_count_poller_->Next(&face_count_packet)) {
    const int &face_count = face_count_packet.Get<int>();
    if (face_count == 0) {
      return absl::OkStatus();
    }
  }

  std::vector<cv::Rect2d> boxs;
  std::vector<float> scores;

  mediapipe::Packet face_detections_packet;
  if (face_detections_poller_->Next(&face_detections_packet)) {
    auto &detections =
        face_detections_packet.Get<std::vector<mediapipe::Detection>>();
    for (const auto &detection : detections) {
      cv::Rect2d box(
          detection.location_data().relative_bounding_box().xmin() * img.cols,
          detection.location_data().relative_bounding_box().ymin() * img.rows,
          detection.location_data().relative_bounding_box().width() * img.cols,
          detection.location_data().relative_bounding_box().height() *
              img.rows);
      boxs.push_back(box);
      scores.push_back(detection.score().Get(0));
    }
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxs, scores, 0.5, 0.45, nms_result);

  // 带角度的矩形框
  std::vector<cv::RotatedRect> res_rois;

  mediapipe::Packet face_rects_packet;
  if (face_rects_poller_->Next(&face_rects_packet)) {
    auto &rects =
        face_rects_packet.Get<std::vector<mediapipe::NormalizedRect>>();
    for (auto &&i : nms_result) {
      auto rect = rects[i];
      cv::RotatedRect roi(
          cv::Point2f(rect.x_center() * img.cols, rect.y_center() * img.rows),
          cv::Size2f(rect.width() * img.cols, rect.height() * img.rows),
          rect.rotation());

      res_rois.push_back(roi);
    }
  }

  mediapipe::Packet landmarks_packet;
  if (face_landmarks_poller_->Next(&landmarks_packet)) {
    auto &landmarks =
        landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    for (auto &&i : nms_result) {
      FaceInfo face;
      face.roi = res_rois[i];
      face.score = scores[i];
      auto landmark = landmarks[i];
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

std::string calculator_graph_config_contents = R"(
# MediaPipe graph that performs face mesh with TensorFlow Lite on CPU.

# Input image. (ImageFrame)
input_stream: "input_video"

# Output image with rendered results. (ImageFrame)
output_stream: "output_video"
output_stream: "face_count"
output_stream: "face_rects_from_detections"
output_stream: "face_detections"
output_stream: "face_rects_from_landmarks"
# Collection of detected/processed faces, each represented as a list of
# landmarks. (std::vector<NormalizedLandmarkList>)
output_stream: "multi_face_landmarks"

# Throttles the images flowing downstream for flow control. It passes through
# the very first incoming image unaltered, and waits for downstream nodes
# (calculators and subgraphs) in the graph to finish their tasks before it
# passes through another image. All images that come in while waiting are
# dropped, limiting the number of in-flight images in most part of the graph to
# 1. This prevents the downstream nodes from queuing up incoming images and data
# excessively, which leads to increased latency and memory usage, unwanted in
# real-time mobile applications. It also eliminates unnecessarily computation,
# e.g., the output produced by a node may get dropped downstream if the
# subsequent nodes are still busy processing previous inputs.
node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:output_video"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

# Defines side packets for further use in the graph.
node {
  calculator: "ConstantSidePacketCalculator"
  output_side_packet: "PACKET:0:num_faces"
  output_side_packet: "PACKET:1:with_attention"
  node_options: {
    [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
      packet { int_value: $FACE_MAX_NUM }
      packet { bool_value: true }
    }
  }
}

# Subgraph that detects faces and corresponding landmarks.
node {
  calculator: "FaceLandmarkFrontCpu"
  input_stream: "IMAGE:throttled_input_video"
  input_side_packet: "NUM_FACES:num_faces"
  input_side_packet: "WITH_ATTENTION:with_attention"
  output_stream: "LANDMARKS:multi_face_landmarks"
  output_stream: "ROIS_FROM_LANDMARKS:face_rects_from_landmarks"
  output_stream: "DETECTIONS:face_detections"
  output_stream: "ROIS_FROM_DETECTIONS:face_rects_from_detections"
  output_stream: "FACE_COUNT_FROM_LANDMARKS:face_count"
}

# Subgraph that renders face-landmark annotation onto the input image.
node {
  calculator: "FaceRendererCpu"
  input_stream: "IMAGE:throttled_input_video"
  input_stream: "LANDMARKS:multi_face_landmarks"
  input_stream: "NORM_RECTS:face_rects_from_landmarks"
  input_stream: "DETECTIONS:face_detections"
  output_stream: "IMAGE:output_video"
}
)";
