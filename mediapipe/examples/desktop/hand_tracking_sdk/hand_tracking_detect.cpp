#include "hand_tracking_detect.h"

#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/util/global_model_path_map.h"

constexpr char kInputStream[] = "input_video";

MMPGraph::MMPGraph() {}
MMPGraph::~MMPGraph() {}

absl::Status MMPGraph::InitMPPGraph() {
  std::vector<std::string> model_paths = {
      "models/palm_detection_full.tflite",
      "models/hand_landmark_full.tflite",
      "models/handedness.txt",
  };
  return InitMPPGraph(model_paths);
}

absl::Status MMPGraph::InitMPPGraph(std::vector<std::string> model_paths) {
  if (model_paths.size() != 3) {
    return absl::InvalidArgumentError("model_paths should contain 3 elements");
  }
  mediapipe::GlobalModelPathMap::Add(
      "mediapipe/modules/palm_detection/palm_detection_full.tflite",
      model_paths[0]);
  mediapipe::GlobalModelPathMap::Add(
      "mediapipe/modules/hand_landmark/hand_landmark_full.tflite",
      model_paths[1]);
  mediapipe::GlobalModelPathMap::Add(
      "mediapipe/modules/hand_landmark/handedness.txt",
      model_paths[2]);

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ASSIGN_OR_RETURN(auto landmarks_poller,
                   graph.AddOutputStreamPoller("landmarks", true));
  ASSIGN_OR_RETURN(auto handedness_poller,
                   graph.AddOutputStreamPoller("handedness", true));
  landmarks_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(landmarks_poller));

  handedness_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(handedness_poller));

  MP_RETURN_IF_ERROR(graph.StartRun({}));
  return absl::OkStatus();
}

absl::Status MMPGraph::ReleaseMPPGraph() {
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

absl::Status MMPGraph::RunMPPGraph(const cv::Mat& ori_img,
                                   std::vector<HandInfo>& hands) {
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

  std::vector<std::vector<cv::Point>> all_cv_landmarks;

  mediapipe::Packet landmarks_packet;
  if (landmarks_poller_->Next(&landmarks_packet) &&
      !landmarks_packet.IsEmpty()) {
    auto& landmarks =
        landmarks_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    for (const auto& landmark : landmarks) {
      std::vector<cv::Point> cv_landmarks;
      cv::Scalar color(0, 255, 0);
      for (size_t i = 0; i < landmark.landmark_size(); i++) {
        const auto& point = landmark.landmark(i);
        cv_landmarks.push_back(
            cv::Point(point.x() * ori_img.cols, point.y() * ori_img.rows));
      }
      all_cv_landmarks.push_back(cv_landmarks);
    }
  }

  std::vector<mediapipe::ClassificationList> handedness;

  mediapipe::Packet handedness_packet;
  if (handedness_poller_->Next(&handedness_packet) &&
      !handedness_packet.IsEmpty()) {
    handedness =
        handedness_packet.Get<std::vector<mediapipe::ClassificationList>>();
  }

  for (int i = 0; i < all_cv_landmarks.size(); i++) {
    const std::vector<cv::Point>& landmarks = all_cv_landmarks[i];
    HandInfo hand;
    hand.keypoints = landmarks;
    hand.handedness = handedness[i].classification(0).label();
    hand.score = handedness[i].classification(0).score();
    hands.push_back(hand);
  }

  return absl::OkStatus();
}
