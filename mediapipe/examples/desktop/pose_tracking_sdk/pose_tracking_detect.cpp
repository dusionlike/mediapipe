#include "pose_tracking_detect.h"

#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/util/global_model_path_map.h"
#include "mediapipe/framework/port/status_macros.h"

constexpr char kInputStream[] = "input_video";

PoseTrackingMMPGraph::PoseTrackingMMPGraph() {}
PoseTrackingMMPGraph::~PoseTrackingMMPGraph() {}

absl::Status PoseTrackingMMPGraph::InitMPPGraph(bool enable_segmentation) {
  std::vector<std::string> model_paths = {
      "models/pose_detection.tflite",
      "models/pose_landmark_full.tflite",
  };
  return InitMPPGraph(model_paths, enable_segmentation);
}

absl::Status PoseTrackingMMPGraph::InitMPPGraph(
    std::vector<std::string> model_paths, bool enable_segmentation) {
  if (model_paths.size() != 2) {
    return absl::InvalidArgumentError("model_paths should contain 2 elements");
  }
  mediapipe::GlobalModelPathMap::Add(
      "mediapipe/modules/pose_detection/pose_detection.tflite",
      model_paths[0]);
  mediapipe::GlobalModelPathMap::Add(
      "mediapipe/modules/pose_landmark/pose_landmark_full.tflite",
      model_paths[1]);

  // 修改配置中的是否启用分割
  absl::StrReplaceAll({{"$ENABLE_SEGMENTATION", enable_segmentation ? "true" : "false"}},
                      &calculator_graph_config_contents);

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  MP_RETURN_IF_ERROR(graph.Initialize(config));

  MP_ASSIGN_OR_RETURN(auto pose_landmarks_poller,
                   graph.AddOutputStreamPoller("pose_landmarks", true));

  pose_landmarks_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
      std::move(pose_landmarks_poller));

  if (enable_segmentation) {
    MP_ASSIGN_OR_RETURN(auto segmentation_mask_poller,
                     graph.AddOutputStreamPoller("segmentation_mask", true));

    segmentation_mask_poller_ = absl::make_unique<mediapipe::OutputStreamPoller>(
        std::move(segmentation_mask_poller));
  }

  MP_RETURN_IF_ERROR(graph.StartRun({}));
  return absl::OkStatus();
}

absl::Status PoseTrackingMMPGraph::ReleaseMPPGraph() {
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

absl::Status PoseTrackingMMPGraph::RunMPPGraphByImageMode(
    const cv::Mat &img, std::vector<PoseInfo> &poses) {
  return RunMPPGraph(img, poses, true);
}

absl::Status PoseTrackingMMPGraph::RunMPPGraph(const cv::Mat &ori_img,
                                               std::vector<PoseInfo> &poses,
                                               bool is_image_mode) {
  if (graph.GraphInputStreamsClosed()) {
    MP_RETURN_IF_ERROR(graph.StartRun({}));
  }

  cv::Mat img;
  cv::cvtColor(ori_img, img, cv::COLOR_BGR2RGB);
  
  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, img.cols, img.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  img.copyTo(input_frame_mat);

  // Send image packet into the graph.
  if (is_image_mode) {
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                      .At(mediapipe::Timestamp(0))));
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  } else {
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));
  }

  // Get the graph result packet
  mediapipe::Packet pose_landmarks_packet;
  if (!pose_landmarks_poller_->Next(&pose_landmarks_packet)) {
    return absl::UnavailableError("Failed to get pose landmarks packet");
  }

  if (pose_landmarks_packet.IsEmpty()) {
    poses.clear();
    return absl::OkStatus();
  }

  const auto &pose_landmarks =
      pose_landmarks_packet.Get<mediapipe::NormalizedLandmarkList>();

  // Parse pose landmarks
  poses.clear();
  PoseInfo pose_info;

  for (int i = 0; i < pose_landmarks.landmark_size(); ++i) {
    const auto &landmark = pose_landmarks.landmark(i);
    cv::Point3f pt;
    pt.x = landmark.x() * img.cols;
    pt.y = landmark.y() * img.rows;
    pt.z = landmark.z();
    pose_info.landmarks[i] = pt;
  }

  // Get segmentation mask if available
  if (segmentation_mask_poller_) {
    mediapipe::Packet segmentation_mask_packet;
    if (segmentation_mask_poller_->Next(&segmentation_mask_packet)) {
      if (!segmentation_mask_packet.IsEmpty()) {
        const auto &segmentation_mask =
            segmentation_mask_packet.Get<mediapipe::ImageFrame>();
        cv::Mat mask_mat = mediapipe::formats::MatView(&segmentation_mask);
        mask_mat.copyTo(pose_info.segmentation_mask);
      }
    }
  }

  poses.push_back(pose_info);

  return absl::OkStatus();
}
