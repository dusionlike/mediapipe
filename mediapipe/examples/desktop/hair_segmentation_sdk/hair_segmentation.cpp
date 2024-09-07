#include "hair_segmentation.h"

#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/util/global_model_path_map.h"

constexpr char kInputStream[] = "input_video";

HairSegMMPGraph::HairSegMMPGraph() {}
HairSegMMPGraph::~HairSegMMPGraph() {}

absl::Status HairSegMMPGraph::InitMPPGraph() {
  std::vector<std::string> model_paths = {
      "models/hair_segmentation.tflite",
  };
  return InitMPPGraph(model_paths);
}

absl::Status HairSegMMPGraph::InitMPPGraph(
    std::vector<std::string> model_paths) {
  if (model_paths.size() != 1) {
    return absl::InvalidArgumentError("model_paths should contain 1 elements");
  }
  mediapipe::GlobalModelPathMap::Add(
      "mediapipe/models/hair_segmentation.tflite", model_paths[0]);

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ASSIGN_OR_RETURN(auto mask_poller,
                   graph.AddOutputStreamPoller("hair_mask", true));
  mask_poller_ =
      absl::make_unique<mediapipe::OutputStreamPoller>(std::move(mask_poller));

  MP_RETURN_IF_ERROR(graph.StartRun({}));
  return absl::OkStatus();
}

absl::Status HairSegMMPGraph::ReleaseMPPGraph() {
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

absl::Status HairSegMMPGraph::RunMPPGraphByImageMode(const cv::Mat& ori_img,
                                                     cv::Mat& output_mask) {
  auto res = RunMPPGraph(ori_img, output_mask);

  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());

  return res;
}

absl::Status HairSegMMPGraph::RunMPPGraph(const cv::Mat& ori_img,
                                          cv::Mat& output_mask) {
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

  mediapipe::Packet mask_packet;
  if (mask_poller_->Next(&mask_packet) && !mask_packet.IsEmpty()) {
    auto& mask = mask_packet.Get<mediapipe::ImageFrame>();
    // 8UC4
    cv::Mat hair_mask = mediapipe::formats::MatView(&mask);
    // 只取第一个通道
    cv::extractChannel(hair_mask, hair_mask, 0);
    // 恢复原图大小
    cv::resize(hair_mask, output_mask, ori_img.size());
  }

  return absl::OkStatus();
}
