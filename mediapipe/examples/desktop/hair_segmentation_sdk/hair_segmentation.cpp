#include "hair_segmentation.h"

#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/util/global_model_path_map.h"

constexpr char kInputStream[] = "input_video";

MMPGraph::MMPGraph() {}
MMPGraph::~MMPGraph() {}

absl::Status MMPGraph::InitMPPGraph() {
  std::vector<std::string> model_paths = {
      "hair_segmentation.tflite",
  };
  return InitMPPGraph(model_paths);
}

absl::Status MMPGraph::InitMPPGraph(std::vector<std::string> model_paths) {
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

absl::Status MMPGraph::ReleaseMPPGraph() {
  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());
  MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
  MP_RETURN_IF_ERROR(graph.WaitUntilDone());
  return absl::OkStatus();
}

absl::Status MMPGraph::RunMPPGraphByImageMode(const cv::Mat& ori_img,
                                              cv::Mat& output_mask) {
  if (graph.GraphInputStreamsClosed()) {
    MP_RETURN_IF_ERROR(graph.StartRun({}));
  }

  auto res = RunMPPGraph(ori_img, output_mask);

  MP_RETURN_IF_ERROR(graph.CloseAllInputStreams());

  return res;
}

absl::Status MMPGraph::RunMPPGraph(const cv::Mat& ori_img,
                                   cv::Mat& output_mask) {
  cv::Mat img;
  cv::cvtColor(ori_img, img, cv::COLOR_BGR2RGB);

  // resize为512x512，不足部分用白色填充
  float scale = 512.0 / std::max(img.cols, img.rows);
  cv::Mat resized_img;
  cv::resize(img, resized_img, cv::Size(), scale, scale);
  cv::Mat white_img = cv::Mat::ones(512, 512, CV_8UC3) * 255;
  cv::Rect roi((512 - resized_img.cols) / 2, (512 - resized_img.rows) / 2,
               resized_img.cols, resized_img.rows);
  resized_img.copyTo(white_img(roi));

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
    // 截取原图部分
    hair_mask = hair_mask(roi);
    // 恢复原图大小
    cv::resize(hair_mask, output_mask, ori_img.size());
  }

  return absl::OkStatus();
}
