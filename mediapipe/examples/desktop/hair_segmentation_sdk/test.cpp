#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_replace.h"
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

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "hair_mask";
constexpr char kWindowName[] = "MediaPipe";

std::string input_video_path;
int open_camera_id = 0;

std::string output_video_path;

std::string typeToString(int type);

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      "mediapipe/graphs/hair_segmentation/hair_segmentation_desktop_live.pbtxt",
      &calculator_graph_config_contents));

  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream, true));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  // Capture opencv camera or video frame.
  cv::Mat camera_frame_raw =
      cv::imread("D:/work/C/ImageDeal/bin/data/res_18.jpg");
  if (camera_frame_raw.empty()) {
    LOG(INFO) << "Empty frame, end of video reached.";
  }
  cv::Mat camera_frame;
  cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  camera_frame.copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us))));

  cv::Mat output_frame_mat = camera_frame_raw.clone();
  output_frame_mat.convertTo(output_frame_mat, CV_8UC4);

  mediapipe::Packet mask_packet;
  if (poller.Next(&mask_packet) && !mask_packet.IsEmpty()) {
    auto& mask = mask_packet.Get<mediapipe::ImageFrame>();
    cv::Mat hair_mask = mediapipe::formats::MatView(&mask);
    cv::resize(hair_mask, hair_mask, output_frame_mat.size());
    // 分离透明通道
    cv::Mat mask_channels[4];
    cv::split(hair_mask, mask_channels);
    hair_mask = mask_channels[3];
    // hair_mask.convertTo(hair_mask, CV_32FC1);

    // // 开运算
    // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,
    // 11)); cv::morphologyEx(hair_mask, hair_mask, cv::MORPH_OPEN, element,
    // cv::Point(-1, -1), 3);

    // cv::namedWindow("mask", cv::WINDOW_NORMAL);
    // cv::resizeWindow("mask", hair_mask.cols, hair_mask.rows);
    // cv::imshow("mask", hair_mask);
    std::cout << typeToString(hair_mask.type()) << std::endl;
    // std::cout << typeToString(output_frame_mat.type()) << std::endl;

    cv::Mat hair_mat = output_frame_mat.clone();

    for (int i = 0; i < hair_mat.rows; i++) {
      for (int j = 0; j < hair_mat.cols; j++) {
        cv::Vec3b& pixel = hair_mat.at<cv::Vec3b>(i, j);
        float v = hair_mask.at<uchar>(i, j) / 255.0;
        pixel[0] = 255 * (1 - v) + pixel[0] * v;
        pixel[1] = 255 * (1 - v) + pixel[1] * v;
        pixel[2] = 255 * (1 - v) + pixel[2] * v;
      }
    }

    // 转换为hsv
    cv::Mat hsv;
    cv::cvtColor(hair_mat, hsv, cv::COLOR_BGR2HSV);

    float red_weight = 0.5;
    float green_weight = 0.3;
    float blue_weight = 0.2;
    // h大于300的转换为灰度
    for (int i = 0; i < hsv.rows; i++) {
      for (int j = 0; j < hsv.cols; j++) {
        cv::Vec3b& pixel = hsv.at<cv::Vec3b>(i, j);
        cv::Vec3b& pixel2 = hair_mat.at<cv::Vec3b>(i, j);
        if ((pixel[0] > 150 || pixel[0] < 8) && pixel[2] < 127) {
          // pixel2转换为灰度
          int gray = red_weight * pixel2[2] + green_weight * pixel2[1] +
                     blue_weight * pixel2[0];
          pixel2[0] = gray;
          pixel2[1] = gray;
          pixel2[2] = gray;
        }
      }
    }

    // 转换为灰度
    // cv::cvtColor(hair_mat, hair_mat, cv::COLOR_BGR2GRAY);
    // cv::cvtColor(hair_mat, hair_mat, cv::COLOR_GRAY2BGR);

    cv::imwrite("D:/work/C/ImageDeal/bin/data/hair_mat.jpg", hair_mat);
    cv::imshow("hair_mat", hair_mat);
    cv::waitKey(0);

    // 使用蒙版合并图像
    for (int i = 0; i < output_frame_mat.rows; i++) {
      for (int j = 0; j < output_frame_mat.cols; j++) {
        cv::Vec3b& pixel = output_frame_mat.at<cv::Vec3b>(i, j);
        cv::Vec3b& pixel2 = hair_mat.at<cv::Vec3b>(i, j);
        float v = hair_mask.at<uchar>(i, j) / 255.0;
        pixel[0] = pixel2[0] * v + pixel[0] * (1 - v);
        pixel[1] = pixel2[1] * v + pixel[1] * (1 - v);
        pixel[2] = pixel2[2] * v + pixel[2] * (1 - v);
      }
    }
  }

  // // Convert back to opencv for display or saving.
  // cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
  // cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

  // 创建窗口
  cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
  cv::resizeWindow(kWindowName, output_frame_mat.cols * 2,
                   output_frame_mat.rows);
  // 左右合并显示
  cv::Mat show_frame(output_frame_mat.rows, output_frame_mat.cols * 2,
                     output_frame_mat.type());
  output_frame_mat.copyTo(
      show_frame(cv::Rect(0, 0, output_frame_mat.cols, output_frame_mat.rows)));
  camera_frame_raw.copyTo(show_frame(cv::Rect(
      output_frame_mat.cols, 0, output_frame_mat.cols, output_frame_mat.rows)));
  cv::imshow(kWindowName, show_frame);
  // Press any key to exit.
  cv::waitKey(0);

  LOG(INFO) << "Shutting down.";
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}

std::string typeToString(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  r += "C";
  r += (chans + '0');

  return r;
}