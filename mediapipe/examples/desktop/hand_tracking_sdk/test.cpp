// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <shared_mutex>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
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
constexpr char kWindowName[] = "MediaPipe";

std::string input_video_path;
int open_camera_id = 0;

std::string output_video_path;

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      "mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt",
      &calculator_graph_config_contents));
  absl::StrReplaceAll({{"$FACE_MAX_NUM", std::to_string(1)}},
                      &calculator_graph_config_contents);
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !input_video_path.empty();
  if (load_video) {
    capture.open(input_video_path);
  } else {
    capture.open(open_camera_id);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
  }

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("landmarks", true));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller handedness_poller,
                   graph.AddOutputStreamPoller("handedness", true));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      if (!load_video) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

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

    cv::Mat output_frame_mat = camera_frame.clone();

    std::vector<std::vector<cv::Point>> all_cv_landmarks;

    mediapipe::Packet landmarks_packet;
    if (poller.Next(&landmarks_packet) && !landmarks_packet.IsEmpty()) {
      auto& landmarks =
          landmarks_packet
              .Get<std::vector<mediapipe::NormalizedLandmarkList>>();

      for (const auto& landmark : landmarks) {
        std::vector<cv::Point> cv_landmarks;
        cv::Scalar color(0, 255, 0);
        for (size_t i = 0; i < landmark.landmark_size(); i++) {
          const auto& point = landmark.landmark(i);
          cv_landmarks.push_back(cv::Point(point.x() * output_frame_mat.cols,
                                           point.y() * output_frame_mat.rows));
        }
        all_cv_landmarks.push_back(cv_landmarks);
      }
    }

    std::vector<mediapipe::ClassificationList> handedness;

    mediapipe::Packet handedness_packet;
    if (handedness_poller.Next(&handedness_packet) &&
        !handedness_packet.IsEmpty()) {
      handedness =
          handedness_packet.Get<std::vector<mediapipe::ClassificationList>>();
    }

    for (int i = 0; i < all_cv_landmarks.size(); i++) {
      const std::vector<cv::Point>& landmarks = all_cv_landmarks[i];
      for (int j = 0; j < landmarks.size(); j++) {
        cv::circle(output_frame_mat, landmarks[j], 2, cv::Scalar(0, 255, 0),
                   -1);
      }
      cv::Rect rect = cv::boundingRect(landmarks);
      cv::rectangle(output_frame_mat, rect, cv::Scalar(255, 0, 0), 2);
      cv::putText(output_frame_mat,
                  handedness[i].classification(0).label() + " " +
                      std::to_string(handedness[i].classification(0).score()),
                  cv::Point(rect.x, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                  cv::Scalar(255, 0, 0));
    }

    // // Get the graph result packet, or stop if that fails.
    // mediapipe::Packet packet;
    // if (!poller.Next(&packet)) break;
    // auto& output_frame = packet.Get<mediapipe::ImageFrame>();

    // // Convert back to opencv for display or saving.
    // cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
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
