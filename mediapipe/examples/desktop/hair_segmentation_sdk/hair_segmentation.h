#pragma once

#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

class HairSegMMPGraph {
 private:
  mediapipe::CalculatorGraph graph;
  std::unique_ptr<mediapipe::OutputStreamPoller> mask_poller_;

  std::string calculator_graph_config_contents = R"(
# MediaPipe graph that performs hair segmentation with TensorFlow Lite on CPU.
# Used in the example in
# mediapipie/examples/desktop/hair_segmentation:hair_segmentation_cpu

# Images on CPU coming into and out of the graph.
input_stream: "input_video"
output_stream: "hair_mask"

# Throttles the images flowing downstream for flow control. It passes through
# the very first incoming image unaltered, and waits for
# TfLiteTensorsToSegmentationCalculator downstream in the graph to finish
# generating the corresponding hair mask before it passes through another
# image. All images that come in while waiting are dropped, limiting the number
# of in-flight images between this calculator and
# TfLiteTensorsToSegmentationCalculator to 1. This prevents the nodes in between
# from queuing up incoming images and data excessively, which leads to increased
# latency and memory usage, unwanted in real-time mobile applications. It also
# eliminates unnecessarily computation, e.g., a transformed image produced by
# ImageTransformationCalculator may get dropped downstream if the subsequent
# TfLiteConverterCalculator or TfLiteInferenceCalculator is still busy
# processing previous inputs.
node {
  calculator: "FlowLimiterCalculator"
  input_stream: "input_video"
  input_stream: "FINISHED:hair_mask"
  input_stream_info: {
    tag_index: "FINISHED"
    back_edge: true
  }
  output_stream: "throttled_input_video"
}

# Transforms the input image on CPU to a 512x512 image. To scale the image, by
# default it uses the STRETCH scale mode that maps the entire input image to the
# entire transformed image. As a result, image aspect ratio may be changed and
# objects in the image may be deformed (stretched or squeezed), but the hair
# segmentation model used in this graph is agnostic to that deformation.
node: {
  calculator: "ImageTransformationCalculator"
  input_stream: "IMAGE:throttled_input_video"
  output_stream: "IMAGE:transformed_input_video"
  node_options: {
    [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {
      output_width: 512
      output_height: 512
    }
  }
}

# Caches a mask fed back from the previous round of hair segmentation, and upon
# the arrival of the next input image sends out the cached mask with the
# timestamp replaced by that of the input image, essentially generating a packet
# that carries the previous mask. Note that upon the arrival of the very first
# input image, an empty packet is sent out to jump start the feedback loop.
node {
  calculator: "PreviousLoopbackCalculator"
  input_stream: "MAIN:throttled_input_video"
  input_stream: "LOOP:hair_mask"
  input_stream_info: {
    tag_index: "LOOP"
    back_edge: true
  }
  output_stream: "PREV_LOOP:previous_hair_mask"
}

# Embeds the hair mask generated from the previous round of hair segmentation
# as the alpha channel of the current input image.
node {
  calculator: "SetAlphaCalculator"
  input_stream: "IMAGE:transformed_input_video"
  input_stream: "ALPHA:previous_hair_mask"
  output_stream: "IMAGE:mask_embedded_input_video"
}

# Converts the transformed input image on CPU into an image tensor stored in
# TfLiteTensor. The zero_center option is set to false to normalize the
# pixel values to [0.f, 1.f] as opposed to [-1.f, 1.f]. With the
# max_num_channels option set to 4, all 4 RGBA channels are contained in the
# image tensor.
node {
  calculator: "TfLiteConverterCalculator"
  input_stream: "IMAGE:mask_embedded_input_video"
  output_stream: "TENSORS:image_tensor"
  node_options: {
    [type.googleapis.com/mediapipe.TfLiteConverterCalculatorOptions] {
      zero_center: false
      max_num_channels: 4
    }
  }
}

# Generates a single side packet containing a TensorFlow Lite op resolver that
# supports custom ops needed by the model used in this graph.
node {
  calculator: "TfLiteCustomOpResolverCalculator"
  output_side_packet: "op_resolver"
  node_options: {
    [type.googleapis.com/mediapipe.TfLiteCustomOpResolverCalculatorOptions] {
    }
  }
}

# Runs a TensorFlow Lite model on CPU that takes an image tensor and outputs a
# tensor representing the hair segmentation, which has the same width and height
# as the input image tensor.
node {
  calculator: "TfLiteInferenceCalculator"
  input_stream: "TENSORS:image_tensor"
  output_stream: "TENSORS:segmentation_tensor"
  input_side_packet: "CUSTOM_OP_RESOLVER:op_resolver"
  node_options: {
    [type.googleapis.com/mediapipe.TfLiteInferenceCalculatorOptions] {
      model_path: "$MODEL_PATH"
    }
  }
}

# Decodes the segmentation tensor generated by the TensorFlow Lite model into a
# mask of values in [0, 255], stored in a CPU buffer. It also
# takes the mask generated previously as another input to improve the temporal
# consistency.
node {
  calculator: "TfLiteTensorsToSegmentationCalculator"
  input_stream: "TENSORS:segmentation_tensor"
  input_stream: "PREV_MASK:previous_hair_mask"
  output_stream: "MASK:hair_mask"
  node_options: {
    [type.googleapis.com/mediapipe.TfLiteTensorsToSegmentationCalculatorOptions] {
      tensor_width: 512
      tensor_height: 512
      tensor_channels: 2
      combine_with_previous_ratio: 0.9
      output_layer_index: 1
    }
  }
}
)";

 public:
  HairSegMMPGraph();
  ~HairSegMMPGraph();

  absl::Status InitMPPGraph();
  absl::Status InitMPPGraph(std::vector<std::string> model_paths);

  absl::Status ReleaseMPPGraph();

  absl::Status RunMPPGraph(const cv::Mat& img, cv::Mat& output_mask);
  absl::Status RunMPPGraphByImageMode(const cv::Mat& ori_img,
                                      cv::Mat& output_mask);
};
