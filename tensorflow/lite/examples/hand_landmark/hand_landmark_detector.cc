#include <iostream>
#include <vector>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image/label_image.h"
#include "hand_landmark_detector.h"

int main(int argc, char* argv[]) {
  std::string model_file = "tensorflow/lite/examples/hand_landmark/data/hand_landmark.tflite";
  std::string image_file = "tensorflow/lite/examples/hand_landmark/data/hand.bmp";
  int input_width = 256;
  int input_height = 256;
  float input_mean = 0.0f;
  float input_std = 255.0f;

  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag("tflite_model", &model_file, "Model path"),
      tflite::Flag::CreateFlag("image", &image_file, "Image path"),
      tflite::Flag::CreateFlag("input_width", &input_width, "Input width"),
      tflite::Flag::CreateFlag("input_height", &input_height, "Input height"),
      tflite::Flag::CreateFlag("input_mean", &input_mean, "Input mean"),
      tflite::Flag::CreateFlag("input_std", &input_std, "Input std"),
  };

  bool parse_result = tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);
  if (!parse_result) {
	  std::cout << "Failed to parse command-line flags";
    return 1;
  }

  // 加载模型
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
  if (!model) {
	  std::cout << "Failed to load model: " << model_file;
    return 1;
  }

  // 创建解释器
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
	  std::cout << "Failed to create interpreter";
    return 1;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tensors";
    return 1;
  }

  // 获取输入张量信息
  int input_tensor_index = interpreter->inputs()[0];
  TfLiteTensor* input_tensor = interpreter->tensor(input_tensor_index);
  if (input_tensor->type != kTfLiteFloat32) {
    std::cout << "Input tensor type must be float32";
    return 1;
  }

  // 加载图像
  int image_width, image_height, image_channels;
  struct tflite::label_image::Settings s;
  std::vector<uint8_t> image_data = tflite::label_image::read_bmp(
      image_file, &image_width, &image_height, &image_channels, &s);
  
  if (image_data.empty()) {
    std::cout << "Failed to read image: " << image_file;
    return 1;
  }

  // 调整图像大小到模型输入尺寸
  std::vector<uint8_t> resized_image(input_width * input_height * image_channels);

  tflite::label_image::resize<uint8_t>(
      resized_image.data(),          // 输出缓冲区
      image_data.data(),             // 输入缓冲区
      image_height,                  // 原始高度
      image_width,                   // 原始宽度
      image_channels,                // 原始通道数
      input_height,                  // 目标高度
      input_width,                   // 目标宽度
      image_channels,                // 目标通道数
      &s);                      // Settings 指针（设为 nullptr）
				     //
  // 将图像数据复制到输入张量
  float* input_data = interpreter->typed_input_tensor<float>(0);
  std::cout << "image_channels: " << image_channels <<std::endl;
  for (int i = 0; i < input_height * input_width * image_channels; ++i) {
    //input_data[i] = (resized_image[i] - input_mean) / input_std;
    input_data[i] = imageData[i];
  }				     

  if (interpreter->Invoke() != kTfLiteOk) {
	  std::cout << "Failed to invoke interpreter";
    return 1;
  }

  // 获取输出张量
  TfLiteTensor* output_tensor = interpreter->output_tensor(0);
  if (output_tensor->dims->size != 2 || output_tensor->dims->data[1] != 63) {
	  std::cout << "Output tensor shape mismatch";
    return 1;
  }

    float* output_data = interpreter->typed_output_tensor<float>(0);
  std::cout << "Hand Landmarks (x, y, z):\n";
  for (int i = 0; i < 21; ++i) {
    float x = output_data[i * 3];
    float y = output_data[i * 3 + 1];
    float z = output_data[i * 3 + 2];
    std::cout << "Point " << i << ": (" << x << ", " << y << ", " << z << ")\n";
  }


  std::cout << "To continue";



  return 0;
}
