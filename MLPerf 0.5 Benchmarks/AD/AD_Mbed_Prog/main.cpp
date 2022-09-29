#include "mbed.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "util/quantization_helpers.h"
#include "util/tf_micro_model_runner.h"
#include "model.h"
#include "micro_model_settings.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

Timer t;

constexpr int kTensorArenaSize=600*1000;
uint8_t tensor_arena[kTensorArenaSize];
typedef int8_t model_input_t;
typedef int8_t model_output_t;;

float input_float[kInputSize];
int8_t input_quantized[kInputSize];
float result;
 
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;

int main(int argc, char* argv[]) {
  //initialize
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<15> micro_op_resolver(error_reporter);
  micro_op_resolver.AddShape();
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddPack();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddExpandDims();
  micro_op_resolver.AddSpaceToBatchNd();
  micro_op_resolver.AddBatchToSpaceNd();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
  } 
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
  }  

  //inputs
  
  for (int i = 0; i < kInputSize; ++i) {
    input_float[i] = ((float)rand()/(float)(RAND_MAX))*20.0;
  } 
  float input_scale = interpreter->input(0)->params.scale;
  int input_zero_point = interpreter->input(0)->params.zero_point;
  for (int i = 0; i < kInputSize; i++) {
    input_quantized[i] = QuantizeFloatToInt8(
        input_float[i], input_scale, input_zero_point);
  }  
  
  
  
int8_t *model_input_buffer = model_input->data.int8;
int8_t *feature_buffer_ptr = input_quantized;

  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer_ptr[i];
  }
  
  //inference
  t.start();
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
  } 
  t.stop();
  
  //results
  printf("results: ");
  float diffsum = 0;

  TfLiteTensor* output = interpreter->output(0);
  for (size_t i = 0; i < kFeatureElementCount; i++) {
    float converted = DequantizeInt8ToFloat(output->data.int8[i], interpreter->output(0)->params.scale, interpreter->output(0)->params.zero_point);
    float diff = converted - input_float[i];
    diffsum += diff * diff;
  }
  diffsum /= kFeatureElementCount;

  result = diffsum;  
  printf("%0.3f\n", result);
  
  printf("timer output: %f\n", t.read());
  t.reset();
  
}

