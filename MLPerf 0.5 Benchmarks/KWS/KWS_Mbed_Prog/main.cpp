#include "mbed.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "util/quantization_helpers.h"
#include "util/tf_micro_model_runner.h"
#include "kws/kws_input_data.h"
#include "kws/kws_model_data.h"
#include "kws/kws_model_settings.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

Timer t;
constexpr int kTensorArenaSize = 600 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroModelRunner<int8_t, int8_t, 13> *runner;
int8_t input_asint[kKwsInputSize];

int main(int argc, char* argv[]) {
  static tflite::MicroMutableOpResolver<13> resolver;

  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  resolver.AddReshape();
  resolver.AddPad();
  resolver.AddConv2D();
  resolver.AddAdd();
  resolver.AddRelu();
  resolver.AddExpandDims();
  resolver.AddSpaceToBatchNd();
  resolver.AddBatchToSpaceNd();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  
  static tflite::MicroModelRunner<int8_t, int8_t, 13> model_runner(
         g_kws_model_data, resolver, tensor_arena, kTensorArenaSize);
  runner = &model_runner;
  
  for (int i = 0; i < kKwsInputSize; ++i) {
    input_asint[i] = (rand() % (255 - 0 + 1)) + 0;
  } 
  
  t.start();
  runner->SetInput(input_asint);
  runner->Invoke(); 
  t.stop();
  printf("results: ");
  int kCategoryCount = 12;
  for (size_t i = 0; i < kCategoryCount; i++) {
    float converted = DequantizeInt8ToFloat(runner->GetOutput()[i], runner->output_scale(),runner->output_zero_point());
    printf("%0.3f", converted);
    if (i < (kCategoryCount - 1)) {
      printf(",");
    }
    }
    printf("\n");
    printf("timer output: %f\n", t.read());
    t.reset();
  
}

