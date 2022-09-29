#include "mbed.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "util/quantization_helpers.h"
#include "util/tf_micro_model_runner.h"
#include "ic/ic_inputs.h"
#include "ic/ic_model_quant_data.h"
#include "ic/ic_model_settings.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

Timer t;
constexpr int kTensorArenaSize=600*1000;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroModelRunner<int8_t, int8_t, 7> *runner;
int8_t input_asint[kIcInputSize];

int main(int argc, char* argv[]) {
  static tflite::MicroMutableOpResolver<7> resolver;

  resolver.AddAdd();
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddAveragePool2D();
static tflite::MicroModelRunner<int8_t, int8_t, 7> model_runner(
      pretrainedResnet_quant_tflite, resolver, tensor_arena, kTensorArenaSize);
  runner = &model_runner;
  
  for (int i = 0; i < kIcInputSize; ++i) {
input_asint[i] = (rand() % (255 - 0 + 1)) + 0;
  } 
  
  runner->SetInput(input_asint);
  t.start();
  runner->Invoke(); 
  t.stop();
  printf("results: ");
  const int nresults = 10;
  int kCategoryCount = 10;
  for (size_t i = 0; i < kCategoryCount; i++) {
float converted = DequantizeInt8ToFloat(runner->GetOutput()[i], runner->output_scale(),runner->output_zero_point());
    printf("%0.3f", converted);
    if (i < (nresults - 1)) {
      printf(",");
    }
    }
    printf("\n");
    printf("timer output: %f\n", t.read());
    t.reset();
  
}

