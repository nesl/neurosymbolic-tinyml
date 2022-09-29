#include "mbed.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "util/quantization_helpers.h"
#include "util/tf_micro_model_runner.h"
#include "featnn/featnn_model_settings.h"
#include "featnn/features.h"
#include "featnn/featnn_model_data.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>


Timer t;
constexpr int kTensorArenaSize = 500 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroModelRunner<float, float, 13> *runner;
float input_asint[kKwsInputSize];
float input_model[kModelInputSize];

//spectral: get_fft_samples, peak_freq, max_psd, 
//statistical: IQR, max, mean, meanAbsoluteDeviation, median, variance, 
//temporal: abs_energy, entropy, peak_to_peak
//in alphabetical order as Python TSFEL extracts for each category
int feat_mask[12] = {1,1,1,1,1,1,1,1,1,1,1,1};

int main() {
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

    static tflite::MicroModelRunner<float, float, 13> model_runner(
            g_featnn_model_data, resolver, tensor_arena, kTensorArenaSize);
    runner = &model_runner;

    for (int i = 0; i < kKwsInputSize; ++i) {
        input_asint[i] = ((float)rand()/(float)(RAND_MAX))*20.0;;
    }

    extract_feat(input_asint, input_model, feat_mask);

    t.start();
    runner->SetInput(input_model);
    runner->Invoke(); 
    t.stop();

    for (size_t i = 0; i < kCategoryCount; i++) {
        float converted = runner->GetOutput()[i];
        printf("%0.3f", converted);
        if (i < (kCategoryCount - 1)) {
            printf(",");
        }
    }
    printf("\n");
    printf("timer output: %f\n", t.read());
    t.reset();
    
}
