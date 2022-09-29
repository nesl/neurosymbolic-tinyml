#include "features.h"
#include "math.h"
#include "featnn/cmsis_dsp/Include/arm_math.h"

float mean(float* my_ar){
    float sum = 0.0f;
    for(int i = 0; i < kKwsInputSize; i++){
        sum+= my_ar[i];
    }
    return sum/(float)kKwsInputSize;
}

float max(float* my_ar){
    float max_val = -99999.0f;
    for(int i = 0; i < kKwsInputSize; i++){
        if(my_ar[i] > max_val){
          max_val = my_ar[i];
        }
    }
    return max_val;
}

float min(float* my_ar){
    float min_val = 99999.0f;
    for(int i = 0; i < kKwsInputSize; i++){
        if(my_ar[i] < min_val){
          min_val = my_ar[i];
        }
    }
    return min_val;
}

float variance(float* my_ar){
  float avg_val = mean(my_ar);
  float sum = 0.0f;
  for(int i = 0; i < kKwsInputSize; i++){
      sum+= pow((my_ar[i] - avg_val), 2);
  }
  return sum/kKwsInputSize;
}

int compare (const void * a, const void * b)
{
  float fa = *(float*) a;
  float fb = *(float*) b;
  return (fa > fb) - (fa < fb);
}

float median(float* my_ar)
{
qsort(my_ar, kKwsInputSize, sizeof(float), compare);

if(kKwsInputSize%2==0)
    return (my_ar[kKwsInputSize/2]+my_ar[kKwsInputSize/2-1])/2;
else
    return my_ar[kKwsInputSize/2];
}

float meanAbsoluteDeviation(float* my_ar)
{
    float absSum = 0;
    for (int i = 0; i < kKwsInputSize; i++)
        absSum = absSum + abs(my_ar[i] - mean(my_ar));
    return absSum / kKwsInputSize;
}

float IQR(float* my_ar)
{
  qsort(my_ar, kKwsInputSize, sizeof(float), compare);
  return my_ar[(int)floor(((float)kKwsInputSize+1.0)*0.75f)] - my_ar[(int)floor(((float)kKwsInputSize+1.0)*0.25f)];

}

float entropy(float* my_ar)
{
  float entropy_val = 0.0f;
  for(int i=0; i< kKwsInputSize; i++){
    if(my_ar[i] > 0.0){
      entropy_val -= my_ar[i] * (float) log( (double) my_ar[i]);
    }
  }
  entropy_val = entropy_val/ (float) log ((double) 2.0);
  return entropy_val;
}

float peak_to_peak(float* my_ar)
{
  return abs(max(my_ar) - min(my_ar));
}

float abs_energy(float* my_ar)
{
  float sum = 0.0f;
  for (int i = 0; i < kKwsInputSize; i++){
    sum+= abs(my_ar[i])*abs(my_ar[i]);
  }
  return sum;
}

void get_fft_samples(float* my_ar, float* out_ar){
    uint32_t ifftFlag = 0;
    static float32_t testInput[kKwsInputSize];
    static float32_t testOutput[fft_size];
    for (int i = 0; i < kKwsInputSize; i++){
        testInput[i] = (float32_t) my_ar[i];
    }
    arm_rfft_fast_instance_f32 rfft_instance;
    arm_rfft_fast_init_f32(&rfft_instance, fft_size);
    arm_rfft_fast_f32(&rfft_instance, testInput, testOutput, ifftFlag);
    for (int i = 0; i < fft_size; i++){
        out_ar[i] = (float) testOutput[i];
    }
}

float peak_freq(float *my_ar){
    float mean_ar = mean(my_ar);
    for(int i = 0; i < kKwsInputSize; i++){
        my_ar[i] = my_ar[i]-mean_ar;
    }
    float my_out_ar[fft_size];
    get_fft_samples(my_ar, my_out_ar);
    for(int i = 0; i < fft_size; i++){
        my_out_ar[i] = abs(my_out_ar[i]);
    }

    float max_val = -99999.0f;
    int max_idx = 0;
    for(int i = 0; i < fft_size; i++){
        if(my_out_ar[i] > max_val){
          max_val = my_out_ar[i];
          max_idx = i;
        }
    }
    float freq_ar[fft_size];
    for(int i =0; i < fft_size; i++){
        freq_ar[i] = ((float)sampling_rate/(float)kKwsInputSize)*i;
    }
    return freq_ar[max_idx];
}

float max_psd(float *my_ar){
    float my_out_ar[fft_size];
    get_fft_samples(my_ar, my_out_ar);

    for(int i =0; i < fft_size; i++){
        my_out_ar[i] = 1.0f/((float)sampling_rate *(float)kKwsInputSize)*abs(my_out_ar[i])*abs(my_out_ar[i]);
    }
    float max_val = -99999.0f;
    for(int i = 0; i < fft_size; i++){
        if(my_out_ar[i] > max_val){
          max_val = my_out_ar[i];
        }
    }
    return max_val;

}

void extract_feat(float *my_ar, float *feat_ar, int *mask_ar){
    int j = 0;
    if(mask_ar[0]==1){
        j = 64;
        float my_out_ar[fft_size];
        get_fft_samples(my_ar, my_out_ar);
        for(int i=0; i<fft_size; i++){
            feat_ar[i] = my_out_ar[i];
        }
    }
    typedef float (*f)(float[]);       
    f func[11] = {&peak_freq, &max_psd, &IQR, &max, &mean, &meanAbsoluteDeviation, &median, &variance, &abs_energy, &entropy, &peak_to_peak};
    
    for(int i = 1; i < 12; i++){
        if (mask_ar[i] == 1){
            feat_ar[j] = func[i](my_ar);
            j = j+1;
        }
    }
}