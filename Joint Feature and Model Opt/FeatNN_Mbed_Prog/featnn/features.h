#ifndef FEATURES_H_
#define FEATURES_H_

#include "featnn/featnn_model_settings.h"

float mean(float* my_ar);
float max(float* my_ar);
float min(float* my_ar);
float variance(float* my_ar);
int compare (const void * a, const void * b);
float median(float* my_ar);
float meanAbsoluteDeviation(float* my_ar);
float IQR(float* my_ar);
float entropy(float* my_ar);
float peak_to_peak(float* my_ar);
float abs_energy(float* my_ar);
void get_fft_samples(float* my_ar, float* out_ar);
float peak_freq(float* my_ar);
float max_psd(float *my_ar);
void extract_feat(float *my_ar, float *feat_ar, int *mask_ar);
#endif // FEATURES_H_
