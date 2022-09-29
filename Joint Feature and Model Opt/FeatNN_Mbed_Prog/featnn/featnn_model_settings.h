/* Copyright 2020 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/// \file
/// \brief Visual wakewords model settings.

#ifndef V0_1_FEATNN_MODEL_SETTINGS_H_
#define V0_1_FEATNN_MODEL_SETTINGS_H_

constexpr int kKwsInputSize = 128; //window size
constexpr int kModelInputSize = 75; //feat size
constexpr int fft_size = 64; //fft bin size
constexpr int sampling_rate = 50; //sampling_rate_data

constexpr int kCategoryCount = 6;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // V0_1_FEATNN_MODEL_SETTINGS_H_
