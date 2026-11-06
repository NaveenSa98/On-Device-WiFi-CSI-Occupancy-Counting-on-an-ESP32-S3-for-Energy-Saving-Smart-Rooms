// Declarations for the data arrays produced by tools/gen_dryrun_data.py.
//
// Why "extern"? In C++ a global `const` has *internal* linkage (it is private
// to its .cc file) unless it was declared `extern` first. The generated .cc
// files include this header so the arrays can be used from other files.
#pragma once

#include <stdint.h>

constexpr int kNumSubcarriers = 52;  // CSI subcarriers per packet
constexpr int kNumTestWindows = 5;   // test windows per model

// Numbers copied from models/Wicount/cnn_taskA_win{W}_preprocess.json.
struct PreprocessInfo {
  float input_scale;   // real = (int8 - zero_point) * scale
  int input_zero_point;
  float output_scale;
  int output_zero_point;
};

extern const unsigned char g_model_win128[];
extern const unsigned int g_model_win128_len;
extern const PreprocessInfo g_pre_win128;
extern const int8_t g_test_x_win128[kNumTestWindows][128 * kNumSubcarriers];
extern const int8_t g_test_expected_win128[kNumTestWindows];
extern const int8_t g_test_expected_ref_win128[kNumTestWindows];  // Python reference kernels
extern const int8_t g_test_label_win128[kNumTestWindows];  // 0 = 1 person, 1 = 2 people

extern const unsigned char g_model_win512[];
extern const unsigned int g_model_win512_len;
extern const PreprocessInfo g_pre_win512;
extern const int8_t g_test_x_win512[kNumTestWindows][512 * kNumSubcarriers];
extern const int8_t g_test_expected_win512[kNumTestWindows];
extern const int8_t g_test_expected_ref_win512[kNumTestWindows];  // Python reference kernels
extern const int8_t g_test_label_win512[kNumTestWindows];  // 0 = 1 person, 1 = 2 people
