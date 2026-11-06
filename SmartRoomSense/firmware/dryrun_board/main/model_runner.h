// Runs one int8 model on the 5 test windows and measures it.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "dryrun_data.h"

// Everything we need to know about one model (inputs of the test).
struct ModelSpec {
  int window;                  // packets per window: 128 or 512
  const unsigned char* model;  // the .tflite file as a C array
  unsigned int model_len;      // its size in bytes
  const PreprocessInfo* pre;   // scale / zero point from the JSON file
  const int8_t* test_x;        // 5 windows, each window*52 int8 values
  const int8_t* expected;      // 5 int8 outputs from the notebook's Python interpreter
  const int8_t* expected_ref;  // same, from Python's plain reference kernels
  const int8_t* labels;        // true class: 0 = 1 person, 1 = 2 people (Task A)
};

// Everything we measured (outputs of the test).
struct ModelResult {
  bool ran;                  // false if setup failed before any Invoke
  size_t arena_used_bytes;   // arena bytes the interpreter really used
  size_t arena_size_bytes;   // arena bytes we allocated
  const char* arena_in;      // "internal" or "PSRAM"
  bool quant_ok;             // scale/zero point equal to the JSON file
  float invoke_ms_min, invoke_ms_mean, invoke_ms_max;
  int n_exact;               // windows whose output equals Python exactly
  int max_abs_diff;          // largest |board - python| over the 5 windows
  int max_abs_diff_ref;      // same, against Python's reference kernels
  int n_class_same;          // windows where board class == Python class
  int n_class_true;          // windows where board class == true label
};

// Returns true if the model ran all 5 windows. Prints details to serial.
bool RunModel(const ModelSpec& spec, ModelResult* result);
