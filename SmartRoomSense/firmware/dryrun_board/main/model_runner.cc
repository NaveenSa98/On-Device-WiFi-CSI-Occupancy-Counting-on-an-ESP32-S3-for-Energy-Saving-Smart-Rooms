// Loads one int8 TFLite model, runs the 5 test windows, measures time/memory.
//
// Every TFLite Micro call below follows the library's own example
// (managed_components/espressif__esp-tflite-micro/examples/hello_world/main/
//  main_functions.cc). See docs/1_11_board_dry_run.md for the full list.

#include "model_runner.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"  // heap_caps_malloc: choose internal RAM or PSRAM
#include "esp_timer.h"      // esp_timer_get_time: microsecond clock
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"  // vTaskDelay
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"  // MicroPrintf
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

namespace {

// Terms used in this file:
//  * "arena"   = one block of RAM the library uses for all tensors (the
//                model's working memory). We hand it over, the library
//                carves it up. It never calls malloc itself.
//  * "Invoke"  = run the model once (one forward pass = one prediction).
//  * "int8 quantisation" = real_value = (int8_value - zero_point) * scale.

constexpr int kRepeats = 5;                 // Invokes per window (timing)
constexpr size_t kInternalMargin = 32 * 1024;  // keep this much internal RAM free
constexpr size_t kInternalMax = 256 * 1024;    // never ask for more than this
constexpr size_t kPsramArenaBytes = 1024 * 1024;

// A block of RAM for the arena, plus where it came from.
struct Arena {
  uint8_t* ptr = nullptr;
  size_t size = 0;
  const char* where = "";
};

// Allocate an arena. internal=true -> fast internal RAM (size = biggest free
// block minus a margin); internal=false -> 8 MB PSRAM (slower, but large).
Arena AllocArena(bool internal) {
  Arena a;
  if (internal) {
    size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (largest <= kInternalMargin) return a;
    a.size = largest - kInternalMargin;
    if (a.size > kInternalMax) a.size = kInternalMax;
    a.size &= ~static_cast<size_t>(15);  // keep size a multiple of 16
    a.ptr = static_cast<uint8_t*>(
        heap_caps_aligned_alloc(16, a.size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    a.where = "internal";
  } else {
    a.size = kPsramArenaBytes;
    a.ptr = static_cast<uint8_t*>(
        heap_caps_aligned_alloc(16, a.size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    a.where = "PSRAM";
  }
  return a;
}

void FreeArena(Arena* a) {
  heap_caps_free(a->ptr);
  a->ptr = nullptr;
}

void PrintShape(const char* name, const TfLiteIntArray* dims) {
  printf("  %s shape: (", name);
  for (int i = 0; i < dims->size; ++i) printf(i ? ", %d" : "%d", dims->data[i]);
  printf(")\n");
}

// Step d: input must be int8 (1, W, 52); output must be int8.
// Also compare scale / zero point with the numbers from the JSON file.
bool CheckTensors(const ModelSpec& spec, const TfLiteTensor* in,
                  const TfLiteTensor* out, ModelResult* r) {
  PrintShape("input ", in->dims);
  PrintShape("output", out->dims);

  bool shape_ok = in->dims->size == 3 && in->dims->data[0] == 1 &&
                  in->dims->data[1] == spec.window &&
                  in->dims->data[2] == kNumSubcarriers &&
                  in->bytes == static_cast<size_t>(spec.window) * kNumSubcarriers;
  bool type_ok = in->type == kTfLiteInt8 && out->type == kTfLiteInt8;

  const PreprocessInfo& p = *spec.pre;
  bool in_q = fabsf(in->params.scale - p.input_scale) < 1e-7f &&
              in->params.zero_point == p.input_zero_point;
  bool out_q = fabsf(out->params.scale - p.output_scale) < 1e-7f &&
               out->params.zero_point == p.output_zero_point;
  r->quant_ok = in_q && out_q;

  printf("  input  quant: scale=%.8f zp=%d | json: scale=%.8f zp=%d -> %s\n",
         static_cast<double>(in->params.scale), static_cast<int>(in->params.zero_point),
         static_cast<double>(p.input_scale), p.input_zero_point, in_q ? "same" : "DIFFERENT");
  printf("  output quant: scale=%.8f zp=%d | json: scale=%.8f zp=%d -> %s\n",
         static_cast<double>(out->params.scale), static_cast<int>(out->params.zero_point),
         static_cast<double>(p.output_scale), p.output_zero_point, out_q ? "same" : "DIFFERENT");
  printf("  input/output type int8 and input shape (1,%d,%d): %s\n", spec.window,
         kNumSubcarriers, (shape_ok && type_ok) ? "OK" : "WRONG");
  return shape_ok && type_ok;
}

int Abs(int v) { return v < 0 ? -v : v; }

// Model output (int8) -> class. The notebook says: score = (int8 - zp) * scale,
// class 2 people if score > 0.5. Same rule here.
int ClassOf(int q, const TfLiteTensor* out) {
  float score = (q - out->params.zero_point) * out->params.scale;
  return score > 0.5f ? 1 : 0;
}

// Compare the board output of window w with Python and print one line.
void CompareWindow(const ModelSpec& spec, int w, const TfLiteTensor* out, double ms,
                   ModelResult* r) {
  int board = out->data.int8[0];
  int python = spec.expected[w];      // notebook's Python numbers (main reference)
  int python_ref = spec.expected_ref[w];  // Python reference kernels (extra info)
  int diff = board - python;
  if (diff == 0) r->n_exact++;
  if (Abs(diff) > r->max_abs_diff) r->max_abs_diff = Abs(diff);
  if (Abs(board - python_ref) > r->max_abs_diff_ref) r->max_abs_diff_ref = Abs(board - python_ref);

  int board_class = ClassOf(board, out);
  if (board_class == ClassOf(python, out)) r->n_class_same++;
  if (board_class == spec.labels[w]) r->n_class_true++;

  float score = (board - out->params.zero_point) * out->params.scale;
  printf("  window %d: true=%d person(s) | board=%4d python=%4d (ref kernels %4d) diff=%+d | "
         "score=%.3f -> %d person(s) | %.2f ms\n",
         w, spec.labels[w] + 1, board, python, python_ref, diff, static_cast<double>(score),
         board_class + 1, ms);
}

// Steps e + f: run the 5 windows, compare with Python, time each Invoke.
bool RunWindows(const ModelSpec& spec, tflite::MicroInterpreter* interp,
                TfLiteTensor* in, TfLiteTensor* out, ModelResult* r) {
  const size_t window_bytes = static_cast<size_t>(spec.window) * kNumSubcarriers;
  double sum_ms = 0, min_ms = 1e9, max_ms = 0;
  r->n_exact = 0;
  r->max_abs_diff = 0;
  r->max_abs_diff_ref = 0;
  r->n_class_same = 0;
  r->n_class_true = 0;

  for (int w = 0; w < kNumTestWindows; ++w) {
    const int8_t* window_data = spec.test_x + w * window_bytes;

    int64_t total_us = 0;
    for (int k = 0; k < kRepeats; ++k) {
      // IMPORTANT: refill the input before EVERY Invoke. TFLite Micro reuses the
      // arena memory of the input tensor for later layers, so after one Invoke the
      // input tensor holds garbage. (Copy is outside the timed part.)
      memcpy(in->data.int8, window_data, window_bytes);
      int64_t t0 = esp_timer_get_time();
      TfLiteStatus s = interp->Invoke();
      total_us += esp_timer_get_time() - t0;
      if (s != kTfLiteOk) {
        printf("  Invoke FAILED on window %d\n", w);
        return false;
      }
    }
    vTaskDelay(1);  // let the idle task run, else the task watchdog fires (slow kernels)
    double ms = total_us / 1000.0 / kRepeats;  // mean time of one Invoke
    sum_ms += ms;
    if (ms < min_ms) min_ms = ms;
    if (ms > max_ms) max_ms = ms;

    CompareWindow(spec, w, out, ms, r);
  }
  r->invoke_ms_min = static_cast<float>(min_ms);
  r->invoke_ms_mean = static_cast<float>(sum_ms / kNumTestWindows);
  r->invoke_ms_max = static_cast<float>(max_ms);
  return true;
}

enum class Attempt { kOk, kArenaTooSmall, kFailed };

// Build the interpreter on the given arena and run everything.
Attempt TryWithArena(const ModelSpec& spec, const tflite::Model* model, Arena* arena,
                     ModelResult* r) {
  // Step b: register ONLY the 7 ops this model uses (saves flash/RAM).
  tflite::MicroMutableOpResolver<7> resolver;
  if (resolver.AddConv2D() != kTfLiteOk || resolver.AddExpandDims() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk || resolver.AddLogistic() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk || resolver.AddMean() != kTfLiteOk ||
      resolver.AddReshape() != kTfLiteOk) {
    printf("  could not register ops\n");
    return Attempt::kFailed;
  }

  // Step c: the interpreter runs the model inside the arena.
  tflite::MicroInterpreter interp(model, resolver, arena->ptr, arena->size);
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("  AllocateTensors failed with %s arena (%u bytes)\n", arena->where,
           static_cast<unsigned>(arena->size));
    return Attempt::kArenaTooSmall;
  }

  // Step g: bytes of the arena actually used (MicroInterpreter::arena_used_bytes).
  r->arena_used_bytes = interp.arena_used_bytes();
  r->arena_size_bytes = arena->size;
  r->arena_in = arena->where;
  printf("  arena: %s, allocated %u bytes, used %u bytes\n", arena->where,
         static_cast<unsigned>(arena->size), static_cast<unsigned>(r->arena_used_bytes));

  TfLiteTensor* in = interp.input(0);
  TfLiteTensor* out = interp.output(0);
  if (!CheckTensors(spec, in, out, r)) return Attempt::kFailed;
  if (!RunWindows(spec, &interp, in, out, r)) return Attempt::kFailed;
  r->ran = true;
  return Attempt::kOk;
}

}  // namespace

bool RunModel(const ModelSpec& spec, ModelResult* r) {
  memset(r, 0, sizeof(*r));
  r->arena_in = "-";
  printf("\n=== window %d (model %u bytes) ===\n", spec.window, spec.model_len);

  // Step a: map the C array to a Model and check the schema version.
  const tflite::Model* model = tflite::GetModel(spec.model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf("  schema version %u != supported %d\n", static_cast<unsigned>(model->version()),
           TFLITE_SCHEMA_VERSION);
    return false;
  }
  printf("  schema version %u OK\n", static_cast<unsigned>(model->version()));

  // Step c: try internal RAM first, then PSRAM.
  for (bool internal : {true, false}) {
    Arena arena = AllocArena(internal);
    if (arena.ptr == nullptr) {
      printf("  could not allocate %s arena\n", internal ? "internal" : "PSRAM");
      continue;
    }
    Attempt result = TryWithArena(spec, model, &arena, r);
    FreeArena(&arena);  // interpreter is already gone (it was local)
    if (result != Attempt::kArenaTooSmall) return result == Attempt::kOk;
  }
  return false;
}
