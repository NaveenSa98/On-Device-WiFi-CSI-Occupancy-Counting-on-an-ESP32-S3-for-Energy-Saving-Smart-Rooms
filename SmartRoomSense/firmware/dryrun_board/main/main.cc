// Step 1.11 - board dry run of the int8 CNN (no WiFi, no CSI, no MQTT).
//
// app_main() is the entry point of every ESP-IDF program (like main() in
// plain C). It runs once, in the "main task".

#include <stdio.h>

#include "esp_heap_caps.h"
#include "model_runner.h"

namespace {

constexpr float kMaxMeanMs = 100.0f;  // target: mean Invoke time below this
constexpr int kMaxDiff = 1;           // target: at most 1 int8 step different

const char* PassFail(bool ok) { return ok ? "PASS" : "FAIL"; }

void PrintMemory() {
  printf("Internal RAM: total %u, largest free block %u bytes\n",
         static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_INTERNAL)),
         static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)));
  printf("PSRAM:        total %u bytes (0 means PSRAM not found!)\n",
         static_cast<unsigned>(heap_caps_get_total_size(MALLOC_CAP_SPIRAM)));
}

void PrintTable(const ModelSpec* specs, const ModelResult* res, int n) {
  printf("\n| win | model_bytes | arena_used_bytes | arena_in (internal/PSRAM) | "
         "invoke_ms (min/mean/max) | match (n/5, max abs diff) |\n");
  printf("|-----|-------------|------------------|---------------------------|"
         "--------------------------|---------------------------|\n");
  for (int i = 0; i < n; ++i) {
    const ModelResult& r = res[i];
    if (!r.ran) {
      printf("| %d | %u | - | - | - | did not run |\n", specs[i].window, specs[i].model_len);
      continue;
    }
    printf("| %d | %u | %u | %s | %.2f / %.2f / %.2f | %d/%d, %d |\n", specs[i].window,
           specs[i].model_len, static_cast<unsigned>(r.arena_used_bytes), r.arena_in,
           static_cast<double>(r.invoke_ms_min), static_cast<double>(r.invoke_ms_mean),
           static_cast<double>(r.invoke_ms_max), r.n_exact, kNumTestWindows, r.max_abs_diff);
  }
}

void PrintTargets(const ModelSpec* specs, const ModelResult* res, int n) {
  printf("\nTargets:\n");
  for (int i = 0; i < n; ++i) {
    const ModelResult& r = res[i];
    printf("  win %3d: mean < %.0f ms: %s | arena fits: %s | output within %d step: %s | "
           "quant == JSON: %s\n",
           specs[i].window, static_cast<double>(kMaxMeanMs),
           PassFail(r.ran && r.invoke_ms_mean < kMaxMeanMs), PassFail(r.ran),
           kMaxDiff, PassFail(r.ran && r.max_abs_diff <= kMaxDiff),
           PassFail(r.ran && r.quant_ok));
    // Extra information (not one of the 3 targets): class decision and kernel spread.
    printf("          class same as Python: %d/%d | class equals true label: %d/%d | "
           "max diff vs Python reference kernels: %d\n",
           r.n_class_same, kNumTestWindows, r.n_class_true, kNumTestWindows,
           r.max_abs_diff_ref);
  }
}

}  // namespace

extern "C" void app_main(void) {
  printf("\n##### Step 1.11 board dry run #####\n");
  PrintMemory();

  const ModelSpec specs[] = {
      {128, g_model_win128, g_model_win128_len, &g_pre_win128, &g_test_x_win128[0][0],
       g_test_expected_win128, g_test_expected_ref_win128, g_test_label_win128},
      {512, g_model_win512, g_model_win512_len, &g_pre_win512, &g_test_x_win512[0][0],
       g_test_expected_win512, g_test_expected_ref_win512, g_test_label_win512},
  };
  constexpr int kNumModels = sizeof(specs) / sizeof(specs[0]);
  ModelResult results[kNumModels];

  for (int i = 0; i < kNumModels; ++i) RunModel(specs[i], &results[i]);

  PrintTable(specs, results, kNumModels);
  PrintTargets(specs, results, kNumModels);
  printf("\n##### dry run finished #####\n");
}
