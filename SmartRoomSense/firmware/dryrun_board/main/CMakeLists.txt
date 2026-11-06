idf_component_register(SRCS "main.cc" "model_runner.cc" "model_win128.cc" "model_win512.cc" "test_windows_win128.cc" "test_windows_win512.cc"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_timer)

