# SmartRoomSense config note

Status: **draft** (not frozen yet). Freeze after the pilot (guide Step 7), then tag `config-v1`.
Last updated: 2026-10-31

Rule: any change to the router, channel, width, firmware, send rate, gain lock,
layout, or ESP-IDF version = a new dataset version. Never mix versions in training.

> Private note. It holds the router's MAC address and network name.
> Do not publish it in a public repository. Never write the Wi-Fi password here.

---

## 1. Software

| Item | Value | Source |
| --- | --- | --- |
| ESP-IDF version | v5.5.6 | boot log, 2026-10-31 |
| esp-csi commit | TODO (run `git log -1` in the esp-csi folder) | |
| Firmware | `firmware/csi_recv_router` (copy of Espressif example) | |
| Firmware app version | 8610334 | boot log |
| Firmware build time | Oct 31 2026 14:29:56 | boot log |
| Python version | TODO (`uv run python --version`) | |
| Console baud | 921600 | example `sdkconfig.defaults` |

## 2. Receiver board

| Item | Value | Source |
| --- | --- | --- |
| Board | ESP32-S3-DevKitC-1 | |
| Chip revision | v0.2 | boot log |
| Flash detected | 8 MB | boot log |
| Board Wi-Fi MAC | d8:3b:da:77:7b:64 | boot log |
| Label on board | RX | |
| Serial port | TODO (e.g. COM4); recheck every recording day | |
| USB port used | UART | |

## 3. Router (sender)

| Item | Value | Source |
| --- | --- | --- |
| Model / firmware | TODO | router sticker / admin page |
| Network name (SSID) | HRJRR | boot log |
| BSSID (router MAC) | d8:d8:66:4f:2d:c7 | boot log + `mac` field |
| Band | 2.4 GHz | |
| Channel | 11, fixed (not Auto) | boot log + `channel` field |
| Channel width | 20 MHz | boot log `BW20` + `bandwidth` = 0 |
| Wi-Fi mode | b/g/n | boot log |
| Security | WPA2-PSK | boot log |
| Beacon interval / DTIM | 102400 us / 1 | boot log |
| Router IP (ping target) | 192.168.8.1 | boot log |
| Board IP | 192.168.8.111 | boot log |
| Smart connect / band steering | TODO: confirm OFF | |
| Scheduled reboot / auto update | TODO: confirm OFF for collection days | |
| Other devices on this network while recording | None (rule) | |

## 4. Firmware edits (in `main/app_main.c`)

| Setting | Value | Confirmed? |
| --- | --- | --- |
| `esp_wifi_set_ps(WIFI_PS_NONE)` after `example_connect()` | added | Yes: boot log shows `Set ps type: 0` |
| `CONFIG_FORCE_GAIN` | 1 | TODO: confirm in file (log line `fft_force ... agc_force` appears after ~100 packets) |
| `CONFIG_SEND_FREQUENCY` | 100 (ping every 10 ms) | unchanged from example |
| `csi_config` block | unchanged (LLTF only) | `len` = 128 |
| `ping_config.timeout_ms` | 20 (example default 1000) | Yes, 2026-10-31. The IDF ping task is stop-and-wait: a lost reply or failed send blocked it for 1 s, giving 1-4 rows/s with gaps up to 3.5 s. With 20 ms the 5-minute test ran at 94.8 rows/s, largest gap 196 ms. CSI rows come from every reply that arrives, so a short timeout loses nothing. |
| Other log lines while recording | none | A temporary log line printed from another task landed inside CSI rows (damaged row). Never add log lines to the firmware during recording. |

Rule with gain lock: always power on the board with the room **empty**.

## 5. Values seen in the first CSI_DATA rows (2026-10-31)

| Field # | Name | Value | Expected | OK |
| --- | --- | --- | --- | --- |
| 3 | mac | d8:d8:66:4f:2d:c7 | router BSSID | yes |
| 4 | rssi | -29 (connect log: -23) | for information | - |
| 5 | rate | 11 | set by router | - |
| 6 | sig_mode | 1 (HT packet) | set by router | - |
| 7 | mcs | 7 | set by router | - |
| 8 | bandwidth | 0 | 0 | yes |
| 14 | sgi | 1 | for information | - |
| 15 | noise_floor | -97 | for information | - |
| 17 | channel | 11 | 11 | yes |
| 18 | secondary_channel | 0 | 0 | yes |
| 20 | ant | 0 | 0 | yes |
| 23 | len | 128 | 128 | yes |
| 24 | first_word | 0 | 0 or sometimes 1 | yes |

Note: boot log shows `phy: error: pll_cal exceeds 2ms!!!` (info level). Board connected normally. Watch only.

## 6. Kept subcarrier positions

- First row: positions 0 and 27-37 are zero.
- Planned kept positions: **1-26 and 38-63** (52 values).
- Status: confirmed with Program 3 on an empty-room test file (`roomA_d0_s1_n0_quiet_r1.csv`, 2026-10-31): the 12 lowest positions are exactly 0 and 27-37, all zero. Re-confirm on a pilot file before the freeze.
- Raw files keep all 64 positions. The 12 are removed only in preprocessing.

## 7. Room layout

| Item | Room A | Room B |
| --- | --- | --- |
| Room name / description | TODO | TODO |
| Router position | outside the room, close to it (TODO: describe exact spot) | TODO |
| Board position | TODO | TODO |
| Router-to-board distance | TODO (measure; RSSI -23 to -29 suggests close) | TODO |
| Height (router / board) | TODO / TODO | TODO / TODO |
| Board orientation | TODO (arrow on floor tape) | TODO |
| Door standard state | TODO (e.g. closed on the cable) | TODO |
| Fan in room | TODO yes/no | TODO |
| Laptop rule | outside the room, cable through door | same |
| Photos | TODO file names | TODO |

## 8. Checker PASS rules

| Rule | Draft value | Final (after pilot) |
| --- | --- | --- |
| Rate | >= 80 rows/s | TODO |
| Largest gap | <= 500 ms | TODO |
| Restarts | 0 | TODO |
| Settings | one value each; bandwidth 0, secondary_channel 0, len 128 | TODO |
| Damaged rows | 0 | TODO |

## 9. Link tests and pilot (summaries)

| Test | Date | Result |
| --- | --- | --- |
| A: rate and settings (5 min) | TODO | TODO |
| B: movement response | TODO | TODO |
| C: stability (30 min) | TODO | TODO |
| D: restarts | TODO | TODO |
| Pilot: real rate | TODO | TODO |
| Pilot: still person vs empty visible? | TODO | TODO |
| Pilot: changeover time | TODO | TODO |

## 10. Freeze

- Frozen on: TODO
- Git tag: TODO (`config-v1`)
