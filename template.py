from pathlib import Path

GIT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = GIT_ROOT / "SmartRoomSense"

folders = [
    "data/raw/wicount",
    "data/raw/ehunam",
    "data/raw/roomsense",
    "data/interim",
    "data/processed",
    "data/splits",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/deployment",
    "src/utils",
    "firmware/csi_tx",
    "firmware/csi_rx_infer",
    "firmware/components/preprocess",
    "configs",
    "models/keras",
    "models/tflite",
    "models/c_arrays",
    "reports/figures",
    "reports/tables",
    "reports/thesis",
    "scripts",
    "tests",
    "logs/experiments",
    "docs/ethics",
]

for folder in folders:
    (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)

print(f"Project structure created at: {PROJECT_ROOT}")

