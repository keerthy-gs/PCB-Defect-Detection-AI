from pathlib import Path

from ultralytics import YOLO


# Main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent

# We will create this file in the next step
DATASET_CONFIG = PROJECT_FOLDER / "merged_dataset.yaml"


def train_model():
    """Train the YOLOv8 model on the merged PCB defect dataset."""

    if not DATASET_CONFIG.exists():
        raise FileNotFoundError(
            "merged_dataset.yaml was not found. "
            "Please add the merged dataset configuration file first."
        )

    # Start with the pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATASET_CONFIG),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(PROJECT_FOLDER / "runs"),
        name="pcb_defect_yolov8n",
        pretrained=True,
        plots=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    train_model()
