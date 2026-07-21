import cv2
from ultralytics import YOLO


class PCBDetector:

    def __init__(self, model_path="models/best.pt"):

        self.model = YOLO(model_path)

        self.class_names = {
            0: "Open",
            1: "Short",
            2: "Mouse Bite",
            3: "Spur",
            4: "Copper",
            5: "Pin-hole"
        }

    def predict(self, image_path, conf=0.25):

        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=False,
            verbose=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                confidence = float(box.conf[0])

                detections.append({
                    "class": self.class_names[cls],
                    "confidence": round(confidence * 100, 2)
                })

        annotated_image = results[0].plot()

        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB
        )

        return annotated_image, detections