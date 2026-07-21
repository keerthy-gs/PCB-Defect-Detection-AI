from src.detector import PCBDetector

detector = PCBDetector()

annotated_image, detections = detector.predict(
    "dataset/merged/images/test/deeppcb_00041201.jpg"
)

print("\nDetected Defects:\n")

for defect in detections:
    print(f"{defect['class']} : {defect['confidence']}%")