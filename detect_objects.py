
import cv2
from ultralytics import YOLO

# Load a small pretrained object detection model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open camera")

while True:
    success, frame = cap.read()

    if not success:
        break

    # Run AI inference
    results = model(frame, verbose=False)

    # Draw bounding boxes and class labels
    annotated_frame = results[0].plot()

    cv2.imshow("AI Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
