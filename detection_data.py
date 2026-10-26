
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not available")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        label = model.names[class_id]

        print(
            f"Object: {label}, "
            f"Confidence: {confidence:.2f}, "
            f"Center: ({center_x:.1f}, {center_y:.1f})"
        )

        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            f"{label}: {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

    cv2.imshow("YOLO Detection Data", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
