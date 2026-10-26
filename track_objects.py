
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    success, frame = cap.read()

    if not success:
        break

    # YOLO built-in tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated = results[0].plot()

    if results[0].boxes is not None:
        boxes = results[0].boxes

        if boxes.id is not None:
            for box, track_id in zip(
                boxes, boxes.id.int().cpu().tolist()
            ):
                class_id = int(box.cls[0])
                name = model.names[class_id]

                print(
                    f"Track ID: {track_id}, "
                    f"Object: {name}"
                )

    cv2.imshow("AI Object Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
