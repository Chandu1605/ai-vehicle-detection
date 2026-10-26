"""YOLO detections -> persistent C++ TrackManager/Kalman filter -> OpenCV display.

Run from the ai-vehicle-detection folder using its .venv Python interpreter.
"""
import subprocess
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent / 'adas-object-tracker' / 'build' / 'Release' / 'tracker_bridge.exe'
MODEL = HERE / 'yolo11n.pt'
ALLOWED_CLASSES = {'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
CONFIDENCE_THRESHOLD = 0.50


def main():
    if not BRIDGE.is_file():
        raise FileNotFoundError(f'C++ tracker bridge not found: {BRIDGE}')

    model = YOLO(str(MODEL) if MODEL.is_file() else 'yolo11n.pt')
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError('Could not open webcam (camera 0)')

    tracker = None
    try:
        tracker = subprocess.Popen(
            [str(BRIDGE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        frame_id = 0
        previous_time = time.perf_counter()

        while True:
            success, frame = camera.read()
            if not success:
                print('Camera stopped providing frames.')
                break

            now = time.perf_counter()
            dt = max(now - previous_time, 0.001)
            previous_time = now
            frame_id += 1

            inference_start = time.perf_counter()
            result = model(frame, verbose=False)[0]
            inference_ms = (time.perf_counter() - inference_start) * 1000

            detections = []
            boxes_to_draw = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id]
                if confidence < CONFIDENCE_THRESHOLD or label not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append((cx, cy, class_id))
                boxes_to_draw.append((int(x1), int(y1), int(x2), int(y2), label, confidence))

            # Exactly ONE request to C++ for each video frame, including zero detections.
            message = f'{frame_id} {dt:.6f} {len(detections)}\n'
            message += ''.join(
                f'{x:.3f} {y:.3f} {class_id}\n'
                for x, y, class_id in detections
            )
            tracker.stdin.write(message)
            tracker.stdin.flush()

            # Exactly ONE header followed by the declared number of track records.
            header = tracker.stdout.readline()
            if not header:
                raise RuntimeError(f'C++ bridge closed unexpectedly (exit={tracker.poll()})')
            fields = header.split()
            if len(fields) != 2:
                raise RuntimeError(f'Unexpected bridge header: {header!r}')
            returned_frame, track_count = map(int, fields)
            if returned_frame != frame_id:
                raise RuntimeError(
                    f'Frame mismatch: Python sent {frame_id}, C++ returned {returned_frame}'
                )

            tracks = []
            for _ in range(track_count):
                line = tracker.stdout.readline()
                if not line:
                    raise RuntimeError('C++ bridge returned an incomplete track response')
                values = line.split()
                if len(values) != 6:
                    raise RuntimeError(f'Unexpected track record: {line!r}')
                track_id = int(values[0])
                x, y, vx, vy = map(float, values[1:5])
                missed = int(values[5])
                tracks.append((track_id, x, y, vx, vy, missed))

            # Detection boxes show YOLO results; markers/IDs show C++ Kalman tracks.
            for x1, y1, x2, y2, label, confidence in boxes_to_draw:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
                cv2.putText(
                    frame, f'{label} {confidence:.2f}',
                    (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2,
                )

            for track_id, x, y, vx, vy, missed in tracks:
                if missed != 0:
                    continue  # Hide tracks during temporary detection loss.
                px, py = int(round(x)), int(round(y))
                cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
                cv2.putText(
                    frame, f'ID:{track_id} V:({vx:.1f},{vy:.1f})',
                    (px + 8, max(18, py - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2,
                )

            cv2.putText(
                frame, f'YOLO: {inference_ms:.0f} ms | Detections: {len(detections)} | Tracks: {len(tracks)}',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2,
            )
            cv2.imshow('YOLO + C++ Kalman Tracker (Q to quit)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if tracker is not None:
            if tracker.stdin is not None:
                try:
                    tracker.stdin.close()
                except OSError:
                    pass
            try:
                tracker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                tracker.terminate()
                tracker.wait(timeout=3)
            if tracker.stdout is not None:
                tracker.stdout.close()


if __name__ == '__main__':
    main()
