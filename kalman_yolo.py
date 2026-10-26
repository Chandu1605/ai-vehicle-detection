r"""YOLO + C++ Kalman tracker: duplicate filtering, trails, FPS, and road video.

Examples (PowerShell, from ai-vehicle-detection):
    .\.venv\Scripts\python.exe .\kalman_yolo.py
    .\.venv\Scripts\python.exe .\kalman_yolo.py --source .\road.mp4
    .\.venv\Scripts\python.exe .\kalman_yolo.py --source .\road.mp4 --vehicles-only

Press Q to quit. Requires the existing class-aware tracker_bridge.exe.
"""

import argparse
import subprocess
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent / 'adas-object-tracker' / 'build' / 'Release' / 'tracker_bridge.exe'
MODEL = HERE / 'yolo11n.pt'
ALLOWED_CLASSES = {'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
CONFIDENCE_THRESHOLD = 0.60
NMS_IOU = 0.45
TRAIL_LENGTH = 75
FPS_SMOOTHING = 0.15


def intersection_over_smaller(a, b):
    """Intersection area divided by the smaller box area."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0 else 0.0


def remove_nested_person_boxes(candidates):
    """Remove likely duplicate person boxes contained in a larger person box.

    This is intentionally limited to person boxes; a smaller vehicle or a
    genuinely occluded second person should not be removed indiscriminately.
    """
    kept = []
    for candidate in sorted(candidates, key=lambda item: item['confidence'], reverse=True):
        if candidate['label'] != 'person':
            kept.append(candidate)
            continue
        box = candidate['box']
        duplicate = False
        for other in candidates:
            if other is candidate or other['label'] != 'person':
                continue
            other_box = other['box']
            area = (box[2] - box[0]) * (box[3] - box[1])
            other_area = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
            # Remove only substantially smaller boxes nearly contained in a
            # larger person box, with no weaker confidence than the small one.
            if (other_area >= area * 1.8
                    and other['confidence'] >= candidate['confidence'] - 0.15
                    and intersection_over_smaller(box, other_box) >= 0.90):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def draw_metrics(frame, fps, inference_ms, detection_count, track_count, source_label):
    lines = [
        f'FPS: {fps:.1f} | YOLO: {inference_ms:.0f} ms',
        f'Detections: {detection_count} | Tracks: {track_count} | {source_label}',
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.52, 1
    width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 20
    width = min(width, frame.shape[1])
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + i * 26), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def parse_source(source):
    if source.isdigit():
        return int(source), f'Camera {source}'
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Video file not found: {path}')
    return str(path), path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='0', help='Webcam index (0) or video file path')
    parser.add_argument('--vehicles-only', action='store_true',
                        help='Track road vehicles and bicycles, not people')
    parser.add_argument('--show-missed', action='store_true',
                        help='Show predicted positions during brief detection loss')
    args = parser.parse_args()

    if not BRIDGE.is_file():
        raise FileNotFoundError(f'C++ tracker bridge not found: {BRIDGE}')

    source, source_label = parse_source(args.source)
    model = YOLO(str(MODEL) if MODEL.is_file() else 'yolo11n.pt')
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError(f'Could not open video source: {args.source}')

    tracker = None
    try:
        tracker = subprocess.Popen(
            [str(BRIDGE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        frame_id = 0
        previous_time = time.perf_counter()
        frame_start_previous = None
        smoothed_fps = 0.0
        track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
        allowed = VEHICLE_CLASSES if args.vehicles_only else ALLOWED_CLASSES

        while True:
            success, frame = camera.read()
            if not success:
                print('Video ended or camera stopped providing frames.')
                break

            now = time.perf_counter()
            if frame_start_previous is not None:
                instant_fps = 1.0 / max(now - frame_start_previous, 0.001)
                smoothed_fps = (FPS_SMOOTHING * instant_fps
                                + (1.0 - FPS_SMOOTHING) * smoothed_fps) if smoothed_fps else instant_fps
            frame_start_previous = now
            dt = max(now - previous_time, 0.001)
            previous_time = now
            frame_id += 1

            inference_start = time.perf_counter()
            result = model(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_IOU,
                           verbose=False)[0]
            inference_ms = (time.perf_counter() - inference_start) * 1000

            candidates = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id]
                if confidence < CONFIDENCE_THRESHOLD or label not in allowed:
                    continue
                coords = tuple(map(float, box.xyxy[0].tolist()))
                candidates.append({
                    'box': coords, 'class_id': class_id,
                    'label': label, 'confidence': confidence,
                })

            candidates = remove_nested_person_boxes(candidates)
            detections = []
            for item in candidates:
                x1, y1, x2, y2 = item['box']
                detections.append(((x1 + x2) / 2, (y1 + y2) / 2, item['class_id']))

            # Exactly one request per frame, even when detections is empty.
            message = f'{frame_id} {dt:.6f} {len(detections)}\n'
            message += ''.join(f'{x:.3f} {y:.3f} {class_id}\n'
                               for x, y, class_id in detections)
            tracker.stdin.write(message)
            tracker.stdin.flush()

            header = tracker.stdout.readline()
            if not header:
                raise RuntimeError(f'C++ bridge stopped unexpectedly (exit={tracker.poll()})')
            fields = header.split()
            if len(fields) != 2:
                raise RuntimeError(f'Invalid C++ header: {header!r}')
            returned_frame, track_count = map(int, fields)
            if returned_frame != frame_id:
                raise RuntimeError(f'Frame mismatch: sent {frame_id}, received {returned_frame}')

            tracks = []
            for _ in range(track_count):
                line = tracker.stdout.readline()
                if not line:
                    raise RuntimeError('Incomplete C++ track response')
                values = line.split()
                if len(values) != 6:
                    raise RuntimeError(f'Invalid track record: {line!r}')
                track_id = int(values[0])
                x, y, vx, vy = map(float, values[1:5])
                missed = int(values[5])
                tracks.append((track_id, x, y, vx, vy, missed))

            for item in candidates:
                x1, y1, x2, y2 = map(int, item['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
                cv2.putText(frame, f"{item['label']} {item['confidence']:.2f}",
                            (x1, max(80, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 180, 0), 2, cv2.LINE_AA)

            live_ids = {track_id for track_id, *_ in tracks}
            for old_id in list(track_history):
                if old_id not in live_ids:
                    del track_history[old_id]

            for track_id, x, y, vx, vy, missed in tracks:
                if missed and not args.show_missed:
                    continue
                px, py = int(round(x)), int(round(y))
                trail = track_history[track_id]
                if missed == 0:
                    trail.append((px, py))
                points = list(trail)
                for i in range(1, len(points)):
                    thickness = 2 + (3 * i // max(len(points) - 1, 1))
                    cv2.line(frame, points[i - 1], points[i], (0, 165, 255),
                             thickness, cv2.LINE_AA)
                color = (0, 255, 0) if missed == 0 else (150, 150, 150)
                cv2.circle(frame, (px, py), 7, color, -1, cv2.LINE_AA)
                cv2.putText(frame, f'ID:{track_id} V:({vx:.1f},{vy:.1f})',
                            (px + 10, max(78, py - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, color, 2, cv2.LINE_AA)

            draw_metrics(frame, smoothed_fps, inference_ms,
                         len(detections), len(tracks), source_label)
            # cv2.imshow('YOLO + C++ Kalman Tracker (Q to quit)', frame)
            cv2.imshow("ADAS - YOLO + C++ Kalman Tracker", frame)
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
