
import cv2

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Unable to open camera")
    raise SystemExit(1)

while True:
    success, frame = video.read()

    if not success:
        break

    cv2.imshow("AI Vehicle Detection - Camera Input", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
