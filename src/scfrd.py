import time
import cv2 as cv
from PIL import Image
from scrfd import SCRFD, Threshold

from capture import open_camera

detector = SCRFD.from_path("./src/models/scrfd.onnx")
threshold = Threshold(probability=0.4)
cap = open_camera()

prev = time.perf_counter()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    faces = detector.detect(image, threshold=threshold)

    for face in faces:
        p1, p2 = face.bbox.upper_left, face.bbox.lower_right
        cv.rectangle(frame, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (0, 255, 0), 2)
        cv.putText(frame, f"{face.probability:.2f}", (int(p1.x), int(p1.y) - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for kp in face.keypoints.model_dump().values():
            cv.circle(frame, (int(kp["x"]), int(kp["y"])), 2, (0, 0, 255), -1)

    now = time.perf_counter()
    fps = 1.0 / (now - prev)
    prev = now
    cv.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv.imshow("SCRFD", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()