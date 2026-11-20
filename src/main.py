import time
import cv2 as cv
from PIL import Image
from scrfd import SCRFD, Threshold

from capture import open_camera
from alignment import align_face, keypoints_to_array
from eval_age_gender import AgeGenderEstimator
import numpy as np 
import onnxruntime as ort

detector = SCRFD.from_path(
    "./models/scrfd.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)

age_gender = AgeGenderEstimator("./models/genderage.onnx")

expression_session = ort.InferenceSession(
    "./models/expression_best.onnx",
    providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)

expression_input = expression_session.get_inputs()[0].name
expression_output = expression_session.get_outputs()[0].name

expression_classes = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

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
        kps = keypoints_to_array(face)
        crop_id   = align_face(frame, kps, size=112)               # ArcFace (Stage 3)
        crop_attr = align_face(frame, kps, size=224, margin=0.2)   # idade/género/expressão
        cv.imshow("aligned", crop_id)   # para validar visualmente

        result = age_gender.predict(crop_attr) # Age and gender estimation

        expression_image = cv.cvtColor(crop_attr, cv.COLOR_BGR2RGB)
        expression_image = cv.resize(expression_image, (224, 224))
        expression_image = expression_image.astype(np.float32) / 255.0

        expression_image = (
            expression_image - np.array([0.485, 0.456, 0.406], dtype=np.float32)
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        expression_image = np.transpose(expression_image, (2, 0, 1))
        expression_image = np.expand_dims(expression_image, axis=0).astype(np.float32)

        expression_scores = expression_session.run(
            [expression_output],
            {expression_input: expression_image}
        )[0][0]

        expression_probabilities = np.exp(expression_scores - np.max(expression_scores))
        expression_probabilities /= np.sum(expression_probabilities)

        expression_index = int(np.argmax(expression_probabilities))
        expression = expression_classes[expression_index]
        expression_confidence = float(expression_probabilities[expression_index])

        p1, p2 = face.bbox.upper_left, face.bbox.lower_right

        x1, y1 = int(p1.x), int(p1.y)
        x2, y2 = int(p2.x), int(p2.y)

        cv.rectangle(frame, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (0, 255, 0), 2)

        # Display predictions
        gender_text = result.gender
        age_text = f"{result.age} years"

        cv.putText(frame, f"{gender_text} | {age_text}", (x1, y1 - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv.putText(frame, f"{expression} ({expression_confidence:.2f})", (x1, y1 - 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv.putText(frame, f"{face.probability:.2f}", (int(p1.x), int(p1.y) - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        for kp in face.keypoints.model_dump().values():
            cv.circle(frame, (int(kp["x"]), int(kp["y"])), 2, (0, 0, 255), -1)

    now = time.perf_counter()
    fps = 1.0 / (now - prev)
    prev = now
    cv.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv.imshow("Facial Analysis", frame)
    cv.imshow("SCRFD", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()