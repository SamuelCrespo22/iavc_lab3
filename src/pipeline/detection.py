"""Deteção de caras com SCRFD e os recortes usados pelos modelos seguintes."""
import cv2 as cv
import numpy as np
from PIL import Image
from scrfd import SCRFD, Face, Threshold

from src.config import DETECTION_THRESHOLD, DETECTOR_MODEL, MODELS_DIR, onnx_providers
from src.pipeline.alignment import align_face, keypoints_to_array


def load_detector(model_name: str = DETECTOR_MODEL) -> SCRFD:
    return SCRFD.from_path(str(MODELS_DIR / model_name), providers=onnx_providers())


def detect(detector: SCRFD, frame_bgr: np.ndarray, probability: float = DETECTION_THRESHOLD) -> list[Face]:
    image = Image.fromarray(cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB))
    return detector.detect(image, threshold=Threshold(probability=probability))


def face_area(face: Face) -> float:
    p1, p2 = face.bbox.upper_left, face.bbox.lower_right
    return max(0.0, p2.x - p1.x) * max(0.0, p2.y - p1.y)


def largest_face(detector: SCRFD, frame_bgr: np.ndarray,
                 probability: float = DETECTION_THRESHOLD) -> Face | None:
    """A maior cara da imagem (pessoa a registar)."""
    faces = detect(detector, frame_bgr, probability)
    return max(faces, key=face_area) if faces else None


def identity_crop(frame_bgr: np.ndarray, face: Face) -> np.ndarray:
    """Crop 112x112 alinhado com o template ArcFace."""
    return align_face(frame_bgr, keypoints_to_array(face), size=112)


def attribute_crop(frame_bgr: np.ndarray, face: Face, size: int = 96, scale: float = 1.5) -> np.ndarray:
    """
    Crop esperado pelo genderage.onnx do InsightFace: centrado na bounding box,
    lado = 1.5x o maior lado da cara, sem rotação, redimensionado para 96x96.
    """
    p1, p2 = face.bbox.upper_left, face.bbox.lower_right
    cx, cy = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
    s = size / (max(p2.x - p1.x, p2.y - p1.y) * scale)
    M = np.array([[s, 0, size / 2 - s * cx],
                  [0, s, size / 2 - s * cy]], dtype=np.float32)
    return cv.warpAffine(frame_bgr, M, (size, size), borderValue=0)


def expression_crop(frame_bgr: np.ndarray, face: Face) -> np.ndarray:
    """Crop 224x224 alinhado com margem, usado pelo modelo de expressão."""
    return align_face(frame_bgr, keypoints_to_array(face), size=224, margin=0.2)
