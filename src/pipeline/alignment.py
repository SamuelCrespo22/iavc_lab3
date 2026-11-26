"""
Alinhamento por semelhança (rotação, escala uniforme e translação): leva os 5 landmarks do SCRFD para as
posições do template ArcFace. Usado pela identidade (112x112) e pela expressão (224x224 com margem).
"""
import cv2 as cv
import numpy as np

# Template ArcFace (InsightFace) para 112x112, na ordem:
# left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def keypoints_to_array(face) -> np.ndarray:
    k = face.keypoints
    return np.array([[p.x, p.y] for p in
                     (k.left_eye, k.right_eye, k.nose, k.left_mouth, k.right_mouth)],
                    dtype=np.float32)


def align_face(frame: np.ndarray, kps: np.ndarray, size: int = 112,
               margin: float = 0.0) -> np.ndarray:
    """
    frame: imagem original (BGR), kps: (5,2) em coordenadas da imagem.
    size: lado do crop de saída. margin: >0 afasta a cara das bordas
    (ex.: 0.2 inclui mais cabelo e queixo, usado na expressão).
    """
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    if margin > 0:
        center = np.array([size / 2, size / 2], dtype=np.float32)
        dst = (dst - center) * (1 - margin) + center

    M, _ = cv.estimateAffinePartial2D(kps, dst, method=cv.LMEDS)
    return cv.warpAffine(frame, M, (size, size), borderValue=0)
