"""
Expressão facial visível: MobileNetV3-Small com fine-tuning no AffectNet (train_expression.py),
exportado para ONNX. A label descreve a configuração visível da cara, não o estado emocional.
"""
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import onnxruntime as ort

from src.config import EXPRESSION_MODEL, EXPRESSION_UNCERTAIN_BELOW, onnx_providers

CLASSES = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ExpressionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    is_uncertain: bool


class ExpressionEstimator:
    def __init__(self, model_path=EXPRESSION_MODEL, confidence_threshold: float = EXPRESSION_UNCERTAIN_BELOW):
        self.session = ort.InferenceSession(str(model_path), providers=onnx_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold

    def predict(self, crop_bgr: np.ndarray) -> ExpressionResult:
        """crop_bgr: crop alinhado (224x224 com margem), como no treino."""
        img = cv.resize(cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        blob = np.transpose(img, (2, 0, 1))[np.newaxis].astype(np.float32)

        logits = self.session.run(None, {self.input_name: blob})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        best = int(np.argmax(probs))
        confidence = float(probs[best])
        return ExpressionResult(CLASSES[best], confidence, dict(zip(CLASSES, probs.tolist())),
                                confidence < self.confidence_threshold)
