"""
Expressão facial visível: MobileNetV3-Small com fine-tuning no AffectNet (train_expression.py),
exportado para ONNX. A label descreve a configuração visível da cara, não o estado emocional.
"""
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import onnxruntime as ort

from src.config import EXPRESSION_MODEL, EXPRESSION_UNCERTAIN_BELOW, IMAGENET_MEAN, IMAGENET_STD, onnx_providers

# Índice de saída do modelo = pasta 0–7 do AffectNetCustom usado no treino. Não é a ordem oficial do
# AffectNet (0 = Neutral): foi confirmada pela matriz de confusão do BTFER e pelas fotos da galeria.
CLASSES = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral", "Contempt"]
BTFER_CLASSES = [c for c in CLASSES if c != "Contempt"]  # as 7 comuns ao BTFER (avaliação cross-dataset)
IMAGENET_MEAN, IMAGENET_STD = np.float32(IMAGENET_MEAN), np.float32(IMAGENET_STD)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


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

        probs = softmax(self.session.run(None, {self.input_name: blob})[0][0])
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        return ExpressionResult(CLASSES[best], confidence, dict(zip(CLASSES, probs.tolist())),
                                confidence < self.confidence_threshold)
