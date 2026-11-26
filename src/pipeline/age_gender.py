"""
Idade e género com o genderage.onnx do InsightFace: crop 96x96 (attribute_crop), uma saída com
[logit mulher, logit homem, idade / 100]. A normalização da imagem está dentro do modelo.
"""
from dataclasses import dataclass

import cv2 as cv
import numpy as np
import onnxruntime as ort

from src.config import AGE_GENDER_MODEL, GENDER_UNCERTAIN_BELOW, onnx_providers
from src.pipeline.expression import softmax

GENDERS = ["Female", "Male"]  # índice da saída do InsightFace


@dataclass
class AgeGenderResult:
    age: int
    age_range: str
    gender: str  # "Uncertain" abaixo do limiar de confiança
    gender_confidence: float
    is_uncertain: bool


def age_gender_result(age_years: float, gender_probs: np.ndarray, genders: list[str],
                      uncertain_below: float) -> AgeGenderResult:
    """Resultado comum ao genderage e ao multitarefa: idade limitada a 0–100 e década (ex.: 30-39)."""
    age = int(np.clip(round(age_years), 0, 100))
    best = int(gender_probs.argmax())
    confidence = float(gender_probs[best])
    uncertain = confidence < uncertain_below
    return AgeGenderResult(age, f"{age // 10 * 10}-{age // 10 * 10 + 9}", "Uncertain" if uncertain else genders[best],
                           round(confidence, 3), uncertain)


class AgeGenderEstimator:
    def __init__(self, model_path=AGE_GENDER_MODEL, confidence_threshold: float = GENDER_UNCERTAIN_BELOW):
        self.session = ort.InferenceSession(str(model_path), providers=onnx_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold

    def predict(self, crop_bgr: np.ndarray) -> AgeGenderResult:
        """crop_bgr: attribute_crop (96x96, BGR). Entrada do modelo: RGB, NCHW, valores 0–255."""
        img = cv.cvtColor(cv.resize(crop_bgr, (96, 96)), cv.COLOR_BGR2RGB).astype(np.float32)
        pred = self.session.run(None, {self.input_name: np.transpose(img, (2, 0, 1))[np.newaxis]})[0][0]
        return age_gender_result(float(pred[2]) * 100, softmax(pred[:2]), GENDERS, self.confidence_threshold)
