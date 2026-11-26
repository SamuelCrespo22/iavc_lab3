"""
Modelo multitarefa (train_multitask.py, exportado para ONNX): uma MobileNetV3-Small dá a expressão, o género
e a idade a partir de um só crop (multitask_crop). Devolve os mesmos resultados que age_gender.py e expression.py.
"""
import cv2 as cv
import numpy as np
import onnxruntime as ort

from src.config import EXPRESSION_UNCERTAIN_BELOW, GENDER_UNCERTAIN_BELOW, onnx_providers
from src.pipeline.age_gender import AgeGenderResult
from src.pipeline.expression import CLASSES, ExpressionResult

GENDERS = ["Male", "Female"]  # índice = label do UTKFace (0 = homem, 1 = mulher)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class MultiTaskEstimator:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(str(model_path), providers=onnx_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.size = self.session.get_inputs()[0].shape[-1]  # cada modelo traz o seu tamanho de entrada
        self.output_names = [o.name for o in self.session.get_outputs()]  # expression, gender, age
        self.gender_uncertain_below = GENDER_UNCERTAIN_BELOW

    def predict(self, crop_bgr: np.ndarray) -> tuple[AgeGenderResult, ExpressionResult]:
        """crop_bgr: multitask_crop na aplicação, ou uma imagem já recortada pela cara (datasets)."""
        img = cv.resize(cv.cvtColor(crop_bgr, cv.COLOR_BGR2RGB), (self.size, self.size), interpolation=cv.INTER_AREA)
        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        outputs = self.session.run(None, {self.input_name: np.transpose(img, (2, 0, 1))[np.newaxis]})
        out = {name: o[0] for name, o in zip(self.output_names, outputs)}

        expr = softmax(out["expression"])
        best = int(expr.argmax())
        expression = ExpressionResult(CLASSES[best], float(expr[best]), dict(zip(CLASSES, expr.tolist())),
                                      float(expr[best]) < EXPRESSION_UNCERTAIN_BELOW)

        gender = softmax(out["gender"])
        g = int(gender.argmax())
        age = int(np.clip(round(float(out["age"][0]) * 10), 0, 100))
        uncertain = float(gender[g]) < self.gender_uncertain_below
        age_gender = AgeGenderResult(age, f"{age // 10 * 10}-{age // 10 * 10 + 9}",
                                     "Uncertain" if uncertain else GENDERS[g], round(float(gender[g]), 3), uncertain)
        return age_gender, expression
