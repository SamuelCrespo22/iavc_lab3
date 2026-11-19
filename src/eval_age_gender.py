import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass

@dataclass
class AgeGenderResult:
    age: int
    age_range: str
    gender: str
    gender_confidence: float
    is_uncertain: bool

class AgeGenderEstimator:
    def __init__(self, model_path: str, confidence_threshold: float = 0.65):
        # Usa CPUExecutionProvider (ou CoreMLExecutionProvider no Mac)
        self.session = ort.InferenceSession(
            model_path, 
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold

    def _get_age_range(self, age: int) -> str:
        lower = (age // 10) * 10
        return f"{lower}-{lower + 9}"

    def predict(self, aligned_face: np.ndarray) -> AgeGenderResult:
        """
        aligned_face: imagem BGR recortada e alinhada (96x96 ou 112x112)
        """
        # Pré-processamento padrão para modelos InsightFace genderage
        img = cv2.resize(aligned_face, (96, 96))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 128.0
        blob = np.expand_dims(img, axis=0)

        # Inferência
        preds = self.session.run(None, {self.input_name: blob})[0][0]

        # Saída típica do genderage.onnx: [logit_female, logit_male, age_normalized]
        gender_logits = preds[:2]
        probs = np.exp(gender_logits) / np.sum(np.exp(gender_logits))
        gender_idx = int(np.argmax(probs))
        confidence = float(probs[gender_idx])

        gender_label = "Male" if gender_idx == 1 else "Female"
        is_uncertain = confidence < self.confidence_threshold
        if is_uncertain:
            gender_label = "Uncertain"

        age = int(round(float(preds[2]) * 100))
        age = max(0, min(116, age))  # Limite do UTKFace/IMDB-WIKI

        return AgeGenderResult(
            age=age,
            age_range=self._get_age_range(age),
            gender=gender_label,
            gender_confidence=round(confidence, 2),
            is_uncertain=is_uncertain
        )