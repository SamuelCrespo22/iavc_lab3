
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

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.65
    ):
        available_providers = ort.get_available_providers()

        providers = []
        if "CoreMLExecutionProvider" in available_providers:
            providers.append("CoreMLExecutionProvider")

        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.confidence_threshold = confidence_threshold

        # O modelo InsightFace espera imagens de 96x96.
        if self.input_shape[1:] != [3, 96, 96]:
            raise ValueError(
                f"Unexpected model input shape: {self.input_shape}"
            )

        print("AgeGender model loaded successfully.")
        print("Input shape:", self.input_shape)
        print("Providers:", self.session.get_providers())

    def _get_age_range(self, age: int) -> str:
        lower = (age // 10) * 10
        upper = lower + 9
        return f"{lower}-{upper}"

    def predict(self, aligned_face: np.ndarray) -> AgeGenderResult:
        """
        aligned_face:
            Face crop in BGR format.

        Expected input:
            96x96 RGB, NCHW, float32, values in [0, 255].

        Output:
            Estimated age, age range, gender and confidence.
        """

        # 1. Resize to the model's expected dimensions.
        img = cv2.resize(aligned_face, (96, 96))

        # 2. Convert BGR to RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Convert HWC to CHW.
        img = np.transpose(img, (2, 0, 1))

        # 4. Convert to float32.
        # No external normalization: the model includes it internally.
        img = img.astype(np.float32)

        # 5. Add batch dimension.
        blob = np.expand_dims(img, axis=0)

        # 6. Run inference.
        preds = self.session.run(
            None,
            {self.input_name: blob}
        )[0][0]

        if len(preds) != 3:
            raise ValueError(
                f"Unexpected model output: {preds.shape}"
            )

        # 7. Gender classification.
        gender_logits = preds[:2]

        # Stable softmax for confidence estimation.
        gender_logits = gender_logits - np.max(gender_logits)
        probs = np.exp(gender_logits)
        probs = probs / np.sum(probs)

        gender_idx = int(np.argmax(probs))
        confidence = float(probs[gender_idx])

        # InsightFace mapping:
        # 0 = Female
        # 1 = Male
        gender_label = "Male" if gender_idx == 1 else "Female"

        is_uncertain = confidence < self.confidence_threshold

        if is_uncertain:
            gender_label = "Uncertain"

        # 8. Age regression.
        age = int(round(float(preds[2]) * 100))
        age = max(0, min(100, age))

        return AgeGenderResult(
            age=age,
            age_range=self._get_age_range(age),
            gender=gender_label,
            gender_confidence=round(confidence, 3),
            is_uncertain=is_uncertain
        )
