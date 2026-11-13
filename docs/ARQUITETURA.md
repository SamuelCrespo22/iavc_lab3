## Divisão dos scripts por função.

---

#### capture.py

Recebe: OpenCV VideoCapture e webcam/video file.
Retorna: Frame, timestamp e frame rate.

#### detection.py

Recebe: SCRFD (comparar com alternativa lightweight).
Retorna: Face boxes, landmarks e confidence.

#### alignment.py

Recebe: landmarks e geometric alignment.
Retorna: normalised face crop.

#### attribute.py

Recebe: Age estimator, gender-label classifier e expression model.
Retorna: Age estimate/range, gender classification e expression probabilities.

#### identity.py

Recebe: Pre-trained face embedding model e cosine similarity para a galeria de templates.
Retorna: Identidade registada ou "Unknown".

#### interface.py

Recebe: OpenCV overlay.
Retorna: Live annotated stream e painel de status.

#### monitoring.py

Recebe: Timing e resource instrumentation.
Retorna: FPS, latência, CPU/GPU e uso de memória.

---


