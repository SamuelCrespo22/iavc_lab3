"""
Identificação em galeria fechada (< 20 pessoas) com embeddings ArcFace.

- FaceEmbedder: crop alinhado 112x112 (BGR) -> embedding de 512 dimensões, norma 1.
- Gallery: template por pessoa (média dos embeddings de registo, renormalizada),
  similaridade cosseno e rejeição "Unknown" abaixo do threshold.

Privacidade: a galeria guarda apenas embeddings (nunca imagens) em gallery/,
que está no .gitignore. Os embeddings são dados biométricos: não partilhar.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import onnxruntime as ort

from src.config import (EMBEDDING_MODEL, GALLERY_DIR, MAX_IDENTITIES, PROVISIONAL_IDENTITY_THRESHOLD,
                        onnx_providers)
from src.pipeline.detection import identity_crop, largest_face

GALLERY_PATH = GALLERY_DIR / "gallery.npz"
THRESHOLD_PATH = GALLERY_DIR / "threshold.json"
UNKNOWN = "Unknown"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class FaceEmbedder:
    """ArcFace ResNet-50 (InsightFace w600k_r50)."""

    def __init__(self, model_path: Path = EMBEDDING_MODEL):
        self.session = ort.InferenceSession(str(model_path), providers=onnx_providers())
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, crop_112_bgr: np.ndarray) -> np.ndarray:
        # Pré-processamento do InsightFace para ArcFace: RGB, (x - 127.5) / 127.5, NCHW.
        img = cv.cvtColor(crop_112_bgr, cv.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        blob = np.transpose(img, (2, 0, 1))[np.newaxis]
        embedding = self.session.run(None, {self.input_name: blob})[0][0]
        return l2_normalize(embedding)


def embed_image(path: Path, detector, embedder: FaceEmbedder) -> np.ndarray | None:
    """Lê uma foto, deteta a maior cara e devolve o embedding (None se não houver cara)."""
    frame = cv.imread(str(path))
    face = largest_face(detector, frame) if frame is not None else None
    return embedder.embed(identity_crop(frame, face)) if face is not None else None


@dataclass
class IdentityResult:
    name: str     # pessoa reconhecida ou "Unknown"
    score: float  # similaridade cosseno com o template mais próximo (-1 a 1); nan se a galeria estiver vazia

    @property
    def is_known(self) -> bool:
        return self.name != UNKNOWN


def load_threshold() -> float:
    if THRESHOLD_PATH.exists():
        return float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    return PROVISIONAL_IDENTITY_THRESHOLD


def save_threshold(threshold: float, details: dict) -> None:
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold, **details}, indent=4))


class Gallery:
    """
    Galeria de identidades. Uma pessoa é aceite quando score > threshold.
    Com path=None fica só em memória (usado na avaliação).
    """

    def __init__(self, threshold: float | None = None, path: Path | None = GALLERY_PATH):
        self.path = path
        self.threshold = load_threshold() if threshold is None else threshold
        self.embeddings: dict[str, np.ndarray] = {}  # nome -> (n, 512)
        self._templates = np.empty((0, 512), dtype=np.float32)

    # ---- persistência ----
    @classmethod
    def load(cls, path: Path = GALLERY_PATH, threshold: float | None = None) -> "Gallery":
        gallery = cls(threshold=threshold, path=path)
        if path.exists():
            data = np.load(path, allow_pickle=False)
            for name in np.unique(data["labels"]):
                gallery.embeddings[str(name)] = data["embeddings"][data["labels"] == name]
            gallery._rebuild()
        return gallery

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        labels = np.array([n for n, e in self.embeddings.items() for _ in range(len(e))])
        stacked = (np.concatenate(list(self.embeddings.values()))
                   if self.embeddings else np.empty((0, 512), dtype=np.float32))
        np.savez(self.path, embeddings=stacked, labels=labels)

    # ---- gestão de identidades ----
    def names(self) -> list[str]:
        return list(self.embeddings)

    def enrol(self, name: str, embeddings: list[np.ndarray] | np.ndarray) -> None:
        embeddings = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if len(embeddings) == 0:
            raise ValueError(f"Sem embeddings para registar '{name}'.")
        if name == UNKNOWN:
            raise ValueError(f"'{UNKNOWN}' é um nome reservado.")
        if name not in self.embeddings and len(self.embeddings) >= MAX_IDENTITIES:
            raise ValueError(f"A galeria já tem {MAX_IDENTITIES} identidades (máximo).")
        previous = self.embeddings.get(name, np.empty((0, 512), dtype=np.float32))
        self.embeddings[name] = np.concatenate([previous, embeddings])
        self._rebuild()

    def remove(self, name: str) -> bool:
        removed = self.embeddings.pop(name, None) is not None
        self._rebuild()
        return removed

    def clear(self) -> None:
        """Apaga todas as identidades e o ficheiro da galeria."""
        self.embeddings.clear()
        self._rebuild()
        if self.path is not None and self.path.exists():
            self.path.unlink()

    def _rebuild(self) -> None:
        self._templates = (np.stack([l2_normalize(e.mean(axis=0)) for e in self.embeddings.values()])
                           if self.embeddings else np.empty((0, 512), dtype=np.float32))

    # ---- identificação ----
    def scores(self, embedding: np.ndarray) -> dict[str, float]:
        """Similaridade cosseno com o template de cada pessoa."""
        return dict(zip(self.embeddings, (self._templates @ embedding).tolist()))

    def match(self, embedding: np.ndarray) -> IdentityResult:
        if not self.embeddings:
            return IdentityResult(UNKNOWN, float("nan"))
        sims = self._templates @ embedding
        best = int(np.argmax(sims))
        score = float(sims[best])
        return IdentityResult(self.names()[best] if score > self.threshold else UNKNOWN, score)
