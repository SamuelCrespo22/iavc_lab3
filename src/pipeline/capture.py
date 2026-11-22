"""Abertura da fonte de vídeo: webcam (índice, ex. 0) ou ficheiro de vídeo (caminho)."""
import sys

import cv2 as cv

from src.config import CAMERA_HEIGHT, CAMERA_WIDTH


def parse_source(source: str | int) -> int | str:
    return int(source) if str(source).isdigit() else str(source)


def open_source(source: str | int = 0, width: int = CAMERA_WIDTH, height: int = CAMERA_HEIGHT) -> cv.VideoCapture:
    source = parse_source(source)
    if isinstance(source, int):
        # AVFoundation só existe em macOS; nos outros sistemas o OpenCV escolhe o backend.
        backend = cv.CAP_AVFOUNDATION if sys.platform == "darwin" else cv.CAP_ANY
        cap = cv.VideoCapture(source, backend)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {source}")
    return cap
