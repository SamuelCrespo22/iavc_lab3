"""
Monitorização em tempo real: FPS, latência por fase, CPU e memória do processo.

Uso no ciclo principal:
    with monitor.stage("detection"):
        ...
    monitor.end_frame()

A latência de cada fase é a soma por frame (ex.: a identidade corre uma vez por cara).
Ao sair, save() grava as médias da sessão em results/system/run_<data>.csv.
"""
import csv
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

import psutil

from src.config import RESULTS_DIR, ROOT_DIR

SYSTEM_RESULTS_DIR = RESULTS_DIR / "system"


class Monitor:
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing      # peso da média móvel (mais alto = mais estável)
        self.fps = 0.0
        self.latency_ms: dict[str, float] = {}
        self.cpu_percent = 0.0
        self.memory_mb = 0.0

        self._process = psutil.Process()
        self._process.cpu_percent(None)  # a primeira leitura é sempre 0
        self._current = defaultdict(float)
        self._totals = defaultdict(float)
        self._frames = 0
        self._frame_time = 0.0
        self._last_frame: float | None = None
        self._last_resources = 0.0
        self._resource_samples: list[tuple[float, float]] = []

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._current[name] += (time.perf_counter() - start) * 1000

    def end_frame(self) -> None:
        now = time.perf_counter()
        if self._last_frame is not None:
            dt = now - self._last_frame
            self._frame_time += dt
            self.fps = 1 / dt if self.fps == 0 else self.smoothing * self.fps + (1 - self.smoothing) / dt
        self._last_frame = now

        for name in self._current.keys() | self.latency_ms.keys():  # fase que não correu conta 0 ms
            ms = self._current.get(name, 0.0)
            self.latency_ms[name] = self.smoothing * self.latency_ms.get(name, ms) + (1 - self.smoothing) * ms
            self._totals[name] += ms
        self._current.clear()
        self._frames += 1

        if now - self._last_resources >= 1.0:
            self._last_resources = now
            self.cpu_percent = self._process.cpu_percent(None)
            self.memory_mb = self._process.memory_info().rss / 2**20
            self._resource_samples.append((self.cpu_percent, self.memory_mb))

    def pause(self) -> None:
        """Chamar ao parar a câmara, para a pausa não contar no FPS."""
        self._last_frame = None

    def save(self, info: dict | None = None) -> str | None:
        if self._frames < 2:
            return None
        SYSTEM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = SYSTEM_RESULTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
        cpu = [c for c, _ in self._resource_samples[1:]] or [0.0]
        rows = [("frames", self._frames),
                ("fps_mean", (self._frames - 1) / self._frame_time),
                *((f"latency_ms_{name}", total / self._frames) for name, total in self._totals.items()),
                ("cpu_percent_mean", sum(cpu) / len(cpu)),
                ("memory_mb_max", max((m for _, m in self._resource_samples), default=0.0)),
                *((key, value) for key, value in (info or {}).items())]
        with path.open("w", newline="") as f:
            csv.writer(f).writerows([("metric", "value"), *rows])
        return str(path.relative_to(ROOT_DIR))
