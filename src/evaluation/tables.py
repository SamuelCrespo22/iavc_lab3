"""Funções partilhadas pelas avaliações: CSV, tabelas Markdown e tamanho dos modelos."""
import csv
from pathlib import Path

import numpy as np
import onnx

from src.config import MODELS_DIR


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    return "\n".join(lines + ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows])


def model_size(name: str) -> tuple[float, float]:
    """(milhões de parâmetros, MB) de um modelo ONNX em models/, contando os pesos num <nome>.onnx.data."""
    path = MODELS_DIR / name
    graph = onnx.load(str(path), load_external_data=False).graph
    files = [path, path.with_name(path.name + ".data")]
    return (sum(int(np.prod(t.dims)) for t in graph.initializer) / 1e6,
            sum(p.stat().st_size for p in files if p.exists()) / 2**20)


def pct(value) -> str:
    return "—" if value is None or value == "" else f"{float(value):.1%}"
