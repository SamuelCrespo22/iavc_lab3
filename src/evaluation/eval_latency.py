"""
Latência de inferência de cada modelo da aplicação, com o CoreML (como na aplicação) e só com a CPU: só a
inferência de uma cara (ou de um frame, no detetor), sem recortes. Também regista em quantos blocos o CoreML parte
cada modelo, a partir do log do ONNX Runtime: as operações que o CoreML não suporta ficam na CPU, entre blocos.

    python -m src.evaluation.eval_latency     # -> results/system/model_latency.csv
"""
import json
import os
import re
import tempfile

import numpy as np
import onnxruntime as ort

from src.config import MODELS_DIR, RESULTS_DIR
from src.evaluation.eval_attributes import latency_ms
from src.evaluation.tables import write_csv

MODELS = {"scrfd.onnx": 640, "w600k_r50.onnx": 112, "genderage.onnx": 96, "expression/expression_best.onnx": 224}
COREML = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
PARTITIONS = re.compile(r"partitions supported by CoreML: (\d+)(?: number of nodes in the graph: (\d+) "
                        r"number of nodes supported by CoreML: (\d+))?")


def coreml_session(path):
    """Sessão com o CoreML e (blocos, nós no CoreML, nós no total), lidos do log do ONNX Runtime (stderr)."""
    options = ort.SessionOptions()
    options.log_severity_level = 1  # INFO: com um só bloco, a contagem não aparece como aviso
    with tempfile.TemporaryFile(mode="w+") as log:
        stderr = os.dup(2)
        os.dup2(log.fileno(), 2)
        try:
            session = ort.InferenceSession(str(path), options, providers=COREML)
        finally:
            os.dup2(stderr, 2)
            os.close(stderr)
        log.seek(0)
        found = PARTITIONS.search(log.read())
    return session, tuple(int(v) if v else None for v in (found[1], found[3], found[2])) if found else (None,) * 3


def run_ms(session, size) -> float:
    """Mediana de 5 medições de 50 inferências (o CoreML varia bastante entre medições)."""
    feed = {session.get_inputs()[0].name: np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)}
    return float(np.median([latency_ms(lambda: session.run(None, feed), runs=50) for _ in range(5)]))


def main():
    models = dict(MODELS)
    for run in sorted(p for p in (MODELS_DIR / "multitask").iterdir() if (p / "multitask.onnx").exists()):
        models[f"multitask/{run.name}/multitask.onnx"] = json.loads((run / "config.json").read_text())["input_size"]
    has_coreml = "CoreMLExecutionProvider" in ort.get_available_providers()

    rows = []
    for name, size in models.items():
        path = MODELS_DIR / name
        cpu = run_ms(ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]), size)
        coreml, (blocks, in_coreml, nodes) = coreml_session(path) if has_coreml else (None, (None, None, None))
        rows.append({"model": name, "input_px": size, "coreml_ms": run_ms(coreml, size) if coreml else None,
                     "cpu_ms": cpu, "coreml_partitions": blocks, "nodes_in_coreml": in_coreml, "nodes": nodes})
        r = rows[-1]
        print(f"{name:42} {size:4} px | CoreML {r['coreml_ms'] or float('nan'):6.1f} ms "
              f"({blocks} blocos, {in_coreml}/{nodes} nós) | CPU {cpu:6.1f} ms", flush=True)
    write_csv(RESULTS_DIR / "system" / "model_latency.csv", rows)


if __name__ == "__main__":
    main()
