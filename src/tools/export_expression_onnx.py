"""
Exporta o modelo de expressão (checkpoint do train_expression.py) para ONNX e confirma que
dá as mesmas saídas que o PyTorch. Os pesos podem ficar num ficheiro ao lado (<nome>.onnx.data):
nesse caso, os dois ficheiros são necessários.

    python -m src.tools.export_expression_onnx
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models

from src.config import EXPRESSION_CHECKPOINT, EXPRESSION_MODEL
from src.pipeline.expression import CLASSES


def load_model(checkpoint_path: Path) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=EXPRESSION_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=EXPRESSION_MODEL)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), args.output,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=17)

    # Verificação: o ONNX (em CPU) tem de dar as mesmas saídas que o PyTorch.
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        expected = model(x).numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    got = session.run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    same_class = bool((expected.argmax(1) == got.argmax(1)).all())

    files = [p.name for p in (args.output, args.output.with_name(args.output.name + ".data")) if p.exists()]
    print(f"Exportado para {args.output.parent}: {', '.join(files)}")
    print(f"Diferença máxima para o PyTorch: {diff:.1e}; mesma classe: {same_class}")
    if diff > 1e-3 or not same_class:
        raise SystemExit("A saída do ONNX não coincide com a do PyTorch.")


if __name__ == "__main__":
    main()
