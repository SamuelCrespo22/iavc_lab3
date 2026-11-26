"""
Arquiteturas do modelo de expressão, num só sítio: usado pelo treino, pela exportação para ONNX
e pela avaliação no BTFER. Ambas recebem 224x224 com a normalização do ImageNet.
"""
from pathlib import Path

import torch.nn as nn
from torchvision import models

from src.config import EXPRESSION_CHECKPOINT, MODELS_DIR

ARCHS = ("mobilenet", "resnet50")


def build_model(arch: str, num_classes: int, pretrained: bool) -> nn.Module:
    """pretrained=True carrega os pesos do ImageNet (treino); False só cria a rede (carregar um checkpoint)."""
    weights = "DEFAULT" if pretrained else None
    if arch == "mobilenet":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Arquitetura desconhecida: {arch} (opções: {', '.join(ARCHS)})")
    return model


def checkpoint_arch(checkpoint: dict) -> str:
    """Os checkpoints anteriores ao --arch não guardam a arquitetura: são da MobileNet."""
    return checkpoint.get("arch", "mobilenet")


def suffix(arch: str) -> str:
    """Sufixo dos ficheiros de saída: vazio para a MobileNet, para manter os nomes de sempre."""
    return "" if arch == "mobilenet" else f"_{arch}"


def checkpoint_path(arch: str) -> Path:
    return EXPRESSION_CHECKPOINT if arch == "mobilenet" else MODELS_DIR / f"expression_{arch}.pth"
