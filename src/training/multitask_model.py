"""
Backbone partilhado pela expressão, género e idade (MobileNetV3-Small, ResNet-18 ou ResNet-50, pesos do ImageNet):
uma cabeça e uma loss por tarefa. A identidade continua no ArcFace (identity.py).
"""
import torch.nn as nn
from torchvision import models

from src.pipeline.expression import CLASSES

TASKS = {  # tarefa -> (nº de saídas, loss); a ordem é a das saídas do modelo e do ONNX
    "expression": (len(CLASSES), nn.CrossEntropyLoss()),
    "gender":     (2, nn.CrossEntropyLoss()),
    "age":        (1, nn.L1Loss()),  # alvo = idade / 10
}
ARCHS = ("mobilenet", "resnet18", "resnet50")


def build_backbone(arch: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Rede sem a camada de classificação do ImageNet; devolve (rede, nº de features)."""
    weights = "DEFAULT" if pretrained else None
    if arch == "mobilenet":
        backbone = models.mobilenet_v3_small(weights=weights)
        dim = backbone.classifier[3].in_features  # 1024
        backbone.classifier[3] = nn.Identity()
    else:
        backbone = getattr(models, arch)(weights=weights)
        dim = backbone.fc.in_features  # 512 / 2048
        backbone.fc = nn.Identity()
    return backbone, dim


class MultiTaskNet(nn.Module):
    def __init__(self, pretrained: bool = False, arch: str = "mobilenet"):
        super().__init__()
        self.backbone, dim = build_backbone(arch, pretrained)
        self.heads = nn.ModuleDict({t: nn.Linear(dim, n) for t, (n, _) in TASKS.items()})

    def forward(self, x):
        """Uma saída por tarefa, pela ordem de TASKS (em tuplo, como o ONNX precisa)."""
        features = self.backbone(x)
        return tuple(head(features) for head in self.heads.values())


def task_loss(task, out, y):
    """Loss de uma tarefa, só com amostras que têm label."""
    if task == "age":
        out, y = out.squeeze(1), y.float() / 10
    return TASKS[task][1](out, y)


def multitask_loss(outputs, labels, weights):
    """
    outputs: saídas do modelo (pela ordem de TASKS). labels[t] = -1 nas amostras sem label dessa tarefa
    (o AffectNet não tem idade, o UTKFace não tem expressão). Devolve 0.0 (não um tensor) se o batch não tiver labels.
    """
    total = 0.0
    for t, out in zip(TASKS, outputs):
        mask = labels[t] >= 0
        if mask.any():
            total = total + weights[t] * task_loss(t, out[mask], labels[t][mask])
    return total
