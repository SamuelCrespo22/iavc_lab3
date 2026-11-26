"""
MobileNetV3-Small partilhada pela expressão, género e idade: um backbone, uma cabeça e uma loss por tarefa.
A identidade continua no ArcFace (identity.py).
"""
import torch.nn as nn
from torchvision import models

from src.pipeline.expression import CLASSES

GENDERS = ["Male", "Female"]  # índice = label do UTKFace (0 = homem, 1 = mulher)

TASKS = {  # tarefa -> (nº de saídas, loss)
    "expression": (len(CLASSES), nn.CrossEntropyLoss()),
    "gender":     (2, nn.CrossEntropyLoss()),
    "age":        (1, nn.L1Loss()),  # alvo = idade / 10
}


class MultiTaskNet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        dim = self.backbone.classifier[3].in_features  # 1024
        self.backbone.classifier[3] = nn.Identity()
        self.heads = nn.ModuleDict({t: nn.Linear(dim, n) for t, (n, _) in TASKS.items()})

    def forward(self, x):
        features = self.backbone(x)
        return {t: head(features) for t, head in self.heads.items()}


def task_loss(task, out, y):
    """Loss de uma tarefa, só com amostras que têm label."""
    if task == "age":
        out, y = out.squeeze(1), y.float() / 10
    return TASKS[task][1](out, y)


def multitask_loss(outputs, labels, weights):
    """
    labels[t] = -1 nas amostras sem label dessa tarefa (o AffectNet não tem idade, o UTKFace não tem expressão).
    Devolve 0.0 (não um tensor) se o batch não tiver nenhuma label.
    """
    total = 0.0
    for t, y in labels.items():
        mask = y >= 0
        if mask.any():
            total = total + weights[t] * task_loss(t, outputs[t][mask], y[mask])
    return total
