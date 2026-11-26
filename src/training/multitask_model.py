"""
Backbone partilhado pela expressão, género e idade (MobileNetV3-Small, ResNet-18 ou ResNet-50, pesos do ImageNet):
uma cabeça e uma loss por tarefa. A identidade continua no ArcFace (identity.py).
"""
import torch.nn as nn
from torchvision import models

from src.pipeline.expression import CLASSES
from src.pipeline.multitask import GENDERS

TASKS = {  # tarefa -> (nº de saídas, loss)
    "expression": (len(CLASSES), nn.CrossEntropyLoss()),
    "gender":     (2, nn.CrossEntropyLoss()),
    "age":        (1, nn.L1Loss()),  # alvo = idade / 10
}


ARCHS = ("mobilenet", "resnet18", "resnet50")
RESNET_STAGES = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")  # pela ordem da rede


def build_backbone(arch: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Rede sem a camada de classificação do ImageNet; devolve (rede, nº de features)."""
    weights = "DEFAULT" if pretrained else None
    if arch == "mobilenet":
        backbone = models.mobilenet_v3_small(weights=weights)
        dim = backbone.classifier[3].in_features  # 1024
        backbone.classifier[3] = nn.Identity()
    elif arch in ("resnet18", "resnet50"):
        backbone = getattr(models, arch)(weights=weights)
        dim = backbone.fc.in_features  # 512 / 2048
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Arquitetura desconhecida: {arch} (opções: {', '.join(ARCHS)})")
    return backbone, dim


class MultiTaskNet(nn.Module):
    def __init__(self, pretrained: bool = False, arch: str = "mobilenet", freeze_until: str | None = None):
        """freeze_until (só ResNet): congela da entrada até esse bloco, inclusive ("layer4" = só as cabeças treinam)."""
        super().__init__()
        self.backbone, dim = build_backbone(arch, pretrained)
        self.heads = nn.ModuleDict({t: nn.Linear(dim, n) for t, (n, _) in TASKS.items()})
        self.frozen = []
        if freeze_until:
            if not arch.startswith("resnet"):
                raise ValueError("freeze_until só existe para as ResNet")
            stages = RESNET_STAGES[:RESNET_STAGES.index(freeze_until) + 1]
            self.frozen = [getattr(self.backbone, s) for s in stages]
            for module in self.frozen:
                module.requires_grad_(False)

    def train(self, mode: bool = True):
        """A parte congelada fica sempre em eval: a BatchNorm não muda as médias com os dados novos."""
        super().train(mode)
        for module in self.frozen:
            module.eval()
        return self

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
