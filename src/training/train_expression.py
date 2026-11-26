"""
Fine-tuning da MobileNetV3-Small (ImageNet) para a expressão no AffectNetCustom (pastas 0–7, ver CLASSES em
expression.py), com pesos por classe na loss. A melhor época é a do maior F1 macro na validação.

    python -m src.training.train_expression     # -> models/expression/expression_best.{pth,onnx,onnx.data}

O teste fica para a avaliação (python -m src.evaluation.eval_attributes --model separate), com o ONNX.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from src.config import AFFECTNET_DIR, EXPRESSION_CHECKPOINT, EXPRESSION_MODEL, IMAGENET_MEAN, IMAGENET_STD
from src.pipeline.expression import CLASSES
from src.training.train_multitask import export_onnx

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
PATIENCE = 4
IMAGE_SIZE = 224
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def build_model(pretrained: bool = False) -> nn.Module:
    model = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    pred, true = [], []
    for images, labels in loader:
        pred.extend(model(images.to(DEVICE)).argmax(1).cpu().numpy())
        true.extend(labels.numpy())
    return {"accuracy": float(accuracy_score(true, pred)), "f1_macro": float(f1_score(true, pred, average="macro"))}


def main():
    normalize = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(10), transforms.ColorJitter(0.15, 0.15, 0.1), *normalize])
    eval_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), *normalize])
    train_set = datasets.ImageFolder(AFFECTNET_DIR / "train", transform=train_tf)
    val_set = datasets.ImageFolder(AFFECTNET_DIR / "val", transform=eval_tf)
    assert train_set.classes == [str(i) for i in range(len(CLASSES))], train_set.classes
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)
    print(f"{DEVICE} | treino {len(train_set)} | validação {len(val_set)}")

    model = build_model(pretrained=True).to(DEVICE)
    counts = np.bincount(train_set.targets, minlength=len(CLASSES))
    class_weights = torch.tensor(len(train_set) / (len(CLASSES) * counts), dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    EXPRESSION_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best, stale = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val = evaluate(model, val_loader)
        scheduler.step(val["f1_macro"])
        print(f"época {epoch:2} | treino {np.mean(losses):.4f} | val acc {val['accuracy']:.4f} | "
              f"val F1 {val['f1_macro']:.4f}", flush=True)
        if val["f1_macro"] > best:
            best, stale = val["f1_macro"], 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": train_set.classes,
                        "image_size": IMAGE_SIZE, "best_val_f1": best}, EXPRESSION_CHECKPOINT)
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(EXPRESSION_CHECKPOINT, map_location="cpu")["model_state_dict"])
    export_onnx(model, EXPRESSION_MODEL, IMAGE_SIZE, ["output"])


if __name__ == "__main__":
    main()
