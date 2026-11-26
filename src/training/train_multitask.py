"""
Treino do modelo multitarefa: MobileNetV3-Small (ImageNet) partilhada pela expressão, género e idade.

    python -m src.training.prepare_multitask                    # uma vez: data/multitask/labels.csv
    python -m src.training.train_multitask [--run nome] [--input-size 160]   # nome por omissão: data e hora
    python -m src.training.train_multitask --run teste --epochs 1 --limit 2000   # teste rápido

Grava em models/multitask/<run>/:
  multitask.pth   melhor época na validação (soma ponderada das losses de cada tarefa)
  multitask.onnx  o mesmo modelo, com uma saída por tarefa (verificado contra o PyTorch)
  config.json     parâmetros do treino e métricas de validação por época
O teste fica para a avaliação, para não influenciar a escolha do modelo.
"""
import argparse
import csv
import json
import random
import time
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import DATA_DIR, MODELS_DIR, MULTITASK_DATA_DIR, MULTITASK_INPUT_SIZE
from src.pipeline.expression import CLASSES
from src.training.multitask_model import GENDERS, TASKS, MultiTaskNet, multitask_loss, task_loss

LOSS_WEIGHTS = {"expression": 1.0, "gender": 1.0, "age": 1.0}
BATCH_SIZE = 64
LR_BACKBONE, LR_HEADS = 1e-4, 1e-3  # o backbone já vem treinado; as cabeças começam aleatórias
PATIENCE = 4
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class FaceDataset(Dataset):
    """Imagens originais (já recortadas pela cara) redimensionadas para size x size."""

    def __init__(self, split: str, train: bool, size: int, limit: int | None = None):
        with open(MULTITASK_DATA_DIR / "labels.csv") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if limit:
            self.rows = random.Random(0).sample(self.rows, min(limit, len(self.rows)))
        # Recorte aleatório (85–100% da imagem): variação de escala e posição como a da caixa do SCRFD na app,
        # sem rotações nem translações que deixariam cantos pretos.
        resize = ([transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
                   transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.1)]
                  if train else [transforms.Resize((size, size))])
        self.transform = transforms.Compose([*resize, transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        image = Image.open(DATA_DIR / r["path"]).convert("RGB")
        return self.transform(image), {t: int(r[t]) for t in TASKS}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Loss e métrica principal de cada tarefa, só com as amostras que têm a label."""
    model.eval()
    outputs, labels = {t: [] for t in TASKS}, {t: [] for t in TASKS}
    for x, y in loader:
        out = model(x.to(DEVICE))
        for t in TASKS:
            outputs[t].append(out[t].cpu())
            labels[t].append(y[t])
    metrics = {"loss": 0.0}
    for t in TASKS:
        out, y = torch.cat(outputs[t]), torch.cat(labels[t])
        out, y = out[y >= 0], y[y >= 0]
        metrics[f"{t}_loss"] = float(task_loss(t, out, y))
        metrics["loss"] += LOSS_WEIGHTS[t] * metrics[f"{t}_loss"]
    for t in ("expression", "gender"):
        out, y = torch.cat(outputs[t]), torch.cat(labels[t])
        pred, y = out[y >= 0].argmax(1).numpy(), y[y >= 0].numpy()
        metrics[f"{t}_f1_macro"] = float(f1_score(y, pred, average="macro"))
    out, y = torch.cat(outputs["age"]).squeeze(1), torch.cat(labels["age"])
    metrics["age_mae"] = float((out[y >= 0] * 10 - y[y >= 0]).abs().mean())
    return metrics


class OnnxOutputs(nn.Module):
    """O ONNX precisa de saídas em lista: uma por tarefa, pela ordem de TASKS."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return tuple(out[t] for t in TASKS)


def export_onnx(model: nn.Module, path, size: int) -> None:
    wrapper = OnnxOutputs(model.cpu().eval())
    torch.onnx.export(wrapper, torch.randn(1, 3, size, size), path, input_names=["input"],
                      output_names=list(TASKS), opset_version=18,
                      dynamic_axes={"input": {0: "batch"}, **{t: {0: "batch"} for t in TASKS}})
    x = torch.randn(4, 3, size, size)
    with torch.no_grad():
        expected = wrapper(x)
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})
    diff = max(float(np.abs(e.numpy() - g).max()) for e, g in zip(expected, got))
    print(f"ONNX exportado: {path.name} (diferença máxima para o PyTorch: {diff:.1e})")
    if diff > 1e-3:
        raise SystemExit("A saída do ONNX não coincide com a do PyTorch.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--input-size", type=int, default=MULTITASK_INPUT_SIZE, help="lado da imagem de entrada")
    parser.add_argument("--limit", type=int, help="só N imagens de treino e de validação (teste rápido)")
    args = parser.parse_args()

    run_dir = MODELS_DIR / "multitask" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    loader = lambda split, train: DataLoader(FaceDataset(split, train, args.input_size, args.limit), BATCH_SIZE,
                                             shuffle=train,
                                             num_workers=4, persistent_workers=True)
    train_loader, val_loader = loader("train", True), loader("val", False)

    model = MultiTaskNet(pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": LR_BACKBONE},
                                   {"params": model.heads.parameters(), "lr": LR_HEADS}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    config = {"input_size": args.input_size, "batch_size": BATCH_SIZE, "lr_backbone": LR_BACKBONE,
              "lr_heads": LR_HEADS, "loss_weights": LOSS_WEIGHTS, "max_epochs": args.epochs, "patience": PATIENCE,
              "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset),
              "classes": CLASSES, "genders": GENDERS, "history": []}
    print(f"{DEVICE} | treino {config['train_images']} | validação {config['val_images']} | {run_dir}")

    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, losses = time.time(), []
        for x, y in train_loader:
            loss = multitask_loss(model(x.to(DEVICE)), {t: v.to(DEVICE) for t, v in y.items()}, LOSS_WEIGHTS)
            if not torch.is_tensor(loss):  # batch sem nenhuma label
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val = evaluate(model, val_loader)
        scheduler.step(val["loss"])
        config["history"].append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                                  **{f"val_{k}": v for k, v in val.items()}})
        print(f"época {epoch:2} | {time.time() - start:4.0f} s | treino {np.mean(losses):.3f} | val {val['loss']:.3f} | "
              f"expressão F1 {val['expression_f1_macro']:.3f} | género F1 {val['gender_f1_macro']:.3f} | "
              f"idade MAE {val['age_mae']:.1f}", flush=True)

        if val["loss"] < best:
            best, stale = val["loss"], 0
            config["best_epoch"] = epoch
            torch.save({"model_state_dict": model.state_dict(), "config": config}, run_dir / "multitask.pth")
        else:
            stale += 1
        (run_dir / "config.json").write_text(json.dumps(config, indent=4))
        if stale >= PATIENCE:
            print("Early stopping.")
            break

    model.load_state_dict(torch.load(run_dir / "multitask.pth", map_location="cpu")["model_state_dict"])
    export_onnx(model, run_dir / "multitask.onnx", args.input_size)
    print(f"Melhor época: {config['best_epoch']} | {run_dir}")


if __name__ == "__main__":
    main()
