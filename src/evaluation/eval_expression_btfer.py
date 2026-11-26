"""
Avaliação cross-dataset da expressão no BTFER (7 classes comuns, sem re-treino).

    python -m src.evaluation.eval_expression_btfer                                          # MobileNet
    python -m src.evaluation.eval_expression_btfer --checkpoint models/expression_resnet50.pth

A arquitetura é lida do checkpoint; os resultados da ResNet-50 levam o sufixo _resnet50.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import BTFER_DIR, EXPRESSION_CHECKPOINT, RESULTS_DIR
from src.pipeline.expression import CLASSES
from src.training.expression_models import build_model, checkpoint_arch, suffix

DATASET_DIR = BTFER_DIR
OUTPUT_DIR = RESULTS_DIR / "expression"

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# As 7 classes comuns ao BTFER (sem Contempt), na ordem do modelo, e o índice de cada uma na saída.
COMMON_CLASSES = [c for c in CLASSES if c != "Contempt"]
COMMON_OUTPUTS = [CLASSES.index(c) for c in COMMON_CLASSES]

BTFER_NAMES = {"Angry": "Anger"}  # pasta do BTFER -> nome no modelo (as outras têm o mesmo nome)


def btfer_target(folder: str) -> int:
    """Índice em COMMON_CLASSES da pasta do BTFER."""
    return COMMON_CLASSES.index(BTFER_NAMES.get(folder, folder))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=EXPRESSION_CHECKPOINT)
    args = parser.parse_args()

    print(f"Selected device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(
        root=DATASET_DIR,
        transform=transform
    )

    print("\nBTFER classes:", dataset.classes)
    print("Total images:", len(dataset))

    mapped_samples = [
    (path, btfer_target(dataset.classes[label]))
    for path, label in dataset.samples
]

    dataset.samples = mapped_samples
    dataset.imgs = mapped_samples
    dataset.targets = [label for _, label in mapped_samples]

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=DEVICE,
        weights_only=False
    )

    arch = checkpoint_arch(checkpoint)
    model = build_model(arch, len(CLASSES), pretrained=False)
    print("Architecture:", arch)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("\nLoaded AffectNet checkpoint.")
    print("Original classes:", checkpoint["class_names"])

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)

            outputs = model(images)
            outputs = outputs[:, COMMON_OUTPUTS]  # só as 7 classes comuns
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    print("\nPrediction distribution:")

    for i, name in enumerate(COMMON_CLASSES):
        print(f"{name}: {np.sum(y_pred == i)}")

    metrics = {
    "dataset": "BTFER",
    "total_images": len(dataset),
    "accuracy": accuracy_score(y_true, y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    "precision_macro": precision_score(
        y_true, y_pred, labels=list(range(7)),
        average="macro", zero_division=0
    ),
    "recall_macro": recall_score(
        y_true, y_pred, labels=list(range(7)),
        average="macro", zero_division=0
    ),
    "f1_macro": f1_score(
        y_true, y_pred, labels=list(range(7)),
        average="macro", zero_division=0
    )
    }   


    
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(7)),
        target_names=COMMON_CLASSES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(7))
    )

    print("\nBTFER CROSS-DATASET RESULTS")
    print("-" * 40)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(7)),
        target_names=COMMON_CLASSES,
        zero_division=0
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_DIR / f"expression{suffix(arch)}_btfer_metrics.json",
        "w"
    ) as file:
        json.dump(
            {"metrics": metrics, "classification_report": report},
            file,
            indent=4
        )

    np.savetxt(
        OUTPUT_DIR / f"expression{suffix(arch)}_btfer_confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(COMMON_CLASSES),
        comments=""
    )

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
