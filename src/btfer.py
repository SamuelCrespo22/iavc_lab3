
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "BTFER"
CHECKPOINT_PATH = ROOT / "src/models/expression_best.pth"
OUTPUT_DIR = ROOT / "src/models"

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

AFFECTNET_CLASSES = [
    "Neutral",
    "Happy",
    "Sad",
    "Surprise",
    "Fear",
    "Disgust",
    "Anger"
]

BTFER_TO_AFFECTNET = {
    "Neutral": 0,
    "Happy": 1,
    "Sad": 2,
    "Surprise": 3,
    "Fear": 4,
    "Disgust": 5,
    "Angry": 6
}


def main():
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
    (path, BTFER_TO_AFFECTNET[dataset.classes[label]])
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
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = models.mobilenet_v3_small(weights=None)

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        8
    )

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
            outputs = outputs[:, :7]
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    print("\nPrediction distribution:")

    for i, name in enumerate(AFFECTNET_CLASSES + ["Contempt"]):
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
        target_names=AFFECTNET_CLASSES,
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
        target_names=AFFECTNET_CLASSES,
        zero_division=0
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_DIR / "expression_btfer_metrics.json",
        "w"
    ) as file:
        json.dump(
            {"metrics": metrics, "classification_report": report},
            file,
            indent=4
        )

    np.savetxt(
        OUTPUT_DIR / "expression_btfer_confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(AFFECTNET_CLASSES),
        comments=""
    )

    print("\nResults saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
