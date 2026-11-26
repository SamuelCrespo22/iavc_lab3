"""
Escreve data/multitask/labels.csv, a lista de imagens do modelo multitarefa (-1 = sem label para essa tarefa):

  AffectNet (train/val/test já divididos) -> expressão
  UTKFace (dividido aqui 70/15/15, SEED fixa) -> género e idade

O treino usa as imagens originais redimensionadas para MULTITASK_INPUT_SIZE: os dois datasets já vêm recortados pela cara,
com o enquadramento que o multitask_crop reproduz na aplicação.

    python -m src.training.prepare_multitask
"""
import csv
from collections import Counter

import numpy as np

from src.config import AFFECTNET_DIR, DATA_DIR, MULTITASK_DATA_DIR, UTKFACE_DIR

UTK_TRAIN, UTK_VAL = 0.70, 0.15  # o resto é teste
SEED = 0
NO_LABEL = {"expression": -1, "gender": -1, "age": -1}


def affectnet_rows() -> list[dict]:
    """A pasta 0–7 é a classe (ver CLASSES em expression.py)."""
    return [{"path": p.relative_to(DATA_DIR).as_posix(), "source": "affectnet", "split": split,
             **NO_LABEL, "expression": int(p.parent.name)}
            for split in ("train", "val", "test")
            for p in sorted((AFFECTNET_DIR / split).glob("*/*.jpg"))]


def utkface_rows() -> list[dict]:
    """Nome: [idade]_[género]_[etnia]_[data].jpg.chip.jpg (género 0 = homem); os mal formados são ignorados."""
    paths = [p for p in sorted(UTKFACE_DIR.glob("*.jpg")) if len(p.name.split("_")) == 4]
    order = np.random.default_rng(SEED).permutation(len(paths))
    n_train, n_val = int(UTK_TRAIN * len(paths)), int(UTK_VAL * len(paths))
    rows = []
    for rank, i in enumerate(order):
        age, gender = paths[i].name.split("_")[:2]
        rows.append({"path": paths[i].relative_to(DATA_DIR).as_posix(), "source": "utkface",
                     "split": "train" if rank < n_train else "val" if rank < n_train + n_val else "test",
                     **NO_LABEL, "gender": int(gender), "age": int(age)})
    return rows


def main():
    rows = affectnet_rows() + utkface_rows()
    MULTITASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(MULTITASK_DATA_DIR / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "split", *NO_LABEL])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} imagens em {MULTITASK_DATA_DIR / 'labels.csv'}")
    for (source, split), count in sorted(Counter((r["source"], r["split"]) for r in rows).items()):
        print(f"  {source:9} {split:5} {count}")


if __name__ == "__main__":
    main()
