"""
Idade e género por condição da imagem: o teste do UTKFace de eval_attributes.py, com cada degradação aplicada à
imagem original (200 px; a caixa do SCRFD ocupa-a quase toda) antes da borda e do SCRFD.

  original          sem alteração
  desfocada         desfocagem gaussiana (sigma = 3 px): foco ou movimento
  escura            brilho a 25% e ruído gaussiano (sigma = 5): sala mal iluminada
  baixa resolução   reduzida a 40 px e ampliada de novo: a cara mais pequena que a app analisa (MIN_FACE_SIZE)
  oclusão           tapada a cinzento a partir de 60% da altura: boca e queixo (máscara, mão à frente da boca)

Por condição: caras detetadas, MAE, viés, faixa etária certa, género e, com o limiar da app
(GENDER_UNCERTAIN_BELOW), a percentagem de géneros marcados "Uncertain" e a accuracy dos restantes.

    python -m src.evaluation.eval_conditions --model r50_224_full   # -> results/attributes/<model>/by_condition.csv
    python -m src.evaluation.eval_conditions --model separate
"""
import argparse
import csv

import cv2 as cv
import numpy as np

from src.config import ATTRIBUTES_RESULTS_DIR, GENDER_UNCERTAIN_BELOW, MIN_FACE_SIZE, MULTITASK_DATA_DIR
from src.evaluation.eval_attributes import MultiTask, Separate, age_gender_metrics, predict_utkface
from src.evaluation.tables import write_csv

rng = np.random.default_rng(0)


def darken(img):
    return np.clip(img * 0.25 + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)


def low_resolution(img, size=MIN_FACE_SIZE):
    small = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)
    return cv.resize(small, img.shape[1::-1], interpolation=cv.INTER_LINEAR)


def occlude(img, top=0.6):
    out = img.copy()
    out[int(top * img.shape[0]):] = 128
    return out


CONDITIONS = {"original": None, "desfocada": lambda img: cv.GaussianBlur(img, (0, 0), 3), "escura": darken,
              "baixa resolução": low_resolution, "oclusão": occlude}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help='"separate" ou uma corrida em models/multitask/')
    args = parser.parse_args()

    option = Separate() if args.model == "separate" else MultiTask(args.model)
    with open(MULTITASK_DATA_DIR / "labels.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test" and r["source"] == "utkface"]

    results = []
    for name, degrade in CONDITIONS.items():
        records, _, _ = predict_utkface(option, rows, degrade)
        m = age_gender_metrics(records)
        confident = [r for r in records if r["gender_confidence"] >= GENDER_UNCERTAIN_BELOW]
        results.append({"condition": name, "detected": len(records) / len(rows), "age_mae": m["age_mae"],
                        "age_bias": m["age_bias"], "age_group_accuracy": m["age_group_accuracy"],
                        "gender_accuracy": m["gender"]["accuracy"], "gender_f1": m["gender"]["f1_macro"],
                        "gender_uncertain": 1 - len(confident) / len(records),
                        "gender_accuracy_confident": np.mean([r["gender"] == r["pred_gender"] for r in confident])})
        r = results[-1]
        print(f"{name:16} detetadas {r['detected']:.1%} | MAE {r['age_mae']:.2f} | viés {r['age_bias']:+.2f} | "
              f"género {r['gender_accuracy']:.1%} | incerto {r['gender_uncertain']:.1%} "
              f"(restantes {r['gender_accuracy_confident']:.1%})", flush=True)
    write_csv(ATTRIBUTES_RESULTS_DIR / args.model / "by_condition.csv", results)


if __name__ == "__main__":
    main()
