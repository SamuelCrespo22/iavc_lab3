"""
Avaliação de uma opção de modelos de atributos (expressão, idade e género), sempre pelo ONNX, como na aplicação.

  --model separate   pipeline atual: genderage (96 px) + MobileNetV3 de expressão (224 px)
  --model <corrida>  modelo multitarefa em models/multitask/<corrida>/ (mnet160, r50_224_full, ...)

  Expressão     AffectNet test (8 classes) e BTFER (7 classes comuns, cross-dataset): imagens redimensionadas.
  Idade/género  teste do UTKFace (15%, prepare_multitask.py) pelo caminho da aplicação: SCRFD na imagem com
                borda e o crop de cada modelo. Também por faixa etária, etnia e género. Os multitarefa treinaram
                nos outros 70% do UTKFace; o genderage nunca o viu.
  Latência      ms por cara para idade, género e expressão (crops incluídos), com os providers da aplicação.

Saídas em results/attributes/<model>/: metrics.json, by_group.csv e as matrizes de confusão.
A tabela com todas as opções: python -m src.evaluation.compare_models

    python -m src.evaluation.eval_attributes --model separate
    python -m src.evaluation.eval_attributes --model mnet160
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.config import AGE_GENDER_MODEL, ATTRIBUTES_RESULTS_DIR, BTFER_DIR, DATA_DIR, MODELS_DIR, MULTITASK_DATA_DIR
from src.evaluation.tables import write_csv
from src.pipeline.age_gender import AgeGenderEstimator
from src.pipeline.detection import attribute_crop, expression_crop, largest_face, load_detector, multitask_crop
from src.pipeline.expression import BTFER_CLASSES, CLASSES, ExpressionEstimator
from src.pipeline.multitask import GENDERS, MultiTaskEstimator

BTFER_NAMES = {"Angry": "Anger"}                  # pasta do BTFER -> nome em CLASSES
RACES = ["White", "Black", "Asian", "Indian", "Others"]  # 3.º campo do nome do ficheiro do UTKFace
PADDING = 0.5  # o UTKFace vem recortado à cara: o SCRFD precisa de borda (com 0.25 falha ~30% das caras)


class Separate:
    """Pipeline atual: genderage + MobileNetV3 de expressão, cada um com o seu crop."""

    def __init__(self):
        self.age_gender = AgeGenderEstimator(str(AGE_GENDER_MODEL), confidence_threshold=0.0)
        self.expression = ExpressionEstimator()

    def expression_of(self, image):
        return self.expression.predict(image)

    def analyse(self, frame, face):
        return (self.age_gender.predict(attribute_crop(frame, face)),
                self.expression.predict(expression_crop(frame, face)))


class MultiTask:
    """Uma só rede para idade, género e expressão, com o multitask_crop ao tamanho do modelo."""

    def __init__(self, run: str):
        self.model = MultiTaskEstimator(MODELS_DIR / "multitask" / run / "multitask.onnx")
        self.model.gender_uncertain_below = 0.0  # como no genderage: compara-se a previsão original

    def expression_of(self, image):
        return self.model.predict(image)[1]

    def analyse(self, frame, face):
        return self.model.predict(multitask_crop(frame, face, self.model.size))


def classification_metrics(y_true, y_pred, labels) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision_macro": p, "recall_macro": r, "f1_macro": f1,
            "recall_per_class": {c: report[c]["recall"] for c in labels}}


def save_confusion(path, y_true, y_pred, labels):
    np.savetxt(path, confusion_matrix(y_true, y_pred, labels=labels), delimiter=",", fmt="%d",
               header="linhas = verdadeiro, colunas = previsto: " + ",".join(labels))


def eval_expression(option, samples, labels, path) -> dict:
    """samples: (imagem, classe verdadeira). A previsão é a classe mais provável entre labels."""
    y_true, y_pred = [], []
    for image_path, label in samples:
        probabilities = option.expression_of(cv.imread(str(image_path))).probabilities
        y_true.append(label)
        y_pred.append(max(labels, key=probabilities.get))
    save_confusion(path, y_true, y_pred, labels)
    return classification_metrics(y_true, y_pred, labels)


def age_group(age: int) -> str:
    return "70+" if age >= 70 else f"{age // 10 * 10}-{age // 10 * 10 + 9}"


def age_gender_metrics(records: list[dict]) -> dict:
    errors = np.array([r["pred_age"] - r["age"] for r in records])
    return {"n": len(records), "age_mae": float(np.abs(errors).mean()), "age_rmse": float(np.sqrt((errors ** 2).mean())),
            "age_bias": float(errors.mean()),  # > 0: sobrestima a idade
            "age_group_accuracy": float(np.mean([age_group(r["age"]) == age_group(r["pred_age"]) for r in records])),
            "gender": classification_metrics([r["gender"] for r in records], [r["pred_gender"] for r in records],
                                             GENDERS)}


def eval_utkface(option, rows, out_dir) -> tuple[dict, tuple]:
    """Idade e género no teste do UTKFace. Devolve as métricas e uma (imagem, cara) para medir a latência."""
    detector, records, no_face, sample = load_detector(), [], 0, None
    for r in rows:
        img = cv.imread(str(DATA_DIR / r["path"]))
        pad = int(PADDING * max(img.shape[:2]))
        img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)
        face = largest_face(detector, img)
        if face is None:
            no_face += 1
            continue
        sample = sample or (img, face)
        age_gender, _ = option.analyse(img, face)
        records.append({"age": int(r["age"]), "gender": GENDERS[int(r["gender"])],
                        "race": RACES[int(Path(r["path"]).name.split("_")[2])],
                        "pred_age": age_gender.age, "pred_gender": age_gender.gender})

    groups = []
    for key, group_of in (("age_group", lambda r: age_group(r["age"])), ("race", lambda r: r["race"]),
                          ("gender", lambda r: r["gender"])):
        for group in sorted({group_of(r) for r in records}):
            m = age_gender_metrics([r for r in records if group_of(r) == group])
            groups.append({"by": key, "group": group, "n": m["n"], "age_mae": m["age_mae"],
                           "age_bias": m["age_bias"], "gender_accuracy": m["gender"]["accuracy"]})
    write_csv(out_dir / "by_group.csv", groups)
    save_confusion(out_dir / "utkface_gender_confusion.csv", [r["gender"] for r in records],
                   [r["pred_gender"] for r in records], GENDERS)
    return {"no_face_detected": no_face, **age_gender_metrics(records), "by_group": groups}, sample


def latency_ms(fn, runs: int = 200) -> float:
    for _ in range(20):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help='"separate" ou uma corrida em models/multitask/')
    args = parser.parse_args()

    option = Separate() if args.model == "separate" else MultiTask(args.model)
    out_dir = ATTRIBUTES_RESULTS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(MULTITASK_DATA_DIR / "labels.csv") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]

    print("AffectNet test...", flush=True)
    affectnet = [(DATA_DIR / r["path"], CLASSES[int(r["expression"])]) for r in test if r["source"] == "affectnet"]
    metrics = {"model": args.model,
               "affectnet": eval_expression(option, affectnet, CLASSES, out_dir / "affectnet_confusion.csv")}
    print("BTFER...", flush=True)
    btfer = [(p, BTFER_NAMES.get(folder.name, folder.name))
             for folder in sorted(d for d in BTFER_DIR.iterdir() if d.is_dir()) for p in sorted(folder.iterdir())]
    metrics["btfer"] = eval_expression(option, btfer, BTFER_CLASSES, out_dir / "btfer_confusion.csv")
    print("UTKFace test...", flush=True)
    metrics["utkface"], (img, face) = eval_utkface(option, [r for r in test if r["source"] == "utkface"], out_dir)
    metrics["latency_ms"] = latency_ms(lambda: option.analyse(img, face))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))

    a, b, u = metrics["affectnet"], metrics["btfer"], metrics["utkface"]
    print(f"\n{args.model}\n"
          f"  Expressão  AffectNet acc {a['accuracy']:.1%} F1 {a['f1_macro']:.3f} | "
          f"BTFER acc {b['accuracy']:.1%} F1 {b['f1_macro']:.3f}\n"
          f"  Idade      MAE {u['age_mae']:.2f} | viés {u['age_bias']:+.2f} | faixa certa {u['age_group_accuracy']:.1%}\n"
          f"  Género     acc {u['gender']['accuracy']:.1%} | {u['n']} caras, {u['no_face_detected']} sem cara\n"
          f"  Latência   {metrics['latency_ms']:.1f} ms por cara\n  {out_dir}")


if __name__ == "__main__":
    main()
