"""
Avaliação do modelo multitarefa no conjunto de teste, lado a lado com a pipeline atual.

  Expressão   AffectNet test (8 classes) e BTFER (7 classes comuns): imagens redimensionadas, como no treino
              e como nas avaliações do modelo de expressão atual (results/expression/).
  Idade/género  teste do UTKFace (prepare_multitask.py), pelo caminho da aplicação: SCRFD na imagem com
              borda, depois multitask_crop (multitarefa) e attribute_crop + genderage (atual), na mesma cara.
  Latência    ms por cara com os providers da aplicação (multitarefa vs genderage + expressão atual).

Saídas em results/multitask/<run>/: metrics.json e as matrizes de confusão.

    python -m src.evaluation.eval_multitask [--run mnet112]
"""
import argparse
import csv
import json
import time

import cv2 as cv
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.config import (AGE_GENDER_MODEL, BTFER_DIR, DATA_DIR, MODELS_DIR, MULTITASK_DATA_DIR, MULTITASK_RUN,
                        RESULTS_DIR)
from src.evaluation.eval_age_gender import PADDING, age_group
from src.pipeline.age_gender import AgeGenderEstimator
from src.pipeline.detection import attribute_crop, largest_face, load_detector, multitask_crop
from src.pipeline.expression import CLASSES, ExpressionEstimator
from src.pipeline.multitask import GENDERS, MultiTaskEstimator

BTFER_NAMES = {"Angry": "Anger"}  # pasta do BTFER -> nome em CLASSES
COMMON = [c for c in CLASSES if c != "Contempt"]  # o BTFER não tem Contempt


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


def eval_affectnet(model, rows, out_dir) -> dict:
    y_true, y_pred = [], []
    for r in rows:
        _, expression = model.predict(cv.imread(str(DATA_DIR / r["path"])))
        y_true.append(CLASSES[int(r["expression"])])
        y_pred.append(expression.label)
    save_confusion(out_dir / "affectnet_confusion.csv", y_true, y_pred, CLASSES)
    return classification_metrics(y_true, y_pred, CLASSES)


def eval_btfer(model, out_dir) -> dict:
    """Só as 7 classes comuns: a previsão é a mais provável entre elas (como em eval_expression_btfer.py)."""
    y_true, y_pred = [], []
    for folder in sorted(p for p in BTFER_DIR.iterdir() if p.is_dir()):
        for path in sorted(folder.iterdir()):
            _, expression = model.predict(cv.imread(str(path)))
            y_true.append(BTFER_NAMES.get(folder.name, folder.name))
            y_pred.append(max(COMMON, key=expression.probabilities.get))
    save_confusion(out_dir / "btfer_confusion.csv", y_true, y_pred, COMMON)
    return classification_metrics(y_true, y_pred, COMMON)


def age_gender_metrics(ages, genders, pred_ages, pred_genders) -> dict:
    errors = np.array(pred_ages) - np.array(ages)
    return {"n": len(ages), "age_mae": float(np.abs(errors).mean()), "age_rmse": float(np.sqrt((errors ** 2).mean())),
            "age_bias": float(errors.mean()),
            "age_group_accuracy": float(np.mean([age_group(a) == age_group(p) for a, p in zip(ages, pred_ages)])),
            "gender": classification_metrics(genders, pred_genders, GENDERS)}


def eval_utkface(model, baseline, detector, rows, out_dir) -> dict:
    ages, genders, preds = [], [], {"multitask": ([], []), "baseline": ([], [])}
    no_face = 0
    for r in rows:
        img = cv.imread(str(DATA_DIR / r["path"]))
        pad = int(PADDING * max(img.shape[:2]))
        img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)
        face = largest_face(detector, img)
        if face is None:
            no_face += 1
            continue
        ages.append(int(r["age"]))
        genders.append(GENDERS[int(r["gender"])])
        for name, result in (("multitask", model.predict(multitask_crop(img, face))[0]),
                             ("baseline", baseline.predict(attribute_crop(img, face)))):
            preds[name][0].append(result.age)
            preds[name][1].append(result.gender)
    for name, (_, pred_genders) in preds.items():
        save_confusion(out_dir / f"utkface_gender_confusion_{name}.csv", genders, pred_genders, GENDERS)
    return {"no_face_detected": no_face,
            **{name: age_gender_metrics(ages, genders, *p) for name, p in preds.items()}}


def latency_ms(fn, crop, runs=200) -> float:
    for _ in range(20):
        fn(crop)
    start = time.perf_counter()
    for _ in range(runs):
        fn(crop)
    return (time.perf_counter() - start) / runs * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=MULTITASK_RUN)
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "multitask" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    model = MultiTaskEstimator(MODELS_DIR / "multitask" / args.run / "multitask.onnx")
    baseline = AgeGenderEstimator(str(AGE_GENDER_MODEL), confidence_threshold=0.0)
    model.gender_uncertain_below = 0.0  # os dois sem limiar de incerteza: compara-se a previsão original
    with open(MULTITASK_DATA_DIR / "labels.csv") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]

    metrics = {"run": args.run}
    print("AffectNet test...", flush=True)
    metrics["affectnet"] = eval_affectnet(model, [r for r in test if r["source"] == "affectnet"], out_dir)
    print("BTFER...", flush=True)
    metrics["btfer"] = eval_btfer(model, out_dir)
    print("UTKFace test (multitarefa e genderage na mesma cara)...", flush=True)
    detector = load_detector()
    metrics["utkface"] = eval_utkface(model, baseline, detector, [r for r in test if r["source"] == "utkface"],
                                      out_dir)

    sample = cv.imread(str(DATA_DIR / test[0]["path"]))
    expression = ExpressionEstimator()
    metrics["latency_ms"] = {"multitask": latency_ms(model.predict, sample),
                             "genderage": latency_ms(baseline.predict, cv.resize(sample, (96, 96))),
                             "expression": latency_ms(expression.predict, cv.resize(sample, (224, 224)))}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))

    current = {name: json.loads((RESULTS_DIR / "expression" / f).read_text())
               for name, f in (("affectnet", "expression_test_metrics.json"),
                               ("btfer", "expression_btfer_metrics.json"))}
    current["btfer"] = current["btfer"]["metrics"]
    u, lat = metrics["utkface"], metrics["latency_ms"]
    print(f"\n{'':32} {'multitarefa':>12} {'atual':>12}")
    for name in ("affectnet", "btfer"):
        for key in ("accuracy", "f1_macro"):
            print(f"Expressão {name:9} {key:12} {metrics[name][key]:12.3f} {current[name][key]:12.3f}")
    for key in ("age_mae", "age_rmse", "age_bias", "age_group_accuracy"):
        print(f"Idade UTKFace {key:18} {u['multitask'][key]:12.2f} {u['baseline'][key]:12.2f}")
    for key in ("accuracy", "f1_macro"):
        print(f"Género UTKFace {key:17} {u['multitask']['gender'][key]:12.3f} {u['baseline']['gender'][key]:12.3f}")
    print(f"Latência por cara (ms) {'':8} {lat['multitask']:12.1f} {lat['genderage'] + lat['expression']:12.1f}"
          f"  (genderage {lat['genderage']:.1f} + expressão {lat['expression']:.1f})")
    print(f"\nUTKFace: {u['multitask']['n']} caras avaliadas, {u['no_face_detected']} sem cara | {out_dir}")


if __name__ == "__main__":
    main()
