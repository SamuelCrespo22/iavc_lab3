"""
Avaliação de idade e género (genderage.onnx do InsightFace) no UTKFace.

Usa o mesmo caminho que a aplicação: SCRFD -> attribute_crop (96x96) -> genderage.
As imagens do UTKFace são recortes justos, por isso leva uma borda antes da deteção.
O genderage não foi treinado no UTKFace: é uma avaliação cross-dataset.

Labels no nome do ficheiro: [idade]_[género]_[etnia]_[data].jpg (género 0 = homem, 1 = mulher).

Saídas (results/age_gender/):
  predictions.csv       uma linha por imagem
  metrics.json          idade (MAE, RMSE, faixa etária) e género (P/R/F1, cobertura)
  gender_confusion.csv  matriz de confusão do género
  by_group.csv          resultados por faixa etária, etnia e género

Uso: python src/eval_age_gender.py [--limit N]
"""
import argparse
import csv
import json

import cv2 as cv
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from age_gender import AgeGenderEstimator
from detection import MODELS_DIR, ROOT_DIR, attribute_crop, largest_face, load_detector

UTK_DIR = ROOT_DIR / "data" / "UTKFace"
RESULTS_DIR = ROOT_DIR / "results" / "age_gender"
GENDERS = ["Male", "Female"]
RACES = ["White", "Black", "Asian", "Indian", "Others"]
PADDING = 0.5  # borda à volta da imagem, em fração do lado (com 0.25 o SCRFD falha ~30% das caras)
SEED = 0


def parse_labels(filename: str) -> tuple[int, str, str] | None:
    parts = filename.split("_")
    try:
        return int(parts[0]), GENDERS[int(parts[1])], RACES[int(parts[2])]
    except (ValueError, IndexError):
        return None


def age_group(age: int) -> str:
    return "70+" if age >= 70 else f"{age // 10 * 10}-{age // 10 * 10 + 9}"


def predict_all(paths, detector, estimator: AgeGenderEstimator, uncertain_below: float):
    rows, bad_name, no_face = [], [], []
    for i, path in enumerate(paths, 1):
        labels = parse_labels(path.name)
        if labels is None:
            bad_name.append(path.name)
            continue
        img = cv.imread(str(path))
        pad = int(PADDING * max(img.shape[:2]))
        img = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT, value=0)
        face = largest_face(detector, img)
        if face is None:
            no_face.append(path.name)
            continue

        pred = estimator.predict(attribute_crop(img, face))
        age, gender, race = labels
        rows.append({
            "image": path.name, "age": age, "gender": gender, "race": race,
            "pred_age": pred.age, "pred_gender": pred.gender,
            "gender_confidence": pred.gender_confidence,
            "uncertain": pred.gender_confidence < uncertain_below,
        })
        if i % 2000 == 0:
            print(f"  {i}/{len(paths)}")
    return rows, bad_name, no_face


def gender_metrics(rows: list[dict]) -> dict:
    y_true = [r["gender"] for r in rows]
    y_pred = [r["pred_gender"] for r in rows]
    return classification_report(y_true, y_pred, labels=GENDERS, output_dict=True, zero_division=0)


def group_rows(rows: list[dict]) -> list[dict]:
    out = []
    for key, fn in (("age_group", lambda r: age_group(r["age"])),
                    ("race", lambda r: r["race"]),
                    ("gender", lambda r: r["gender"])):
        for group in sorted({fn(r) for r in rows}):
            sub = [r for r in rows if fn(r) == group]
            errors = np.array([r["pred_age"] - r["age"] for r in sub])
            both_classes = len({r["gender"] for r in sub}) == 2  # F1 macro não faz sentido com uma só classe
            out.append({
                "by": key, "group": group, "n": len(sub),
                "age_mae": float(np.mean(np.abs(errors))),
                "age_bias": float(np.mean(errors)),  # > 0: sobrestima a idade
                "gender_accuracy": float(np.mean([r["gender"] == r["pred_gender"] for r in sub])),
                "gender_f1_macro": float(f1_score([r["gender"] for r in sub], [r["pred_gender"] for r in sub],
                                                  labels=GENDERS, average="macro", zero_division=0))
                                   if both_classes else "",
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="avaliar só N imagens aleatórias (teste rápido)")
    args = parser.parse_args()

    paths = sorted(UTK_DIR.glob("*.jpg"))
    if args.limit:
        paths = [paths[i] for i in np.random.default_rng(SEED).choice(len(paths), args.limit, replace=False)]

    detector = load_detector()
    estimator = AgeGenderEstimator(str(MODELS_DIR / "genderage.onnx"))
    # Guardar o limiar de incerteza e desligá-lo no predict(), para manter a previsão original.
    uncertain_below, estimator.confidence_threshold = estimator.confidence_threshold, 0.0

    print(f"A avaliar {len(paths)} imagens...")
    rows, bad_name, no_face = predict_all(paths, detector, estimator, uncertain_below)

    errors = np.array([r["pred_age"] - r["age"] for r in rows])
    confident = [r for r in rows if not r["uncertain"]]
    metrics = {
        "dataset": "UTKFace", "images": len(paths), "evaluated": len(rows),
        "bad_filename": bad_name, "no_face_detected": len(no_face),
        "age": {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "bias": float(np.mean(errors)),
            "age_group_accuracy": float(np.mean([age_group(r["age"]) == age_group(r["pred_age"]) for r in rows])),
        },
        "gender": {
            "all": gender_metrics(rows),
            "uncertain_threshold": uncertain_below,
            "coverage": len(confident) / len(rows),
            "confident_only": gender_metrics(confident),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4))
    for name, table in (("predictions.csv", rows), ("by_group.csv", group_rows(rows))):
        with (RESULTS_DIR / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    matrix = confusion_matrix([r["gender"] for r in rows], [r["pred_gender"] for r in rows], labels=GENDERS)
    np.savetxt(RESULTS_DIR / "gender_confusion.csv", matrix, delimiter=",", fmt="%d",
               header="linhas = verdadeiro, colunas = previsto: " + ",".join(GENDERS))

    a, g = metrics["age"], metrics["gender"]
    print(f"\nAvaliadas {len(rows)}/{len(paths)} | sem cara: {len(no_face)} | nome inválido: {len(bad_name)}")
    print(f"Idade:  MAE {a['mae']:.2f} | RMSE {a['rmse']:.2f} | viés {a['bias']:+.2f} | "
          f"faixa etária certa {a['age_group_accuracy']:.2%}")
    print(f"Género: accuracy {g['all']['accuracy']:.2%} | F1 macro {g['all']['macro avg']['f1-score']:.3f} | "
          f"cobertura {g['coverage']:.1%} | accuracy nas confiantes {g['confident_only']['accuracy']:.2%}")
    print(f"Resultados em {RESULTS_DIR.relative_to(ROOT_DIR)}/")


if __name__ == "__main__":
    main()
