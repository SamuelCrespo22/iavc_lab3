"""
Tabelas de comparação de todas as opções de atributos (expressão, idade e género) e da identidade, a partir dos
resultados já guardados (não corre nenhum modelo). As opções são a pipeline atual ("separate") e cada corrida
terminada em models/multitask/ (com multitask.onnx); sem eval_attributes ainda, aparece como "por avaliar".

    python -m src.evaluation.compare_models     # -> results/comparison/*.csv e comparison.md
"""
import csv
import json

import numpy as np

from src.config import ATTRIBUTES_RESULTS_DIR, MODELS_DIR, RESULTS_DIR
from src.evaluation.tables import markdown_table, model_size, pct, write_csv
from src.pipeline.expression import BTFER_CLASSES, CLASSES

OUT_DIR = RESULTS_DIR / "comparison"
ARCH_NAMES = {"mobilenet": "MobileNetV3-S", "resnet18": "ResNet-18", "resnet50": "ResNet-50"}
IDENTITY_COLUMNS = {"rank1": "Rank-1", "dir": "DIR", "frr": "FRR", "misid": "Misid.", "far": "FAR"}
AGE_GROUPS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
DATASETS = (("AffectNet", "affectnet"), ("BTFER", "btfer"))


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def read_csv(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def per_class(confusion_path) -> dict[str, tuple]:
    """classe -> (precisão, recall, F1), de uma matriz guardada por eval_attributes (linhas = verdadeiro)."""
    labels = confusion_path.read_text().splitlines()[0].split(": ")[1].split(",")
    m = np.loadtxt(confusion_path, delimiter=",")
    precision, recall = m.diagonal() / m.sum(0), m.diagonal() / m.sum(1)
    return {c: (p, r, 2 * p * r / (p + r)) for c, p, r in zip(labels, precision, recall)}


def options() -> dict[str, dict]:
    """opção -> colunas que a descrevem. Uma corrida só conta quando já tem o ONNX (treino terminado)."""
    expression, genderage = model_size("expression/expression_best.onnx"), model_size("genderage.onnx")
    found = {"separate": {"Opção": "separate", "Modelos": "genderage + MobileNetV3-S expressão (atual)",
                          "Entrada": "96 / 224 px", "Parâmetros (M)": round(expression[0] + genderage[0], 2),
                          "MB": round(expression[1] + genderage[1], 1)}}
    for run in sorted(p for p in (MODELS_DIR / "multitask").iterdir() if (p / "multitask.onnx").exists()):
        config = load(run / "config.json")  # as primeiras corridas não guardavam a arquitetura
        params, mb = model_size(f"multitask/{run.name}/multitask.onnx")
        found[run.name] = {"Opção": run.name,
                           "Modelos": f"Multitarefa {ARCH_NAMES[config.get('arch', 'mobilenet')]}",
                           "Entrada": f"{config['input_size']} px", "Parâmetros (M)": round(params, 2),
                           "MB": round(mb, 1)}
    return found


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    described = options()
    results = {name: load(ATTRIBUTES_RESULTS_DIR / name / "metrics.json") for name in described}
    evaluated = {name: m for name, m in results.items() if m}
    pending = [name for name in described if name not in evaluated]

    expression = [{**cols, **{f"{label} {k}": pct(m[key][metric]) if m else "—" for label, key in DATASETS
                              for k, metric in (("acc.", "accuracy"), ("F1 macro", "f1_macro"))},
                   "ms/cara": round(m["latency_ms"], 1) if m else "—"}
                  for name, cols in described.items() for m in [results[name]]]
    per_class_recall = [{"Opção": name, "Dataset": label,
                         **{c: pct(m[key]["recall_per_class"].get(c)) for c in CLASSES}}
                        for name, m in evaluated.items() for label, key in DATASETS]
    age_gender = [{"Opção": name, "Caras": u["n"], "MAE (anos)": round(u["age_mae"], 2), "RMSE": round(u["age_rmse"], 2),
                   "Viés": f"{u['age_bias']:+.2f}", "Faixa etária certa": pct(u["age_group_accuracy"]),
                   "Género acc.": pct(u["gender"]["accuracy"]), "Género F1": pct(u["gender"]["f1_macro"]),
                   "ms/cara": round(m["latency_ms"], 1)}
                  for name, m in evaluated.items() for u in [m["utkface"]]]
    gender = [{"Opção": name, **{f"{c} {k}": pct(v) for c, values in per_class(
                   ATTRIBUTES_RESULTS_DIR / name / "utkface_gender_confusion.csv").items()
                                 for k, v in zip(("precisão", "recall", "F1"), values)}}
              for name in evaluated]
    conditions = [{"Opção": name, "Condição": r["condition"], "Detetadas": pct(r["detected"]),
                   "MAE (anos)": round(float(r["age_mae"]), 2), "Viés": f"{float(r['age_bias']):+.2f}",
                   "Faixa etária certa": pct(r["age_group_accuracy"]), "Género acc.": pct(r["gender_accuracy"]),
                   "Género incerto": pct(r["gender_uncertain"]),
                   "Género acc. (sem incertos)": pct(r["gender_accuracy_confident"])}
                  for name in evaluated for r in read_csv(ATTRIBUTES_RESULTS_DIR / name / "by_condition.csv")]

    def by_group(key, value, fmt):
        rows = []
        for name, m in evaluated.items():
            groups = {g["group"]: g for g in m["utkface"]["by_group"] if g["by"] == key}
            rows.append({"Opção": name, **{f"{g} (n={groups[g]['n']})": fmt(groups[g][value])
                                           for g in (AGE_GROUPS if key == "age_group" else groups)}})
        return rows

    identity_metrics = load(RESULTS_DIR / "identity" / "metrics.json")
    identity = [{"Registo": ", ".join(r["enrol_images"]), "Threshold": round(r["threshold"], 3),
                 "Conjunto": {"val": "validação", "test": "teste"}[split], "Genuínos": r[split]["n_genuine"],
                 "Impostores": r[split]["n_impostor"], **{v: pct(r[split][k]) for k, v in IDENTITY_COLUMNS.items()}}
                for r in (identity_metrics["enrol_2_images"], identity_metrics["enrol_1_image"])
                for split in ("val", "test")]
    by_condition = [{"Condição": r["condition"], "Genuínos": r["n_genuine"], "Impostores": r["n_impostor"],
                     **{v: pct(r[k]) for k, v in IDENTITY_COLUMNS.items()}}
                    for r in read_csv(RESULTS_DIR / "identity" / "by_condition.csv")]

    tables = {"expression": expression, "expression_per_class": per_class_recall, "age_gender": age_gender,
              "gender_per_class": gender, "age_gender_by_condition": conditions,
              "age_mae_by_age_group": by_group("age_group", "age_mae", lambda v: f"{v:.1f}"),
              "gender_accuracy_by_race": by_group("race", "gender_accuracy", pct),
              "identity": identity, "identity_by_condition": by_condition}
    for name, rows in tables.items():
        if rows:
            write_csv(OUT_DIR / f"{name}.csv", rows)
    table = lambda name: markdown_table(tables[name]) if tables[name] else "Nenhuma opção avaliada."

    (OUT_DIR / "comparison.md").write_text("\n\n".join([
        "# Comparação de modelos",
        "Gerado por `python -m src.evaluation.compare_models` a partir de `results/attributes/` "
        "(`eval_attributes.py`) e `results/identity/` (`eval_identity.py`). Latência: ms por cara para idade, "
        "género e expressão juntos (crops incluídos), no portátil (M1, CoreML)."
        + (f" Por avaliar: {', '.join(pending)}." if pending else ""),
        "## Expressão",
        f"AffectNet: teste do AffectNetCustom (8 classes, 400 por classe). BTFER: cross-dataset, {len(BTFER_CLASSES)} "
        "classes comuns (sem Contempt), sem re-treino. Os dois conjuntos são equilibrados, por isso a accuracy é "
        "igual à accuracy equilibrada (em `metrics.json`).",
        table("expression"), "### Recall por classe", table("expression_per_class"),
        "## Idade e género",
        "Teste do UTKFace (15%, seed 0), pelo caminho da aplicação (SCRFD e o crop de cada modelo). Os multitarefa "
        "treinaram nos outros 70% do UTKFace; o genderage nunca o viu (cross-dataset; no UTKFace completo, 23 688 "
        "caras: MAE 9.82, género 85.8%). Viés > 0: sobrestima. Faixa etária certa: mesma década (70+ junta o resto).",
        table("age_gender"), "### Género por classe (labels do UTKFace)", table("gender_per_class"),
        "### Por condição da imagem",
        "O mesmo teste com a imagem degradada (`eval_conditions.py`): desfocada (sigma 3 px), escura (brilho a 25% "
        "e ruído), baixa resolução (cara reduzida a 40 px, o mínimo da app) e oclusão (boca e queixo tapados). "
        "Género incerto: confiança abaixo do limiar da app (0.65), mostrado como \"Uncertain\".",
        table("age_gender_by_condition"), "### MAE da idade por faixa etária", table("age_mae_by_age_group"),
        "### Accuracy do género por etnia (labels do UTKFace)", table("gender_accuracy_by_race"),
        "## Identidade",
        "Um só modelo (ArcFace R50, embedding de 512 e similaridade cosseno): compara-se o número de fotos de "
        "registo. Threshold calibrado na validação (FAR = 0) e aplicado ao teste. Impostores por leave-one-out.",
        table("identity"), "### Teste por tipo de foto (registo com 2 fotos)", table("identity_by_condition"),
    ]) + "\n")
    print(f"Tabelas em {OUT_DIR}: comparison.md e {sum(1 for r in tables.values() if r)} CSV")
    if pending:
        print(f"Por avaliar (python -m src.evaluation.eval_attributes --model <opção>): {', '.join(pending)}")


if __name__ == "__main__":
    main()
