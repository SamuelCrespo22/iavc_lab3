"""
Tabelas de comparação de todas as opções de atributos (expressão, idade e género) e da identidade, a partir dos
resultados já guardados (não corre nenhum modelo). As opções são a pipeline atual ("separate") e cada corrida em
models/multitask/; uma corrida ainda sem eval_attributes aparece como "por avaliar".

    python -m src.evaluation.compare_models     # -> results/comparison/*.csv e comparison.md
"""
import csv
import json

from src.config import MODELS_DIR, RESULTS_DIR
from src.evaluation.eval_attributes import COMMON, OUT_ROOT as ATTRIBUTES_DIR
from src.evaluation.tables import markdown_table, model_size, pct, write_csv
from src.pipeline.expression import CLASSES

OUT_DIR = RESULTS_DIR / "comparison"
ARCH_NAMES = {"mobilenet": "MobileNetV3-S", "resnet18": "ResNet-18", "resnet50": "ResNet-50"}
IDENTITY_COLUMNS = {"rank1": "Rank-1", "dir": "DIR", "frr": "FRR", "misid": "Misid.", "far": "FAR"}
AGE_GROUPS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
NOT_EVALUATED = "—"


def load(path):
    return json.loads(path.read_text())


def options() -> dict[str, dict]:
    """opção -> colunas que a descrevem (modelos, entrada, tamanho)."""
    expression = model_size("expression/expression_best.onnx")
    genderage = model_size("genderage.onnx")
    found = {"separate": {"Opção": "separate", "Modelos": "genderage + MobileNetV3-S expressão (atual)",
                          "Entrada": "96 / 224 px", "Parâmetros (M)": round(expression[0] + genderage[0], 2),
                          "MB": round(expression[1] + genderage[1], 1)}}
    for run_dir in sorted(p for p in (MODELS_DIR / "multitask").iterdir() if (p / "config.json").exists()):
        config = load(run_dir / "config.json")
        frozen = f", congelada até {config['freeze_until']}" if config.get("freeze_until") else ""
        params, mb = model_size(f"multitask/{run_dir.name}/multitask.onnx")
        found[run_dir.name] = {"Opção": run_dir.name,  # as primeiras corridas não guardavam a arquitetura
                               "Modelos": f"Multitarefa {ARCH_NAMES[config.get('arch', 'mobilenet')]}{frozen}",
                               "Entrada": f"{config['input_size']} px", "Parâmetros (M)": round(params, 2),
                               "MB": round(mb, 1)}
    return found


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    described = options()
    results = {name: load(ATTRIBUTES_DIR / name / "metrics.json") if (ATTRIBUTES_DIR / name / "metrics.json").exists()
               else None for name in described}
    evaluated = {name: m for name, m in results.items() if m}
    pending = [name for name, m in results.items() if m is None]

    expression = []
    for name, cols in described.items():
        m = results[name]
        expression.append({**cols, **{f"{dataset} {label}": pct(m[key][metric]) if m else NOT_EVALUATED
                                      for dataset, key in (("AffectNet", "affectnet"), ("BTFER", "btfer"))
                                      for label, metric in (("acc.", "accuracy"), ("acc. equilibrada", "balanced_accuracy"),
                                                            ("F1 macro", "f1_macro"))},
                           "ms/cara": round(m["latency_ms"], 1) if m else NOT_EVALUATED})
    per_class = [{"Opção": name, "Dataset": dataset, **{c: pct(m[key]["recall_per_class"].get(c)) for c in CLASSES}}
                 for name, m in evaluated.items() for dataset, key in (("AffectNet", "affectnet"), ("BTFER", "btfer"))]

    age_gender = []
    for name, cols in described.items():
        u = results[name] and results[name]["utkface"]
        values = ({"Caras": u["n"], "MAE (anos)": round(u["age_mae"], 2), "RMSE": round(u["age_rmse"], 2),
                   "Viés": f"{u['age_bias']:+.2f}", "Faixa etária certa": pct(u["age_group_accuracy"]),
                   "Género acc.": pct(u["gender"]["accuracy"]), "Género F1": pct(u["gender"]["f1_macro"])} if u else
                  dict.fromkeys(("Caras", "MAE (anos)", "RMSE", "Viés", "Faixa etária certa", "Género acc.",
                                 "Género F1"), NOT_EVALUATED))
        age_gender.append({**cols, **values,
                           "ms/cara": round(results[name]["latency_ms"], 1) if u else NOT_EVALUATED})

    def by_group(key, value, fmt):
        rows = []
        for name, m in evaluated.items():
            groups = {g["group"]: g for g in m["utkface"]["by_group"] if g["by"] == key}
            rows.append({"Opção": name, **{f"{g} (n={groups[g]['n']})": fmt(groups[g][value])
                                           for g in (AGE_GROUPS if key == "age_group" else groups)}})
        return rows

    age_by_group = by_group("age_group", "age_mae", lambda v: f"{v:.1f}")
    gender_by_race = by_group("race", "gender_accuracy", pct)

    identity_metrics = load(RESULTS_DIR / "identity" / "metrics.json")
    identity = [{"Registo": ", ".join(r["enrol_images"]), "Threshold": round(r["threshold"], 3),
                 "Conjunto": {"val": "validação", "test": "teste"}[split],
                 "Genuínos": r[split]["n_genuine"], "Impostores": r[split]["n_impostor"],
                 **{IDENTITY_COLUMNS[k]: pct(r[split][k]) for k in IDENTITY_COLUMNS}}
                for r in (identity_metrics["enrol_2_images"], identity_metrics["enrol_1_image"])
                for split in ("val", "test")]
    with open(RESULTS_DIR / "identity" / "by_condition.csv") as f:
        by_condition = [{"Condição": r["condition"], "Genuínos": r["n_genuine"], "Impostores": r["n_impostor"],
                         **{IDENTITY_COLUMNS[k]: pct(r[k]) for k in IDENTITY_COLUMNS}} for r in csv.DictReader(f)]

    tables = {"expression": expression, "expression_per_class": per_class, "age_gender": age_gender,
              "age_mae_by_age_group": age_by_group, "gender_accuracy_by_race": gender_by_race,
              "identity": identity, "identity_by_condition": by_condition}
    for name, rows in tables.items():
        if rows:
            write_csv(OUT_DIR / f"{name}.csv", rows)

    (OUT_DIR / "comparison.md").write_text("\n\n".join([
        "# Comparação de modelos",
        "Gerado por `python -m src.evaluation.compare_models` a partir de `results/attributes/` "
        "(`eval_attributes.py`) e `results/identity/` (`eval_identity.py`). Latência: ms por cara para idade, "
        "género e expressão juntos (crops incluídos), no portátil (M1, CoreML)."
        + (f" Por avaliar: {', '.join(pending)}." if pending else ""),
        "## Expressão",
        f"AffectNet: teste do AffectNetCustom (8 classes, 400 por classe). BTFER: cross-dataset, {len(COMMON)} classes "
        "comuns (sem Contempt), sem re-treino. Imagens redimensionadas para a entrada de cada modelo.",
        markdown_table(expression),
        "### Recall por classe",
        markdown_table(per_class) if per_class else "Nenhuma opção avaliada.",
        "## Idade e género",
        "Teste do UTKFace (15%, seed 0), pelo caminho da aplicação (SCRFD e o crop de cada modelo). Os multitarefa "
        "treinaram nos outros 70% do UTKFace; o genderage nunca o viu (cross-dataset; no UTKFace completo, 23 688 "
        "caras: MAE 9.82, género 85.8%). Viés > 0: sobrestima. Faixa etária certa: mesma década (70+ junta o resto).",
        markdown_table(age_gender),
        "### MAE da idade por faixa etária",
        markdown_table(age_by_group) if age_by_group else "Nenhuma opção avaliada.",
        "### Accuracy do género por etnia (labels do UTKFace)",
        markdown_table(gender_by_race) if gender_by_race else "Nenhuma opção avaliada.",
        "## Identidade",
        "Um só modelo (ArcFace R50, embedding de 512 e similaridade cosseno): compara-se o número de fotos de "
        "registo. Threshold calibrado na validação (FAR = 0) e aplicado ao teste. Impostores por leave-one-out.",
        markdown_table(identity),
        "### Teste por tipo de foto (registo com 2 fotos)",
        markdown_table(by_condition),
    ]) + "\n")
    print(f"Tabelas em {OUT_DIR}: comparison.md e {len([r for r in tables.values() if r])} CSV")
    if pending:
        print(f"Por avaliar (python -m src.evaluation.eval_attributes --model <opção>): {', '.join(pending)}")


if __name__ == "__main__":
    main()
