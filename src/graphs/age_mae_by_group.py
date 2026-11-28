"""
Figura do relatório: MAE da idade por faixa etária no teste do UTKFace, modelos separados contra o multitarefa da
aplicação (MULTITASK_RUN). Lê results/attributes/<opção>/metrics.json (eval_attributes.py).

    python -m src.graphs.age_mae_by_group     # -> results/images/age_mae_by_group.{pdf,png}
"""
import json

import numpy as np

from src.config import ATTRIBUTES_RESULTS_DIR, MODELS_DIR, MULTITASK_RUN
from src.evaluation.compare_models import AGE_GROUPS, ARCH_NAMES
from src.graphs.common import COMMA, figure, save


def mae_by_group(option: str) -> list[float]:
    groups = json.loads((ATTRIBUTES_RESULTS_DIR / option / "metrics.json").read_text())["utkface"]["by_group"]
    mae = {g["group"]: g["age_mae"] for g in groups if g["by"] == "age_group"}
    return [mae[g] for g in AGE_GROUPS]


def main():
    arch = json.loads((MODELS_DIR / "multitask" / MULTITASK_RUN / "config.json").read_text())["arch"]
    options = {"separate": ("Separados (genderage)", "0.7"),
               MULTITASK_RUN: (f"{ARCH_NAMES[arch]} multitarefa", "0.15")}
    fig, ax = figure()
    x, width = np.arange(len(AGE_GROUPS)), 0.38
    for i, (option, (label, color)) in enumerate(options.items()):
        ax.bar(x + (i - 0.5) * width, mae_by_group(option), width, label=label, color=color)
    ax.set_xticks(x, [g.replace("-", "–") for g in AGE_GROUPS], fontsize=6.5)
    ax.set_xlabel("Idade real (anos)")
    ax.set_ylabel("MAE (anos)")
    ax.yaxis.set_major_formatter(COMMA)
    ax.legend(ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "age_mae_by_group")


if __name__ == "__main__":
    main()
