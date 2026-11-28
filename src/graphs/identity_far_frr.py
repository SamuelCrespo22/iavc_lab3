"""
Figura do relatório: FAR e FRR no teste da identidade em função do limiar, com o limiar calibrado na validação
(registo com 2 fotos). Lê results/identity/threshold_sweep.csv e metrics.json (eval_identity.py).

    python -m src.graphs.identity_far_frr     # -> results/images/identity_far_frr.{pdf,png}
"""
import json

from src.config import RESULTS_DIR
from src.evaluation.compare_models import read_csv
from src.graphs.common import COMMA, figure, save


def main():
    rows = [r for r in read_csv(RESULTS_DIR / "identity" / "threshold_sweep.csv") if r["split"] == "test"]
    threshold = json.loads((RESULTS_DIR / "identity" / "metrics.json").read_text())["enrol_2_images"]["threshold"]
    thresholds = [float(r["threshold"]) for r in rows]

    fig, ax = figure()
    for key, label, color in (("far", "FAR", "0.6"), ("frr", "FRR", "0.1")):
        ax.step(thresholds, [100 * float(r[key]) for r in rows], where="post", label=label, color=color, lw=1.2)
    ax.axvline(threshold, ls="--", lw=0.8, color="0.3")
    ax.annotate(f"limiar {threshold:.3f}".replace(".", ","), (threshold, 55), xytext=(4, 0),
                textcoords="offset points", fontsize=7)
    ax.set_xlabel("Limiar")
    ax.set_ylabel("Taxa (%)")
    ax.xaxis.set_major_formatter(COMMA)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", ncols=2)
    save(fig, "identity_far_frr")


if __name__ == "__main__":
    main()
