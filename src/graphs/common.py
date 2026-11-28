"""Estilo comum às figuras do relatório (largura de uma coluna) e gravação em results/images/ (PDF e PNG)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src.config import RESULTS_DIR  # noqa: E402

IMAGES_DIR = RESULTS_DIR / "images"
COLUMN_WIDTH = 3.3  # polegadas: uma coluna do relatório (A4, duas colunas, margens de 2 cm)
COMMA = FuncFormatter(lambda value, _: f"{value:g}".replace(".", ","))  # vírgula decimal, como no relatório

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def figure(height: float = 1.9):
    return plt.subplots(figsize=(COLUMN_WIDTH, height))


def save(fig, name: str) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(IMAGES_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"{IMAGES_DIR / name}.pdf e .png")
