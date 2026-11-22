"""
Avaliação da identificação em galeria fechada com data/dados_caras/<pessoa>/<foto>.jpeg.

Protocolo
---------
Divisão por pessoa (as fotos nunca se repetem entre conjuntos):
  - registo:   frontal, frontal2
  - validação: metade das restantes fotos (escolhidas aleatoriamente com SEED fixa)
  - teste:     a outra metade

Dois cenários, ambos avaliados na validação e no teste:
  - Genuínos (galeria completa): todas as pessoas registadas; cada foto deve ser
    reconhecida como a pessoa certa.
  - Impostores (leave-one-out): em cada ronda uma pessoa fica fora da galeria e as
    fotos dela devem dar "Unknown". Há tantas rondas quantas pessoas.

Threshold (critério FAR = 0 na validação):
  threshold = maior score de impostor na validação; aceita-se quando score > threshold.
  É escolhido só com a validação e depois aplicado ao teste.

Métricas
  - rank-1: a pessoa mais próxima é a certa (sem threshold)
  - DIR: genuínos reconhecidos corretamente (pessoa certa e score > threshold)
  - FRR: genuínos rejeitados como "Unknown"
  - misid: genuínos aceites como outra pessoa
  - FAR: impostores aceites como alguém da galeria

Saídas (results/ está no .gitignore):
  results/identity/scores.csv           uma linha por foto e cenário
  results/identity/threshold_sweep.csv  FAR/FRR/DIR para thresholds de 0 a 1
  results/identity/by_condition.csv     resultados de teste por tipo de foto
  results/identity/metrics.json         resumo, com 1 e 2 fotos de registo
  gallery/threshold.json                threshold calibrado, usado pela aplicação
"""
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from src.config import GALLERY_PHOTOS_DIR, RESULTS_DIR, ROOT_DIR
from src.pipeline.detection import load_detector
from src.pipeline.identity import Gallery, FaceEmbedder, embed_image, save_threshold

OUT_DIR = RESULTS_DIR / "identity"
ENROL_STEMS = ("frontal", "frontal2")
SEED = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class Photo:
    person: str
    stem: str
    condition: str             # tipo de foto: frontal, lado, expressao, distancia, oclusao...
    split: str                 # enrol | val | test
    embedding: np.ndarray | None  # None quando não foi detetada nenhuma cara


@dataclass
class ScoreRow:
    scenario: str          # genuine | impostor
    held_out: str          # pessoa fora da galeria (só impostores)
    split: str
    person: str
    image: str
    condition: str
    best_candidate: str
    best_score: float
    true_score: float      # score com o template da própria pessoa (só genuínos)
    rank1_correct: bool


def condition_of(stem: str) -> str:
    return re.sub(r"\d+$", "", stem)


def load_photos(detector, embedder: FaceEmbedder) -> list[Photo]:
    rng = np.random.default_rng(SEED)
    photos = []
    for person_dir in sorted(p for p in GALLERY_PHOTOS_DIR.iterdir() if p.is_dir()):
        paths = sorted(p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        enrol = [p for p in paths if p.stem in ENROL_STEMS]
        rest = [p for p in paths if p.stem not in ENROL_STEMS]
        rest = [rest[i] for i in rng.permutation(len(rest))]
        n_val = len(rest) // 2
        splits = [(p, "enrol") for p in enrol] + \
                 [(p, "val") for p in rest[:n_val]] + [(p, "test") for p in rest[n_val:]]

        for path, split in splits:
            photos.append(Photo(person_dir.name, path.stem, condition_of(path.stem), split,
                                embed_image(path, detector, embedder)))
    return photos


def build_gallery(photos: list[Photo], enrol_stems: tuple[str, ...], exclude: str | None = None) -> Gallery:
    gallery = Gallery(threshold=0.0, path=None)
    for person in sorted({p.person for p in photos} - {exclude}):
        embs = [p.embedding for p in photos
                if p.person == person and p.split == "enrol" and p.stem in enrol_stems
                and p.embedding is not None]
        if not embs:
            raise RuntimeError(f"{person}: nenhuma foto de registo com cara detetada.")
        gallery.enrol(person, embs)
    return gallery


def score_probes(photos: list[Photo], enrol_stems: tuple[str, ...]) -> list[ScoreRow]:
    probes = [p for p in photos if p.split in ("val", "test") and p.embedding is not None]
    people = sorted({p.person for p in photos})
    rows = []

    full = build_gallery(photos, enrol_stems)
    for p in probes:
        scores = full.scores(p.embedding)
        best = max(scores, key=scores.get)
        rows.append(ScoreRow("genuine", "", p.split, p.person, p.stem, p.condition,
                             best, scores[best], scores[p.person], best == p.person))

    for held_out in people:
        gallery = build_gallery(photos, enrol_stems, exclude=held_out)
        for p in probes:
            if p.person != held_out:
                continue
            scores = gallery.scores(p.embedding)
            best = max(scores, key=scores.get)
            rows.append(ScoreRow("impostor", held_out, p.split, p.person, p.stem, p.condition,
                                 best, scores[best], float("nan"), False))
    return rows


def rates(rows: list[ScoreRow], split: str, threshold: float) -> dict:
    genuine = [r for r in rows if r.scenario == "genuine" and r.split == split]
    impostor = [r for r in rows if r.scenario == "impostor" and r.split == split]
    accepted = [r for r in genuine if r.best_score > threshold]
    return {
        "n_genuine": len(genuine),
        "n_impostor": len(impostor),
        "rank1": float(np.mean([r.rank1_correct for r in genuine])) if genuine else float("nan"),
        "dir": sum(r.rank1_correct for r in accepted) / len(genuine) if genuine else float("nan"),
        "frr": 1 - len(accepted) / len(genuine) if genuine else float("nan"),
        "misid": sum(not r.rank1_correct for r in accepted) / len(genuine) if genuine else float("nan"),
        "far": float(np.mean([r.best_score > threshold for r in impostor])) if impostor else float("nan"),
    }


def calibrate_threshold(rows: list[ScoreRow]) -> float:
    """FAR = 0 na validação: o menor threshold que rejeita todos os impostores de validação."""
    impostor_val = [r.best_score for r in rows if r.scenario == "impostor" and r.split == "val"]
    return float(max(impostor_val))


def run_protocol(photos: list[Photo], enrol_stems: tuple[str, ...]) -> tuple[list[ScoreRow], float, dict]:
    rows = score_probes(photos, enrol_stems)
    threshold = calibrate_threshold(rows)
    summary = {
        "enrol_images": list(enrol_stems),
        "threshold": threshold,
        "val": rates(rows, "val", threshold),
        "test": rates(rows, "test", threshold),
    }
    return rows, threshold, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detector, embedder = load_detector(), FaceEmbedder()
    photos = load_photos(detector, embedder)

    missed = [f"{p.person}/{p.stem} ({p.split})" for p in photos if p.embedding is None]
    split_counts = {s: sum(p.split == s for p in photos) for s in ("enrol", "val", "test")}

    rows, threshold, main_summary = run_protocol(photos, ENROL_STEMS)
    _, _, one_image_summary = run_protocol(photos, ENROL_STEMS[:1])

    # Tabelas
    write_csv(OUT_DIR / "scores.csv", [asdict(r) for r in rows])

    sweep = []
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        for split in ("val", "test"):
            sweep.append({"threshold": float(t), "split": split, **rates(rows, split, t)})
    write_csv(OUT_DIR / "threshold_sweep.csv", sweep)

    by_condition = []
    for condition in sorted({r.condition for r in rows if r.split == "test"}):
        cond_rows = [r for r in rows if r.split == "test" and r.condition == condition]
        by_condition.append({"condition": condition, **rates(cond_rows, "test", threshold)})
    write_csv(OUT_DIR / "by_condition.csv", by_condition)

    metrics = {
        "protocol": {
            "people": sorted({p.person for p in photos}),
            "photos_per_split": split_counts,
            "no_face_detected": missed,
            "seed": SEED,
            "threshold_criterion": "FAR = 0 na validação (maior score de impostor); aceita se score > threshold",
        },
        "enrol_2_images": main_summary,
        "enrol_1_image": one_image_summary,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4, ensure_ascii=False))

    save_threshold(threshold, {
        "criterion": "FAR = 0 na validação",
        "enrol_images": list(ENROL_STEMS),
        "source": "src/evaluation/eval_identity.py",
    })

    # Resumo no terminal
    print(f"Fotos: {split_counts} | sem cara detetada: {len(missed)} {missed if missed else ''}")
    for label, summary in (("2 fotos de registo", main_summary), ("1 foto de registo", one_image_summary)):
        print(f"\n== {label} | threshold = {summary['threshold']:.3f}")
        for split in ("val", "test"):
            r = summary[split]
            print(f"  {split:4s} rank-1 {r['rank1']:.2f} | DIR {r['dir']:.2f} | FRR {r['frr']:.2f} | "
                  f"misid {r['misid']:.2f} | FAR {r['far']:.2f} "
                  f"(n genuínos={r['n_genuine']}, n impostores={r['n_impostor']})")
    print(f"\nResultados em {OUT_DIR.relative_to(ROOT_DIR)}/ | threshold guardado em gallery/threshold.json")


if __name__ == "__main__":
    main()
