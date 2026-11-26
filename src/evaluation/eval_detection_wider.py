"""
Avaliação do detetor no WIDER Face (validação, 3 226 imagens).

Protocolo
- O detetor corre em todas as imagens com limiar baixo (0.02), como no protocolo oficial,
  para obter a curva precisão-recall completa.
- AP easy/medium/hard: versão em Python do avaliador oficial (eval_tools/ground_truth/*.mat),
  com IoU >= 0.5. As caras fora de cada nível de dificuldade são ignoradas nesse nível.
- Ponto de funcionamento da aplicação (DETECTION_THRESHOLD = 0.4): precisão e recall.
- Robustez: recall a 0.4 por tamanho da cara, oclusão, pose, blur, iluminação e número de
  caras na imagem. As caras marcadas como inválidas no WIDER ficam de fora.
- Tempo: ms por chamada ao detetor (inclui redimensionar e NMS, não inclui ler o ficheiro).

Nota: a biblioteca scrfd redimensiona com interpolação nearest (o SCRFD original usa
bilinear), o que pode baixar um pouco o AP nas caras pequenas.

Uso:
  python -m src.evaluation.eval_detection_wider --detector scrfd.onnx --examples 6
  python -m src.evaluation.eval_detection_wider --detector det_10g.onnx
  --reuse: recalcula as métricas a partir das previsões guardadas, sem voltar a correr o detetor
  --compare: só refaz as tabelas de comparação a partir dos resultados já guardados

Saídas: results/detection/<detetor>/  metrics.json, robustness.csv, predictions.npz, examples/
        results/detection/  comparison.csv, robustness_comparison.csv e comparison.md (todos os detetores)
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.io import loadmat

from src.config import DETECTION_THRESHOLD, DETECTOR_MODEL, RESULTS_DIR, WIDER_DIR
from src.evaluation.tables import markdown_table, model_size, pct, write_csv
from src.pipeline.detection import detect, load_detector

IMAGES_DIR = WIDER_DIR / "WIDER_val" / "images"
GT_DIR = WIDER_DIR / "eval_tools" / "ground_truth"
OUT_ROOT = RESULTS_DIR / "detection"
EVAL_THRESHOLD = 0.02
IOU = 0.5
THRESH_NUM = 1000
SETTINGS = ("easy", "medium", "hard")
ATTRIBUTES = {"blur": ["nenhum", "normal", "forte"], "illumination": ["normal", "extrema"],
              "occlusion": ["nenhuma", "parcial", "forte"], "pose": ["tipica", "atipica"]}
# intervalos [início, fim): tamanho = raiz da área da caixa, em píxeis da imagem original
SIZE_BINS = [(0, 16, "<16"), (16, 32, "16-32"), (32, 64, "32-64"), (64, 128, "64-128"), (128, np.inf, ">=128")]
CROWD_BINS = [(1, 2, "1"), (2, 6, "2-5"), (6, 21, "6-20"), (21, np.inf, ">20")]
GROUP_ORDER = {"size": [b[2] for b in SIZE_BINS], "faces_in_image": [b[2] for b in CROWD_BINS], **ATTRIBUTES}


# ---------- ground truth ----------
def load_ground_truth() -> dict[str, dict]:
    """event/ficheiro -> caixas (x, y, w, h), atributos e índices de cada nível de dificuldade."""
    mat = loadmat(GT_DIR / "wider_face_val.mat")
    levels = {s: loadmat(GT_DIR / f"wider_{s}_val.mat")["gt_list"] for s in SETTINGS}
    gt = {}
    for i, event in enumerate(mat["event_list"][:, 0]):
        for j, name in enumerate(mat["file_list"][i][0][:, 0]):
            gt[f"{event[0]}/{name[0]}"] = {
                "boxes": mat["face_bbx_list"][i][0][j][0].astype(float).reshape(-1, 4),
                "invalid": mat["invalid_label_list"][i][0][j][0].ravel(),
                **{a: mat[f"{a}_label_list"][i][0][j][0].ravel() for a in ATTRIBUTES},
                **{s: levels[s][i][0][j][0].ravel() - 1 for s in SETTINGS},  # índices a partir de 0
            }
    return gt


# ---------- previsões ----------
def run_detector(model: str, keys: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    detector = load_detector(model)
    preds, times = {}, []
    for i, key in enumerate(keys, 1):
        img = cv.imread(str(IMAGES_DIR / f"{key}.jpg"))
        start = time.perf_counter()
        faces = detect(detector, img, EVAL_THRESHOLD)
        times.append((time.perf_counter() - start) * 1000)
        preds[key] = np.array([[f.bbox.upper_left.x, f.bbox.upper_left.y,
                                f.bbox.lower_right.x - f.bbox.upper_left.x,
                                f.bbox.lower_right.y - f.bbox.upper_left.y, f.probability]
                               for f in faces], dtype=np.float64).reshape(-1, 5)
        if i % 500 == 0:
            print(f"  {i}/{len(keys)}")
    return preds, np.array(times[5:])  # as primeiras chamadas incluem o aquecimento


def save_predictions(path: Path, preds: dict, times: np.ndarray) -> None:
    np.savez_compressed(path, __times__=times, **{k.replace("/", "|"): v for k, v in preds.items()})


def load_predictions(path: Path) -> tuple[dict, np.ndarray]:
    data = np.load(path)
    return {k.replace("|", "/"): data[k] for k in data.files if k != "__times__"}, data["__times__"]


# ---------- AP (avaliador oficial do WIDER, em Python) ----------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre caixas (x, y, w, h), com a convenção +1 píxel do avaliador oficial."""
    a = np.c_[a[:, :2], a[:, :2] + a[:, 2:4]]
    b = np.c_[b[:, :2], b[:, :2] + b[:, 2:4]]
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def image_pr(pred: np.ndarray, gt_boxes: np.ndarray, counted: np.ndarray, thresholds: np.ndarray):
    """Propostas válidas e caras encontradas para cada limiar (pred ordenado por score)."""
    overlaps = iou_matrix(pred, gt_boxes)
    recalled = np.zeros(len(gt_boxes))
    proposal = np.ones(len(pred))
    pred_recall = np.zeros(len(pred))
    for h in range(len(pred)):
        j = overlaps[h].argmax()
        if overlaps[h, j] >= IOU:
            if not counted[j]:            # cara ignorada neste nível: a deteção não conta
                recalled[j], proposal[h] = -1, -1
            elif recalled[j] == 0:
                recalled[j] = 1
        pred_recall[h] = np.sum(recalled == 1)
    n_above = np.searchsorted(-pred[:, 4], -thresholds, side="right")  # previsões com score >= limiar
    out = np.zeros((len(thresholds), 2))
    has = n_above > 0
    out[has, 0] = np.cumsum(proposal == 1)[n_above[has] - 1]
    out[has, 1] = pred_recall[n_above[has] - 1]
    return out


def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def compute_ap(preds: dict, gt: dict) -> dict[str, float]:
    scores = np.concatenate([p[:, 4] for p in preds.values() if len(p)])
    lo, hi = scores.min(), scores.max()
    norm = {k: np.c_[p[:, :4], (p[:, 4] - lo) / (hi - lo)] for k, p in preds.items()}  # como o oficial
    thresholds = 1 - (np.arange(THRESH_NUM) + 1) / THRESH_NUM
    aps = {}
    for setting in SETTINGS:
        pr, n_faces = np.zeros((THRESH_NUM, 2)), 0
        for key, g in gt.items():
            n_faces += len(g[setting])
            if len(g["boxes"]) == 0 or len(norm[key]) == 0:
                continue
            counted = np.zeros(len(g["boxes"]), bool)
            counted[g[setting]] = True
            pr += image_pr(norm[key], g["boxes"], counted, thresholds)
        precision = np.divide(pr[:, 1], pr[:, 0], out=np.zeros(THRESH_NUM), where=pr[:, 0] > 0)
        aps[setting] = voc_ap(pr[:, 1] / n_faces, precision)
    return aps


# ---------- ponto de funcionamento e robustez (limiar da aplicação) ----------
def match(pred: np.ndarray, gt_boxes: np.ndarray) -> tuple[np.ndarray, int]:
    """Emparelhamento guloso 1-para-1 com IoU >= 0.5: (caras encontradas, deteções certas)."""
    found = np.zeros(len(gt_boxes), bool)
    if len(pred) == 0 or len(gt_boxes) == 0:
        return found, 0
    overlaps = iou_matrix(pred, gt_boxes)
    for h in range(len(pred)):
        candidates = np.where(found, -1.0, overlaps[h])
        j = candidates.argmax()
        if candidates[j] >= IOU:
            found[j] = True
    return found, int(found.sum())


def bin_label(value: float, bins) -> str:
    return next(label for lo, hi, label in bins if lo <= value < hi)


def operating_point(preds: dict, gt: dict) -> tuple[dict, list[dict]]:
    faces, true_pos, n_preds = [], 0, 0
    for key, g in gt.items():
        pred = preds[key][preds[key][:, 4] >= DETECTION_THRESHOLD]
        found, tp = match(pred, g["boxes"])
        true_pos, n_preds = true_pos + tp, n_preds + len(pred)
        valid = g["invalid"] == 0
        for k in np.where(valid)[0]:
            w, h = g["boxes"][k, 2:4]
            faces.append({"found": found[k], "side": np.sqrt(max(w * h, 0)),
                          "size": bin_label(np.sqrt(max(w * h, 0)), SIZE_BINS),
                          "faces_in_image": bin_label(valid.sum(), CROWD_BINS),
                          **{a: names[int(g[a][k])] for a, names in ATTRIBUTES.items()}})
    rows = []
    for factor, groups in GROUP_ORDER.items():
        for group in groups:
            sub = [f for f in faces if f[factor] == group]
            big = [f["found"] for f in sub if f["side"] >= 32]  # isola o efeito do tamanho
            if sub:
                rows.append({"factor": factor, "group": group, "n_faces": len(sub),
                             "recall": float(np.mean([f["found"] for f in sub])),
                             "n_faces_32px+": len(big), "recall_32px+": float(np.mean(big)) if big else ""})
    summary = {"threshold": DETECTION_THRESHOLD, "precision": true_pos / max(n_preds, 1),
               "recall": float(np.mean([f["found"] for f in faces])), "valid_faces": len(faces)}
    return summary, rows


# ---------- saídas ----------
def save_examples(preds: dict, gt: dict, n: int, out_dir: Path) -> None:
    """n imagens aleatórias com 3+ caras: ground truth a verde, deteções (>= 0.4) a vermelho."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(k for k, g in gt.items() if (g["invalid"] == 0).sum() >= 3)
    for key in np.random.default_rng(0).choice(candidates, n, replace=False):
        img = cv.imread(str(IMAGES_DIR / f"{key}.jpg"))
        for x, y, w, h in gt[key]["boxes"][gt[key]["invalid"] == 0].astype(int):
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        for x, y, w, h, _ in preds[key][preds[key][:, 4] >= DETECTION_THRESHOLD].astype(int):
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv.imwrite(str(out_dir / f"{key.replace('/', '__')}.jpg"), img)


def write_comparison() -> None:
    """Tabelas lado a lado de todos os detetores já avaliados. Só lê os resultados guardados."""
    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(OUT_ROOT.glob("*/metrics.json"))}
    rows = []
    for m in runs.values():
        params, mb = model_size(m["detector"])
        rows.append({"detector": m["detector"], "params_M": round(params, 2), "size_MB": round(mb, 1),
                     **{f"ap_{s}": round(m["ap"][s], 4) for s in SETTINGS},
                     "precision_at_0.4": round(m["operating_point"]["precision"], 4),
                     "recall_at_0.4": round(m["operating_point"]["recall"], 4),
                     **{f"ms_{k}": round(m["time_ms"][k], 1) for k in ("mean", "median", "p95")}})
    write_csv(OUT_ROOT / "comparison.csv", rows)

    # Robustez: recall a 0.4 de cada detetor por grupo (todas as caras e só as >= 32 px).
    robust = {name: list(csv.DictReader((OUT_ROOT / name / "robustness.csv").open())) for name in runs}
    reference = next(iter(robust.values()))
    assert all([(r["factor"], r["group"]) for r in rows_] == [(r["factor"], r["group"]) for r in reference]
               for rows_ in robust.values()), "robustness.csv com grupos diferentes entre detetores"
    robustness = [{"factor": r["factor"], "group": r["group"], "n_faces": r["n_faces"],
                   **{f"recall_{name}": pct(robust[name][i]["recall"]) for name in robust},
                   "n_faces_32px+": r["n_faces_32px+"],
                   **{f"recall_32px+_{name}": pct(robust[name][i]["recall_32px+"]) for name in robust}}
                  for i, r in enumerate(reference)]
    write_csv(OUT_ROOT / "robustness_comparison.csv", robustness)

    summary = [{"Detetor": r["detector"], "Parâmetros (M)": r["params_M"], "Ficheiro (MB)": r["size_MB"],
                **{f"AP {s}": f"{r[f'ap_{s}']:.1%}" for s in SETTINGS},
                "Precisão @0.4": f"{r['precision_at_0.4']:.1%}", "Recall @0.4": f"{r['recall_at_0.4']:.1%}",
                "ms média / mediana / p95": f"{r['ms_mean']} / {r['ms_median']} / {r['ms_p95']}"} for r in rows]
    (OUT_ROOT / "comparison.md").write_text(
        "# Comparação de detetores no WIDER Face (validação)\n\n"
        f"{next(iter(runs.values()))['images']} imagens; "
        f"AP com IoU >= {IOU} (avaliador oficial em Python); precisão e recall com o limiar da aplicação "
        f"({DETECTION_THRESHOLD}). Tempo: ms por imagem do WIDER (redimensionar para 640 + inferência + NMS).\n\n"
        + markdown_table(summary)
        + "\n\n## Robustez (recall @0.4 por grupo)\n\n"
        "Todas as caras e só as caras com pelo menos 32 px (isola o efeito do tamanho).\n\n"
        + markdown_table(robustness) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Avaliação do detetor no WIDER Face (val).")
    parser.add_argument("--detector", default=DETECTOR_MODEL, help="modelo em models/")
    parser.add_argument("--examples", type=int, default=0, help="guardar N imagens de exemplo")
    parser.add_argument("--reuse", action="store_true", help="usar as previsões já guardadas")
    parser.add_argument("--compare", action="store_true", help="só refazer as tabelas de comparação")
    args = parser.parse_args()
    if args.compare:
        write_comparison()
        return print(f"Tabelas em {OUT_ROOT}: comparison.csv, robustness_comparison.csv, comparison.md")

    out_dir = OUT_ROOT / Path(args.detector).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "predictions.npz"
    gt = load_ground_truth()

    if args.reuse and cache.exists():
        preds, times = load_predictions(cache)
    else:
        print(f"A correr {args.detector} em {len(gt)} imagens...")
        preds, times = run_detector(args.detector, sorted(gt))
        save_predictions(cache, preds, times)
    preds = {k: p[np.argsort(-p[:, 4], kind="stable")] for k, p in preds.items()}

    ap = compute_ap(preds, gt)
    op, robustness = operating_point(preds, gt)
    metrics = {"detector": args.detector, "images": len(gt), "ap": ap, "operating_point": op,
               "time_ms": {"mean": float(times.mean()), "median": float(np.median(times)),
                           "p95": float(np.percentile(times, 95))},
               "protocol": {"eval_threshold": EVAL_THRESHOLD, "iou": IOU, "input_size": 640,
                            "note": "redimensionamento nearest (biblioteca scrfd)"}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))
    write_csv(out_dir / "robustness.csv", robustness)
    if args.examples:
        save_examples(preds, gt, args.examples, out_dir / "examples")
    write_comparison()

    print(f"\n{args.detector} | AP easy {ap['easy']:.4f} | medium {ap['medium']:.4f} | hard {ap['hard']:.4f}")
    print(f"Tempo: média {metrics['time_ms']['mean']:.1f} ms | p95 {metrics['time_ms']['p95']:.1f} ms")
    print(f"Limiar {DETECTION_THRESHOLD}: precisão {op['precision']:.3f} | recall {op['recall']:.3f} "
          f"({op['valid_faces']} caras válidas)")
    print(f"Resultados em results/detection/{out_dir.name}/ e results/detection/comparison.csv")


if __name__ == "__main__":
    main()
