"""
Configuração da aplicação: caminhos, modelos e limiares num só sítio.
Para mudar um limiar ou a localização de um dataset, editar só este ficheiro.
"""
from pathlib import Path

import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---- Pastas (nenhuma é versionada: ver .gitignore) ----
MODELS_DIR = ROOT_DIR / "models"      # pesos dos modelos
DATA_DIR = ROOT_DIR / "data"          # datasets
GALLERY_DIR = ROOT_DIR / "gallery"    # embeddings da galeria (dados biométricos)
RESULTS_DIR = ROOT_DIR / "results"    # métricas e CSV das avaliações

# ---- Datasets ----
UTKFACE_DIR = DATA_DIR / "UTKFace"
WIDER_DIR = DATA_DIR / "WIDER"
GALLERY_PHOTOS_DIR = DATA_DIR / "dados_caras"   # uma subpasta por pessoa
AFFECTNET_DIR = ROOT_DIR / "AffectNetCustom"    # train/ val/ test/
BTFER_DIR = ROOT_DIR / "BTFER"

# ---- Modelos ----
DETECTOR_MODEL = "scrfd.onnx"         # nome do ficheiro em MODELS_DIR (alternativa: det_10g.onnx)
AGE_GENDER_MODEL = MODELS_DIR / "genderage.onnx"
EMBEDDING_MODEL = MODELS_DIR / "w600k_r50.onnx"
EXPRESSION_MODEL = MODELS_DIR / "expression_best.onnx"
EXPRESSION_CHECKPOINT = MODELS_DIR / "expression_best.pth"

# ---- Limiares ----
DETECTION_THRESHOLD = 0.4
MIN_FACE_SIZE = 40
GENDER_UNCERTAIN_BELOW = 0.65
EXPRESSION_UNCERTAIN_BELOW = 0.5
PROVISIONAL_IDENTITY_THRESHOLD = 0.35 # só até eval_identity calibrar (gallery/threshold.json)
MAX_IDENTITIES = 19

# ---- Câmara ----
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480


def onnx_providers() -> list[str]:
    """CoreML quando disponível (macOS), CPU como fallback."""
    available = ort.get_available_providers()
    return [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
