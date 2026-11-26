"""
Registo da galeria de identidades a partir de fotos (na aplicação: teclas e/d/x).

    python -m src.enrol --all data/dados_caras --images frontal frontal2   # uma subpasta por pessoa
    python -m src.enrol --list
    python -m src.enrol --clear     # antes de registar de novo (o registo junta-se ao que já existe)

Só é guardado o embedding de cada foto (gallery/gallery.npz), nunca a imagem.
"""
import argparse
from pathlib import Path

from src.pipeline.detection import load_detector
from src.pipeline.identity import FaceEmbedder, Gallery, embed_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(description="Gerir a galeria de identidades.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--all", type=Path, help="pasta com uma subpasta por pessoa (nome = nome da subpasta)")
    action.add_argument("--list", action="store_true", help="listar identidades registadas")
    action.add_argument("--clear", action="store_true", help="apagar a galeria toda")
    parser.add_argument("--images", nargs="+", metavar="NOME", help="usar só estas fotos, sem extensão")
    args = parser.parse_args()
    gallery = Gallery.load()

    if args.list:
        for name, embeddings in gallery.embeddings.items():
            print(f"{name}: {len(embeddings)} embeddings")
        print(f"{len(gallery.names())} identidades | threshold = {gallery.threshold:.3f}")
    elif args.clear:
        gallery.clear()
        print("Galeria apagada.")
    else:
        detector, embedder = load_detector(), FaceEmbedder()
        for folder in sorted(p for p in args.all.iterdir() if p.is_dir()):
            embeddings = []
            for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
                if args.images and path.stem not in args.images:
                    continue
                embedding = embed_image(path, detector, embedder)
                if embedding is None:
                    print(f"  ! sem cara detetada: {path.name}")
                else:
                    embeddings.append(embedding)
            if embeddings:
                gallery.enrol(folder.name, embeddings)
            print(f"{folder.name}: {len(embeddings)} fotos registadas")
        gallery.save()
        print(f"Galeria guardada: {len(gallery.names())} identidades.")


if __name__ == "__main__":
    main()
