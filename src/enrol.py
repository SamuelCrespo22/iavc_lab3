"""
Gestão da galeria de identidades a partir de fotos.

Exemplos (a partir da raiz do repositório):
    # Registar todas as pessoas de data/dados_caras com as mesmas fotos usadas na avaliação
    python src/enrol.py --all data/dados_caras --images frontal frontal2

    # Registar uma pessoa a partir de uma pasta (nome = nome da pasta, por omissão)
    python src/enrol.py --folder data/dados_caras/Joao

    python src/enrol.py --list
    python src/enrol.py --remove Joao
    python src/enrol.py --clear

Só é guardado o embedding de cada foto (gallery/gallery.npz), nunca a imagem.
"""
import argparse
from pathlib import Path

import numpy as np

from detection import load_detector
from identity import Gallery, FaceEmbedder, embed_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def person_images(folder: Path, stems: list[str] | None) -> list[Path]:
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if stems:
        images = [p for p in images if p.stem in stems]
    return images


def embeddings_from_images(paths: list[Path], detector, embedder: FaceEmbedder) -> list[np.ndarray]:
    embeddings = []
    for path in paths:
        embedding = embed_image(path, detector, embedder)
        if embedding is None:
            print(f"  ! sem cara detetada: {path.name}")
            continue
        embeddings.append(embedding)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Gerir a galeria de identidades.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--folder", type=Path, help="pasta com as fotos de uma pessoa")
    action.add_argument("--all", type=Path, help="pasta com uma subpasta por pessoa")
    action.add_argument("--list", action="store_true", help="listar identidades registadas")
    action.add_argument("--remove", metavar="NOME", help="remover uma identidade")
    action.add_argument("--clear", action="store_true", help="apagar a galeria toda")
    parser.add_argument("--name", help="nome a registar com --folder (por omissão: nome da pasta)")
    parser.add_argument("--images", nargs="+", metavar="NOME",
                        help="usar só estas fotos, sem extensão (ex.: frontal frontal2)")
    args = parser.parse_args()

    gallery = Gallery.load()

    if args.list:
        for name, emb in gallery.embeddings.items():
            print(f"{name}: {len(emb)} embeddings")
        print(f"{len(gallery.names())} identidades | threshold = {gallery.threshold:.3f}")
        return

    if args.remove:
        found = gallery.remove(args.remove)
        gallery.save()
        print(f"Removido: {args.remove}" if found else f"Não existe: {args.remove}")
        return

    if args.clear:
        gallery.clear()
        print("Galeria apagada.")
        return

    folders = [args.folder] if args.folder else sorted(p for p in args.all.iterdir() if p.is_dir())
    detector, embedder = load_detector(), FaceEmbedder()

    for folder in folders:
        name = args.name if (args.folder and args.name) else folder.name
        embeddings = embeddings_from_images(person_images(folder, args.images), detector, embedder)
        if not embeddings:
            print(f"{name}: nenhuma foto utilizável, não registado.")
            continue
        gallery.enrol(name, embeddings)
        print(f"{name}: {len(embeddings)} fotos registadas")

    gallery.save()
    print(f"Galeria guardada: {len(gallery.names())} identidades.")


if __name__ == "__main__":
    main()
