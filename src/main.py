"""
Análise facial em tempo real: deteção, identidade, idade/género e expressão.

Uso (a partir de qualquer pasta):
    python src/main.py                                # webcam 0
    python src/main.py --source 0 --source demo.mp4   # a tecla 's' alterna entre as fontes
    python src/main.py --detector det_10g.onnx        # outro detetor SCRFD

Carregar em 'h' para ver as teclas.
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from scrfd import Face

from age_gender import AgeGenderEstimator, AgeGenderResult
from alignment import align_face, keypoints_to_array
from capture import open_source, parse_source
from detection import MODELS_DIR, attribute_crop, detect, identity_crop, largest_face, load_detector
from expression import EXPRESSION_MODEL, ExpressionEstimator, ExpressionResult
from identity import FaceEmbedder, Gallery, IdentityResult
from monitoring import Monitor

WINDOW = "Facial Analysis"
MIN_FACE_SIZE = 40      # px; caras mais pequenas são detetadas mas não analisadas
ENROL_FRAMES = 5        # imagens captadas no registo pela webcam
ENROL_INTERVAL_MS = 400

FONT = cv.FONT_HERSHEY_SIMPLEX
WHITE, GREY, GREEN, RED, YELLOW = (255, 255, 255), (160, 160, 160), (0, 220, 0), (0, 0, 255), (0, 220, 255)

MODULES = {"1": ("age_gender", "Idade/genero"), "2": ("expression", "Expressao"),
           "3": ("identity", "Identidade"), "4": ("landmarks", "Landmarks")}
HELP = ["espaco  parar/retomar camara", "s  fonte seguinte", "1-4  ligar/desligar modulos",
        "e  registar pessoa (terminal)", "d  apagar pessoa (terminal)", "x  apagar galeria",
        "h  ajuda", "q  sair"]


@dataclass
class FaceAnalysis:
    face: Face
    analysed: bool                       # False se a cara for pequena demais
    identity: IdentityResult | None = None
    age_gender: AgeGenderResult | None = None
    expression: ExpressionResult | None = None


def put(img, text, org, color=WHITE, scale=0.5, thickness=1):
    """Texto com contorno preto, legível sobre qualquer fundo."""
    cv.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
    cv.putText(img, text, org, FONT, scale, color, thickness, cv.LINE_AA)


class App:
    def __init__(self, args):
        self.args = args
        self.sources = args.source or ["0"]
        self.source_index = 0
        self.detector = load_detector(args.detector)
        self.age_gender = AgeGenderEstimator(str(MODELS_DIR / "genderage.onnx"))
        self.embedder = FaceEmbedder()
        self.gallery = Gallery.load()
        self.expression = ExpressionEstimator() if EXPRESSION_MODEL.exists() else None
        if self.expression is None:
            print(f"Aviso: {EXPRESSION_MODEL.name} não encontrado, expressão desligada.")

        self.modules = {"age_gender": True, "expression": self.expression is not None,
                        "identity": True, "landmarks": True}
        self.monitor = Monitor()
        self.cap = open_source(self.sources[0])
        self.paused = False
        self.show_help = False
        self.last_frame = None
        self.message, self.message_until = "", 0.0

    # ---------- análise ----------
    def analyse(self, frame) -> list[FaceAnalysis]:
        with self.monitor.stage("detection"):
            faces = detect(self.detector, frame, self.args.det_threshold)
        results = []
        for face in faces:
            p1, p2 = face.bbox.upper_left, face.bbox.lower_right
            r = FaceAnalysis(face, analysed=min(p2.x - p1.x, p2.y - p1.y) >= MIN_FACE_SIZE)
            if r.analysed and self.modules["identity"]:
                with self.monitor.stage("identity"):
                    r.identity = self.gallery.match(self.embedder.embed(identity_crop(frame, face)))
            if r.analysed and self.modules["age_gender"]:
                with self.monitor.stage("age_gender"):
                    r.age_gender = self.age_gender.predict(attribute_crop(frame, face))
            if r.analysed and self.modules["expression"]:
                with self.monitor.stage("expression"):
                    crop = align_face(frame, keypoints_to_array(face), size=224, margin=0.2)
                    r.expression = self.expression.predict(crop)
            results.append(r)
        return results

    # ---------- desenho ----------
    def draw_face(self, frame, r: FaceAnalysis):
        p1, p2 = r.face.bbox.upper_left, r.face.bbox.lower_right
        x1, y1, x2, y2 = int(p1.x), int(p1.y), int(p2.x), int(p2.y)
        color = WHITE if r.identity is None else (GREEN if r.identity.is_known else RED)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        put(frame, f"det {r.face.probability:.2f}", (x1 + 3, y1 + 14), GREY, 0.4)

        if self.modules["landmarks"]:
            for kp in r.face.keypoints.model_dump().values():
                cv.circle(frame, (int(kp["x"]), int(kp["y"])), 2, YELLOW, -1)

        if not r.analysed:
            put(frame, "pequena demais", (x1, y2 + 16), GREY, 0.45)
            return
        if r.identity is not None:  # identidade em destaque, por cima da caixa
            score = "" if np.isnan(r.identity.score) else f" {r.identity.score:.2f}"
            put(frame, f"{r.identity.name}{score}", (x1, y1 - 8), color, 0.7, 2)

        lines = []  # atributos por baixo da caixa; incertos a cinzento com "?"
        if r.age_gender is not None:
            ag = r.age_gender
            gender = "genero ?" if ag.is_uncertain else f"{ag.gender} {ag.gender_confidence:.2f}"
            lines.append((f"idade ~{ag.age_range} | {gender}", GREY if ag.is_uncertain else WHITE))
        if r.expression is not None:
            ex = r.expression
            lines.append((f"{ex.label}{'?' if ex.is_uncertain else ''} {ex.confidence:.2f}",
                          GREY if ex.is_uncertain else WHITE))
        for i, (text, c) in enumerate(lines):
            put(frame, text, (x1, y2 + 18 + 18 * i), c, 0.5)

    def draw_panel(self, frame):
        m = self.monitor
        lat = " ".join(f"{k[:3]} {v:.0f}" for k, v in m.latency_ms.items() if k != "total")
        source = f"{Path(self.sources[self.source_index]).name} ({self.source_index + 1}/{len(self.sources)})"
        lines = [f"FPS {m.fps:.1f} | total {m.latency_ms.get('total', 0):.0f} ms",
                 f"ms: {lat}",
                 f"CPU {m.cpu_percent:.0f}% | RAM {m.memory_mb:.0f} MB",
                 f"Fonte {source}{' | PARADA' if self.paused else ''}",
                 *(f"[{key}] {label}: {'on' if self.modules[name] else 'off'}"
                   for key, (name, label) in MODULES.items()),
                 f"Galeria: {len(self.gallery.names())} | thr {self.gallery.threshold:.3f}",
                 *(HELP if self.show_help else ["h: ajuda"])]
        if time.time() < self.message_until:
            lines.append(self.message)

        height = 10 + 17 * len(lines)
        overlay = frame.copy()
        cv.rectangle(overlay, (0, 0), (265, height), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, text in enumerate(lines):
            put(frame, text, (6, 18 + 17 * i), YELLOW if text == self.message else WHITE, 0.42)

    def notify(self, text: str):
        print(text)
        self.message, self.message_until = text, time.time() + 4

    # ---------- câmara e fontes ----------
    def toggle_camera(self):
        self.paused = not self.paused
        if self.paused:
            self.cap.release()
            self.monitor.pause()
        else:
            self.cap = open_source(self.sources[self.source_index])

    def next_source(self):
        if len(self.sources) == 1:
            return self.notify("Só há uma fonte (usar --source mais vezes).")
        new_index = (self.source_index + 1) % len(self.sources)
        try:
            new_cap = open_source(self.sources[new_index])
        except RuntimeError as e:
            return self.notify(str(e))
        self.cap.release()
        self.cap, self.source_index, self.paused = new_cap, new_index, False
        self.monitor.pause()

    def read_frame(self):
        ok, frame = self.cap.read()
        if not ok and isinstance(parse_source(self.sources[self.source_index]), str):
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)  # vídeo chegou ao fim: recomeçar
            ok, frame = self.cap.read()
        return frame if ok else None

    # ---------- galeria (os nomes escrevem-se no terminal) ----------
    def ask(self, prompt: str) -> str:
        if self.last_frame is not None:
            frame = self.last_frame.copy()
            put(frame, "Responder no terminal...", (10, frame.shape[0] - 15), YELLOW, 0.7, 2)
            cv.imshow(WINDOW, frame)
            cv.waitKey(1)
        answer = input(prompt).strip()
        self.monitor.pause()
        return answer

    def enrol(self):
        if self.paused:
            return self.notify("Retomar a câmara antes de registar.")
        name = self.ask("Nome a registar (vazio cancela): ")
        if not name or self.ask(f"{name} deu consentimento para o registo? [s/N] ").lower() != "s":
            return self.notify("Registo cancelado.")

        for _ in range(5):  # descartar frames antigos que ficaram no buffer durante o input()
            self.cap.grab()
        embeddings = []
        for i in range(ENROL_FRAMES):
            frame = self.read_frame()
            face = largest_face(self.detector, frame) if frame is not None else None
            if face is not None:
                embeddings.append(self.embedder.embed(identity_crop(frame, face)))
                put(frame, f"A registar {name}: {i + 1}/{ENROL_FRAMES}", (10, 30), YELLOW, 0.7, 2)
                cv.imshow(WINDOW, frame)
            cv.waitKey(ENROL_INTERVAL_MS)
        if not embeddings:
            return self.notify("Nenhuma cara detetada: registo cancelado.")
        try:
            self.gallery.enrol(name, embeddings)
        except ValueError as e:
            return self.notify(str(e))
        self.gallery.save()
        self.notify(f"{name} registado com {len(embeddings)} imagens.")

    def delete(self):
        name = self.ask(f"Apagar quem? {self.gallery.names()}: ")
        if self.gallery.remove(name):
            self.gallery.save()
            self.notify(f"{name} apagado da galeria.")
        else:
            self.notify(f"'{name}' não está na galeria.")

    def clear_gallery(self):
        if self.ask("Apagar a galeria toda? [s/N] ").lower() == "s":
            self.gallery.clear()
            self.notify("Galeria apagada.")

    # ---------- ciclo principal ----------
    def handle_key(self, key: int) -> bool:
        """Devolve False para sair."""
        char = chr(key) if 0 <= key < 256 else ""
        if char == "q":
            return False
        actions = {" ": self.toggle_camera, "s": self.next_source, "e": self.enrol, "d": self.delete,
                   "x": self.clear_gallery, "h": lambda: setattr(self, "show_help", not self.show_help)}
        if char in actions:
            actions[char]()
        elif char in MODULES:
            name, label = MODULES[char]
            if name == "expression" and self.expression is None:
                self.notify("Modelo de expressão não encontrado.")
            else:
                self.modules[name] = not self.modules[name]
        return True

    def run(self):
        cv.namedWindow(WINDOW)
        while True:
            if self.paused:
                frame = self.last_frame.copy() if self.last_frame is not None else np.zeros((480, 640, 3), np.uint8)
            else:
                frame = self.read_frame()
                if frame is None:
                    self.notify("Sem imagem da fonte: câmara parada.")
                    self.toggle_camera()
                    continue
                self.last_frame = frame.copy()
                with self.monitor.stage("total"):
                    for r in self.analyse(frame):
                        self.draw_face(frame, r)
                self.monitor.end_frame()

            self.draw_panel(frame)
            cv.imshow(WINDOW, frame)
            key = cv.waitKey(30 if self.paused else 1) & 0xFF
            if not self.handle_key(key) or cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1:
                break

        self.cap.release()
        cv.destroyAllWindows()
        saved = self.monitor.save({"source": self.sources[self.source_index], "detector": self.args.detector,
                                   **{f"module_{k}": v for k, v in self.modules.items()}})
        if saved:
            print(f"Estatísticas da sessão guardadas em {saved}")


def main():
    parser = argparse.ArgumentParser(description="Análise facial em tempo real.")
    parser.add_argument("--source", action="append",
                        help="índice da webcam ou caminho de um vídeo (pode repetir-se)")
    parser.add_argument("--detector", default="scrfd.onnx", help="modelo SCRFD em src/models/")
    parser.add_argument("--det-threshold", type=float, default=0.4, help="confiança mínima da deteção")
    App(parser.parse_args()).run()


if __name__ == "__main__":
    main()
