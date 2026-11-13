# Análise Facial em Tempo Real e Reconhecimento de Identidade

Trabalho Prático #3 de **Advanced Computer Vision / AI for Computer Vision** (2026/27).

Aplicação local, em Python, que processa o stream da webcam do portátil e faz:

1. **Deteção de caras** — bounding box, confiança e 5 landmarks faciais (SCRFD).
2. **Alinhamento** — crop normalizado a partir dos landmarks (template ArcFace).
3. **Estimação de atributos** — idade aparente, rótulo de género definido pelo dataset e expressão facial visível.
4. **Identificação em galeria fechada** (< 20 pessoas, com consentimento) — embeddings ArcFace + similaridade cosseno, com rejeição "Unknown".
5. **Visualização e monitorização** — overlay em tempo real com FPS e latência.

O enunciado completo está em [assignment.pdf](assignment.pdf).

## Estado atual

| Fase | Descrição | Estado |
|------|-----------|--------|
| 1 | Captura da webcam, deteção de caras, bounding boxes e keypoints | ✅ Feito |
| — | Alinhamento de caras (112×112 para ArcFace, 224×224 com margem para atributos) | ✅ Feito |
| 2 | Idade/género e expressão facial | ⏳ Por fazer |
| 3 | Galeria de identidades (registo, embeddings, threshold "Unknown") | ⏳ Por fazer |
| 4 | Otimização e avaliação | ⏳ Por fazer |
| 5 | Aplicação final e relatório | ⏳ Por fazer |

## Estrutura do repositório

```
.
├── src/
│   ├── main.py        # Ponto de entrada: loop da webcam, deteção e overlay
│   ├── capture.py     # Abertura da câmara (OpenCV VideoCapture)
│   ├── alignment.py   # Alinhamento por landmarks (template ArcFace)
│   └── models/        # Pesos ONNX (não versionados — ver abaixo)
├── data/              # Datasets (não versionados — ver abaixo)
├── assignment.pdf     # Enunciado
└── requirements.txt
```

## Instalação

Requer Python ≥ 3.10 (testado com Python 3.14 em macOS / Apple Silicon).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependências principais:

| Pacote | Uso |
|--------|-----|
| `opencv-python` | Captura de vídeo, alinhamento (transformação afim) e interface |
| `numpy` | Operações sobre arrays |
| `onnxruntime` | Inferência dos modelos ONNX (CoreML no macOS, CPU como fallback) |
| `scrfd` | Wrapper do detetor SCRFD (pré/pós-processamento e NMS) |
| `pillow` | Conversão de frames para o formato esperado pelo `scrfd` |

## Modelos

Os pesos **não estão incluídos no repositório** (por tamanho e por termos de licença). Devem ser colocados em `src/models/`:

| Ficheiro | Modelo | Uso |
|----------|--------|-----|
| `scrfd.onnx` | SCRFD (variante leve, com keypoints) | Deteção de caras — usado atualmente |
| `det_10g.onnx` | SCRFD-10GF | Deteção (alternativa mais precisa, para comparação) |
| `genderage.onnx` | InsightFace gender-age | Idade e género (Fase 2) |
| `w600k_r50.onnx` | ArcFace ResNet-50 (WebFace600K) | Embeddings de identidade (Fase 3) |
| `2d106det.onnx`, `1k3d68.onnx` | Landmarks 2D/3D | Opcionais |

Os ficheiros `det_10g.onnx`, `genderage.onnx`, `w600k_r50.onnx`, `2d106det.onnx` e `1k3d68.onnx` vêm do pack **buffalo_l** do [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package):

```bash
mkdir -p src/models
curl -L -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o /tmp/buffalo_l.zip -d src/models
```

> **Licença:** os modelos pré-treinados do InsightFace estão disponíveis apenas para **investigação não comercial**. Não redistribuir os pesos.

## Datasets

Os datasets ficam em `data/` (ignorado pelo git):

- **UTKFace** (`data/UTKFace/`) — avaliação/fine-tuning de idade e género. O nome de cada imagem segue o formato `[idade]_[género]_[etnia]_[data].jpg.chip.jpg`.
- **WIDER Face** — avaliação do detetor (a adicionar).
- **AffectNet / BTFER / FERPlus** — avaliação de expressões (a adicionar).

## Execução

A partir da raiz do repositório (os caminhos dos modelos são relativos à raiz):

```bash
source .venv/bin/activate
python src/main.py
```

- Janela **SCRFD**: stream da webcam com bounding boxes, confiança, keypoints e FPS.
- Janela **aligned**: o último crop alinhado (112×112), para verificar o alinhamento visualmente.
- Carregar em **`q`** para sair.

No macOS, na primeira execução, é preciso dar permissão de acesso à câmara ao terminal/IDE (Definições do Sistema → Privacidade e Segurança → Câmara).

## Limitações conhecidas

- O caminho do modelo e os parâmetros (threshold de deteção 0.4, resolução 640×480, câmara 0) estão fixos no código; ainda não há seleção de fonte nem ficheiro de configuração.
- Ainda não é possível usar um ficheiro de vídeo como entrada.
- A janela `aligned` mostra apenas a última cara detetada quando há várias caras.
- O FPS mostrado é instantâneo (sem média), por isso oscila de frame para frame.

## Considerações éticas

- **Idade** é uma estimativa de idade *aparente*, não a idade real.
- **Género** é uma classificação visual limitada às categorias do dataset de treino; não é uma determinação da identidade de género de ninguém.
- **Expressão** descreve a configuração visível da cara; não é evidência do estado emocional da pessoa.
- A **galeria de identidades** deve conter apenas voluntários que deram consentimento, guardar só os dados necessários, ter mecanismo de remoção e nunca ser incluída no repositório (a pasta `gallery/` está no `.gitignore`). O sistema é um demonstrador académico e não deve ser usado para controlo de acessos, presenças ou vigilância.
