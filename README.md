# Análise Facial em Tempo Real e Reconhecimento de Identidade

Trabalho Prático #3 de **Advanced Computer Vision / AI for Computer Vision** (2026/27).

Aplicação local, em Python, que processa o stream da webcam do portátil e faz:

1. **Deteção de caras** — bounding box, confiança e 5 landmarks faciais (SCRFD).
2. **Alinhamento** — crop normalizado a partir dos landmarks (template ArcFace).
3. **Estimação de atributos** — idade aparente, rótulo de género definido pelo dataset e expressão facial visível: por omissão com um modelo multitarefa (ResNet-50); em alternativa, com dois modelos separados (genderage + MobileNetV3 de expressão).
4. **Identificação em galeria fechada** (< 20 pessoas, com consentimento) — embeddings ArcFace + similaridade cosseno, com rejeição "Unknown".
5. **Visualização e monitorização** — overlay em tempo real com FPS e latência.

O enunciado completo está em [assignment.pdf](assignment.pdf).

## Estado atual

O estado detalhado está em [MILESTONES.md](MILESTONES.md). Em resumo (28 nov):
- **Feito:** deteção, alinhamento, identidade, idade/género (também por condição da imagem) e expressão, todos avaliados; interface e monitorização; modelo multitarefa (MobileNetV3 a 112/160/224 e ResNet-50 a 224) treinado e avaliado, com a ResNet-50 na app; testes de sistema ao vivo; tabelas de comparação em `results/detection/comparison.md` e `results/comparison/comparison.md`.
- **Falta:** relatório e vídeo.

## Estrutura do repositório

```
.
├── src/
│   ├── main.py                  # Aplicação: webcam/vídeo, análise, overlay, teclas
│   ├── enrol.py                 # Registo da galeria a partir de fotos
│   ├── config.py                # Caminhos, modelos e limiares
│   ├── pipeline/                # Módulos da aplicação
│   │   ├── capture.py           #   webcam ou ficheiro de vídeo
│   │   ├── detection.py         #   SCRFD e recortes para cada modelo
│   │   ├── alignment.py         #   alinhamento por landmarks (template ArcFace)
│   │   ├── identity.py          #   embeddings ArcFace e galeria
│   │   ├── age_gender.py        #   idade e género (InsightFace genderage)
│   │   ├── expression.py        #   expressão facial (MobileNetV3, AffectNet)
│   │   ├── multitask.py         #   idade, género e expressão com o modelo multitarefa
│   │   └── monitoring.py        #   FPS, latência, CPU e memória
│   ├── evaluation/              # Avaliação de cada componente
│   │   ├── eval_detection_wider.py   # WIDER Face; --compare: tabelas dos detetores
│   │   ├── eval_identity.py          # galeria: rank-1, FAR/FRR, threshold
│   │   ├── eval_attributes.py        # expressão, idade e género de uma opção (separate ou multitarefa)
│   │   ├── eval_conditions.py        # idade e género com a imagem degradada (desfocada, escura, ...)
│   │   ├── eval_latency.py           # inferência de cada modelo com o CoreML e só com a CPU
│   │   ├── compare_models.py         # tabelas de todas as opções de atributos e da identidade
│   │   └── tables.py                 # CSV, Markdown e tamanho dos modelos
│   ├── graphs/                  # Figuras do relatório (python -m src.graphs -> results/images/)
│   └── training/
│       ├── train_expression.py  # Fine-tuning da expressão no AffectNet
│       ├── multitask_model.py   # backbone partilhado (MobileNetV3 / ResNet): expressão, género e idade
│       ├── prepare_multitask.py #   lista de imagens e labels (data/multitask/labels.csv)
│       └── train_multitask.py   #   treino e exportação para ONNX (models/multitask/<corrida>/)
├── colab/
│   └── train_multitask.ipynb    # o mesmo treino numa GPU do Google Colab
├── models/                      # Pesos (não versionados; ver abaixo)
├── data/                        # Datasets (não versionados; ver abaixo)
├── gallery/                     # Embeddings da galeria (não versionados)
├── results/                     # Resultados das avaliações (não versionados)
├── relatorio/                   # Relatório (LaTeX): rel.tex + secoes/ (não versionado)
├── assignment.pdf               # Enunciado
├── MILESTONES.md                # Estado do trabalho e decisões
└── requirements.txt
```

## Instalação

Requer Python ≥ 3.10 (testado com Python 3.12 em macOS / Apple Silicon).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

| Pacote | Uso |
|--------|-----|
| `opencv-python`, `numpy`, `pillow` | Vídeo, alinhamento, interface e operações sobre imagens |
| `onnxruntime` | Inferência de todos os modelos na aplicação (CoreML no macOS, CPU como fallback) |
| `scrfd` | Wrapper do detetor SCRFD (pré/pós-processamento e NMS) |
| `psutil` | CPU e memória no painel de monitorização |
| `matplotlib` | Figuras do relatório (`src/graphs/`) |
| `scikit-learn` | Métricas das avaliações |
| `torch`, `torchvision`, `onnx`, `onnxscript` | Só para treinar (expressão e multitarefa) e exportar para ONNX |

## Modelos

Nenhum peso está no repositório git (por tamanho e por termos de licença). Ficam todos em `models/`, na raiz do repositório. Por omissão, a aplicação precisa de `scrfd.onnx`, `w600k_r50.onnx` e da pasta `multitask/r50_224_full/`.

### Treinados por nós (incluídos no zip da entrega)

| Ficheiros em `models/` | Modelo | Uso |
|---|---|---|
| `multitask/r50_224_full/`: `multitask.onnx`, `multitask.onnx.data`, `config.json` | ResNet-50 multitarefa (expressão, género e idade), 224 px | Atributos na aplicação (por omissão) |
| `expression/`: `expression_best.onnx`, `expression_best.onnx.data` | MobileNetV3-Small com fine-tuning no AffectNet | Expressão com `--attributes separate` |

O `.onnx.data` tem os pesos: cada par fica na mesma pasta, sem mudar os nomes. Geram-se de novo com `train_multitask.py` (ou o notebook do Colab) e `train_expression.py`, que gravam também o checkpoint PyTorch (`.pth`, não incluído). Partem dos pesos do ImageNet do torchvision e foram treinados com o AffectNet e o UTKFace, por isso destinam-se só a uso académico, não comercial.

Os índices de saída da expressão seguem a ordem das pastas `0`–`7` do AffectNet usado no treino, que não é a ordem oficial do AffectNet (o dataset não a documenta; identificámo-la a partir das imagens de cada pasta). A ordem está em `CLASSES`, em `src/pipeline/expression.py`.

### Pré-treinados (a descarregar)

| Ficheiro em `models/` | Modelo | Uso |
|---|---|---|
| `scrfd.onnx` | SCRFD-2.5GF com keypoints (0.82 M parâmetros; confirmado pelo AP no WIDER) | Deteção de caras (por omissão) |
| `det_10g.onnx` | SCRFD-10GF | Deteção (alternativa mais precisa, `--detector det_10g.onnx`) |
| `w600k_r50.onnx` | ArcFace ResNet-50 (WebFace600K) | Embeddings de identidade |
| `genderage.onnx` | InsightFace gender-age | Idade e género com `--attributes separate` |

O `scrfd.onnx` vem do repositório [cospectrum/scrfd](https://github.com/cospectrum/scrfd/tree/main/models) (o da biblioteca `scrfd`); os outros três, do pack **buffalo_l** do [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package):

```bash
mkdir -p models
curl -L -o models/scrfd.onnx https://raw.githubusercontent.com/cospectrum/scrfd/main/models/scrfd.onnx
curl -L -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o /tmp/buffalo_l.zip -d models    # traz também 2d106det.onnx e 1k3d68.onnx, que não são usados
```

> **Licença:** os modelos pré-treinados do InsightFace estão disponíveis apenas para **investigação não comercial**. O `scrfd.onnx` do cospectrum/scrfd é um derivado do InsightFace: esse repositório aplica a licença Apache 2.0 só ao código e diz que o modelo é *"used for testing purposes only"*. Por isso segue as mesmas condições. Nenhum destes pesos é redistribuído.

## Datasets

Nenhum dataset é versionado nem incluído na entrega. São usados só para investigação académica, nos termos de cada um. As localizações estão em `src/config.py`:

| Dataset | Origem | Pasta | Uso |
|---|---|---|---|
| WIDER Face (val) | [shuoyang1213.me/WIDERFACE](http://shuoyang1213.me/WIDERFACE/) | `data/WIDER/` (`WIDER_val/`, `wider_face_split/`, `eval_tools/`) | Avaliação do detetor |
| UTKFace (Aligned & Cropped, 200 px) | [susanqq.github.io/UTKFace](https://susanqq.github.io/UTKFace/) | `data/UTKFace/` | Idade e género: treino do multitarefa (70%) e avaliação (15% de teste). Labels no nome: `[idade]_[género]_[etnia]_[data].jpg.chip.jpg` |
| AffectNet alinhado (112 px, 8 classes; treino até 5000 por classe, validação 100, teste 400; divisão já feita) | [kaggle.com/datasets/yakhyokhuja/affectnetaligned](https://www.kaggle.com/datasets/yakhyokhuja/affectnetaligned) | `data/AffectNetCustom/` (`train/`, `val/`, `test/`, uma pasta `0`–`7` por classe) | Treino e avaliação intra-dataset da expressão |
| BTFER (256 px, 7 classes, 300 por classe) | [kaggle.com/datasets/gipinze/btfer7](https://www.kaggle.com/datasets/gipinze/btfer7) | `data/BTFER/` | Avaliação cross-dataset da expressão |
| Fotos da galeria | Fotos dos próprios participantes | `data/dados_caras/<pessoa>/` | Avaliação da identidade (nunca sair do portátil; ver [Galeria de identidades](#galeria-de-identidades)) |
| Divisão do multitarefa | `prepare_multitask.py` | `data/multitask/labels.csv` | AffectNet + UTKFace, com o split de cada imagem |

## Execução

Todos os comandos correm **a partir da raiz do repositório**, com `python -m`:

```bash
source .venv/bin/activate
python -m src.main                                  # aplicação (webcam 0)
python -m src.main --source 0 --source video.mp4    # 's' alterna entre as fontes
python -m src.main --attributes separate            # genderage + MobileNet de expressão (por omissão: multitask, MULTITASK_RUN)

python -m src.enrol --all data/dados_caras --images frontal frontal2   # registar a galeria
python -m src.enrol --list | --clear

python -m src.evaluation.eval_detection_wider --detector scrfd.onnx   # results/detection/ (--compare: tabelas)
python -m src.evaluation.eval_identity              # resultados em results/identity/
python -m src.evaluation.eval_attributes --model separate   # pipeline atual -> results/attributes/separate/
python -m src.evaluation.eval_attributes --model mnet160    # uma corrida multitarefa -> results/attributes/mnet160/
python -m src.evaluation.eval_conditions --model r50_224_full  # idade/género por condição -> .../by_condition.csv
python -m src.evaluation.compare_models             # todas as opções -> results/comparison/comparison.md
python -m src.evaluation.eval_latency               # inferência CoreML vs CPU -> results/system/model_latency.csv
python -m src.graphs                               # figuras do relatório -> results/images/ (PDF e PNG)

python -m src.training.train_expression             # models/expression/: .pth e .onnx (+ .onnx.data), verificado

python -m src.training.prepare_multitask            # uma vez: data/multitask/labels.csv
python -m src.training.train_multitask --run mnet160 --input-size 160                          # MobileNetV3
python -m src.training.train_multitask --run r50_224_full --arch resnet50 --input-size 224   # ResNet-50 (na app)
```

A ResNet-50 treina-se numa GPU do Colab com [colab/train_multitask.ipynb](colab/train_multitask.ipynb) (instruções no próprio notebook); a pasta da corrida volta depois para `models/multitask/`.

Organização dos resultados (`results/`, não versionado): `detection/<detetor>/`, `identity/`, `attributes/<opção>/` (uma pasta por opção de atributos, todas com o mesmo formato), `comparison/` (tabelas para o relatório), `system/` (sessões da aplicação e latência dos modelos) e `images/` (figuras do relatório).

O relatório está em `relatorio/` (não versionado; `rel.tex` e uma secção por ficheiro em `secoes/`). Cada tabela e figura tem um comentário `% Fonte:` com o ficheiro de `results/` e o comando que o gera. As figuras vêm de `results/images/`; no Overleaf, pôr os PDF numa pasta `images/` ao lado do `rel.tex`.

Na aplicação, carregar em **`h`** para ver as teclas: parar/retomar a câmara, mudar de fonte, ligar/desligar módulos, registar/apagar pessoas e sair (**`q`**).

No macOS, na primeira execução, é preciso dar permissão de acesso à câmara ao terminal/IDE (Definições do Sistema → Privacidade e Segurança → Câmara).

## Galeria de identidades

A galeria é tratada como um dado biométrico e fica só no portátil.

- **Registo:** antes de registar alguém, explicar o que é guardado, para quê e como se apaga, e registar só quem concordar. Com a tecla `e`, a aplicação pede no terminal o nome e a confirmação do consentimento (`[s/N]`); sem um "s", o registo é cancelado.
- **O que é guardado:** em `gallery/gallery.npz`, o nome e um embedding ArcFace (512 valores) por imagem: 5 frames da webcam com `e`, ou uma foto por ficheiro com `src.enrol`. Nunca se guardam imagens nem vídeo. O `gallery/threshold.json` tem só o threshold calibrado.
- **Nada sai do portátil:** toda a inferência é local e nenhum frame, crop ou embedding é enviado para serviços externos. A galeria, as fotos (`data/dados_caras/`) e os resultados da identidade (`results/identity/`, que têm os nomes) não estão no git nem na entrega.
- **Remoção:** `d` apaga uma pessoa e `x` a galeria toda (com confirmação); fora da aplicação, `python -m src.enrol --clear`. No fim do trabalho, apagar também `data/dados_caras/` e `results/identity/`.
- **Uso:** demonstrador académico; não serve para controlo de acessos, presenças, vigilância ou outras decisões com consequências para as pessoas.

## Limitações conhecidas

- Os comandos têm de ser corridos a partir da raiz do repositório (`python -m src...`).
- Não há tracking (excluído pelo enunciado): os atributos são calculados em cada frame, por isso as labels podem oscilar.
- Os nomes ao registar/apagar pessoas escrevem-se no terminal; enquanto isso, a janela fica parada.
- A utilização da GPU não é medida: o painel mostra só a CPU e a memória do processo (o CoreML reparte o modelo entre CPU, GPU e Neural Engine e não mostra essa utilização por processo).
- Pipeline `separate`: idade com MAE de ~10 anos no UTKFace e tendência para a média; género pior em crianças, idosos e no grupo Asian; expressão com 54% no AffectNet (8 classes) e 64% no BTFER (7 classes), Fear a pior no BTFER.
- Multitarefa: melhor na idade (MAE 5.4–6.3) e no género (90–92%), mas avaliado no mesmo dataset em que treinou (UTKFace); no AffectNet fica 2–5 pontos abaixo do `separate` (a ResNet-50 em 3), no BTFER a ResNet-50 é a melhor. Contempt fraco a 224 (23%). Números em `results/comparison/comparison.md`.
- Condição da imagem (UTKFace degradado, `results/comparison/comparison.md`): a pior é a escuridão, com a ResNet-50 a passar de MAE 5.4 para 8.6 e de 91% para 86.5% no género (no `separate`, MAE 17 e género 74%). Com a imagem degradada há mais géneros marcados "Uncertain" (ResNet-50: 7% → 14% no escuro).
- Latência: o CoreML não suporta a ativação hard-swish da MobileNetV3 (19 nós) e parte-a em 21 blocos, ficando mais lenta do que só com a CPU (mnet160: 4.9 vs 1.4 ms); a ResNet-50 corre quase toda no CoreML (2 blocos, 1.7 ms), mas só com a CPU demora 38 ms. Medido com `eval_latency.py` (`results/system/model_latency.csv`).
- O threshold da identidade foi calibrado com só 4 pessoas e 10 casos de impostor na validação (ver MILESTONES.md).

## Considerações éticas

- **Idade** é uma estimativa de idade *aparente*, não a idade real.
- **Género** é uma classificação visual limitada às categorias do dataset de treino; não é uma determinação da identidade de género de ninguém.
- **Expressão** descreve a configuração visível da cara; não é evidência do estado emocional da pessoa.
- A **galeria de identidades** contém só quem concordou em ser registado, guarda só embeddings, fica no portátil e pode ser apagada a qualquer momento (ver [Galeria de identidades](#galeria-de-identidades)). O sistema é um demonstrador académico e não deve ser usado para controlo de acessos, presenças ou vigilância.
