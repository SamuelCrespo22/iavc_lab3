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

O estado detalhado está em [MILESTONES.md](MILESTONES.md). Em resumo (26 nov):
- **Feito:** deteção, alinhamento, identidade, idade/género e expressão, todos avaliados; interface e monitorização; modelo multitarefa (MobileNetV3 a 112/160/224 e ResNet-50 a 224) treinado e avaliado, com a ResNet-50 na app; tabelas de comparação em `results/detection/comparison.md` e `results/comparison/comparison.md`.
- **Falta:** testes de sistema ao vivo, consentimento escrito, relatório e vídeo.

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
│   │   ├── compare_models.py         # tabelas de todas as opções de atributos e da identidade
│   │   └── tables.py                 # CSV, Markdown e tamanho dos modelos
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
├── assignment.pdf               # Enunciado
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
| `scikit-learn` | Métricas das avaliações |
| `torch`, `torchvision`, `onnx`, `onnxscript` | Só para treinar (expressão e multitarefa) e exportar para ONNX |

## Modelos

Os pesos **não estão incluídos no repositório** (por tamanho e por termos de licença). Devem ser colocados em `models/`, na raiz do repositório:

| Ficheiro | Modelo | Uso |
|----------|--------|-----|
| `scrfd.onnx` | SCRFD-2.5GF com keypoints (0.82 M parâmetros; confirmado pelo AP no WIDER) | Deteção de caras (por omissão) |
| `det_10g.onnx` | SCRFD-10GF | Deteção (alternativa mais precisa, `--detector det_10g.onnx`) |
| `genderage.onnx` | InsightFace gender-age | Idade e género |
| `w600k_r50.onnx` | ArcFace ResNet-50 (WebFace600K) | Embeddings de identidade |
| `2d106det.onnx`, `1k3d68.onnx` | Landmarks 2D/3D | Não usados |
| `expression/expression_best.pth` | MobileNetV3-Small com fine-tuning no AffectNet | Checkpoint gerado por `src/training/train_expression.py` |
| `expression/expression_best.onnx` + `.onnx.data` | O mesmo modelo exportado para ONNX (o `.data` tem os pesos; são precisos os dois, na mesma pasta) | Expressão na aplicação (`separate`). Gerados por `src/training/train_expression.py`. |
| `multitask/<corrida>/` | Modelo multitarefa: `multitask.onnx` (+ `.data`), `multitask.pth`, `config.json`, `train.log` | Idade, género e expressão (`--attributes multitask`). Gerados por `src/training/train_multitask.py` |

Os índices de saída do modelo de expressão seguem a ordem das pastas `0`–`7` do AffectNet usado no treino, que não é a ordem oficial do AffectNet. A ordem está em `CLASSES`, em `src/pipeline/expression.py`.

O `scrfd.onnx` vem da pasta `models/` do repositório [cospectrum/scrfd](https://github.com/cospectrum/scrfd/tree/main/models), que é o mesmo da biblioteca `scrfd`.

Os ficheiros `det_10g.onnx`, `genderage.onnx`, `w600k_r50.onnx`, `2d106det.onnx` e `1k3d68.onnx` vêm do pack **buffalo_l** do [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package):

```bash
mkdir -p models
curl -L -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o /tmp/buffalo_l.zip -d models
```

> **Licença:** os modelos pré-treinados do InsightFace estão disponíveis apenas para **investigação não comercial**. O `scrfd.onnx` do cospectrum/scrfd é um derivado do InsightFace: esse repositório aplica a licença Apache 2.0 só ao código e diz que o modelo é *"used for testing purposes only"*. Por isso segue as mesmas condições. Não redistribuir nenhum dos pesos.

## Datasets

Nenhum dataset é versionado. As localizações estão em `src/config.py`:

| Dataset | Pasta | Uso |
|---|---|---|
| UTKFace | `data/UTKFace/` | Idade e género: treino do multitarefa (70%) e avaliação (15% de teste). Labels no nome: `[idade]_[género]_[etnia]_[data].jpg.chip.jpg` |
| WIDER Face (val) | `data/WIDER/` (`WIDER_val/`, `wider_face_split/`, `eval_tools/`) | Avaliação do detetor |
| Fotos da galeria | `data/dados_caras/<pessoa>/` | Avaliação da identidade (voluntários com consentimento; nunca sair do portátil) |
| AffectNet | `data/AffectNetCustom/` (`train/`, `val/`, `test/`) | Treino e avaliação intra-dataset da expressão |
| BTFER | `data/BTFER/` | Avaliação cross-dataset da expressão |
| Divisão do multitarefa | `data/multitask/labels.csv` | Gerado por `prepare_multitask.py` (AffectNet + UTKFace, com o split de cada imagem) |

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
python -m src.evaluation.compare_models             # todas as opções -> results/comparison/comparison.md

python -m src.training.train_expression             # models/expression/: .pth e .onnx (+ .onnx.data), verificado

python -m src.training.prepare_multitask            # uma vez: data/multitask/labels.csv
python -m src.training.train_multitask --run mnet160 --input-size 160                          # MobileNetV3
python -m src.training.train_multitask --run r50_224_full --arch resnet50 --input-size 224   # ResNet-50 (na app)
```

As ResNet treinam-se numa GPU do Colab com [colab/train_multitask.ipynb](colab/train_multitask.ipynb) (instruções no próprio notebook); a pasta da corrida volta depois para `models/multitask/`.

Organização dos resultados (`results/`, não versionado): `detection/<detetor>/`, `identity/`, `attributes/<opção>/` (uma pasta por opção de atributos, todas com o mesmo formato), `comparison/` (tabelas para o relatório) e `system/` (sessões da aplicação).

Na aplicação, carregar em **`h`** para ver as teclas: parar/retomar a câmara, mudar de fonte, ligar/desligar módulos, registar/apagar pessoas e sair (**`q`**).

No macOS, na primeira execução, é preciso dar permissão de acesso à câmara ao terminal/IDE (Definições do Sistema → Privacidade e Segurança → Câmara).

## Limitações conhecidas

- Os comandos têm de ser corridos a partir da raiz do repositório (`python -m src...`).
- Não há tracking (excluído pelo enunciado): os atributos são calculados em cada frame, por isso as labels podem oscilar.
- Os nomes ao registar/apagar pessoas escrevem-se no terminal; enquanto isso, a janela fica parada.
- A utilização da GPU não é medida; o painel mostra só a CPU e a memória do processo.
- Pipeline `separate`: idade com MAE de ~10 anos no UTKFace e tendência para a média; género pior em crianças, idosos e no grupo Asian; expressão com 54% no AffectNet (8 classes) e 64% no BTFER (7 classes), Fear a pior no BTFER.
- Multitarefa: melhor na idade (MAE 5.4–6.3) e no género (90–92%), mas avaliado no mesmo dataset em que treinou (UTKFace); no AffectNet fica 2–5 pontos abaixo do `separate` (a ResNet-50 em 3), no BTFER a ResNet-50 é a melhor. Contempt fraco a 224 (23%). Números em `results/comparison/comparison.md`.
- Latência: o CoreML não suporta parte das operações da MobileNetV3 (hard-swish, squeeze-excitation) e parte-a em ~20 blocos, ficando mais lenta do que no CPU (mnet160: 4.8 vs 1.3 ms); a ResNet-50 corre quase toda no CoreML (1.8 ms) mas seria ~15× mais lenta só com CPU (34 ms).
- O threshold da identidade foi calibrado com só 4 pessoas e 10 casos de impostor na validação (ver MILESTONES.md).

## Considerações éticas

- **Idade** é uma estimativa de idade *aparente*, não a idade real.
- **Género** é uma classificação visual limitada às categorias do dataset de treino; não é uma determinação da identidade de género de ninguém.
- **Expressão** descreve a configuração visível da cara; não é evidência do estado emocional da pessoa.
- A **galeria de identidades** deve conter apenas voluntários que deram consentimento, guardar só os dados necessários, ter mecanismo de remoção e nunca ser incluída no repositório (a pasta `gallery/` está no `.gitignore`). O sistema é um demonstrador académico e não deve ser usado para controlo de acessos, presenças ou vigilância.
