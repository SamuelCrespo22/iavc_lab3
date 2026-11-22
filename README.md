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

O estado detalhado de cada fase está em [MILESTONES.md](MILESTONES.md). Em resumo:
- **Feito:** deteção, alinhamento, identidade (avaliada), idade/género (avaliado no UTKFace), interface e monitorização.
- **Falta:** avaliação do detetor no WIDER Face, modelo ONNX e resultados da expressão, testes de sistema ao vivo, relatório.

## Estrutura do repositório

```
.
├── src/
│   ├── main.py                  # Aplicação: webcam/vídeo, análise, overlay, teclas
│   ├── config.py                # Caminhos, modelos e limiares
│   ├── pipeline/                # Módulos da aplicação
│   │   ├── capture.py           #   webcam ou ficheiro de vídeo
│   │   ├── detection.py         #   SCRFD e recortes para cada modelo
│   │   ├── alignment.py         #   alinhamento por landmarks (template ArcFace)
│   │   ├── identity.py          #   embeddings ArcFace e galeria
│   │   ├── age_gender.py        #   idade e género (InsightFace genderage)
│   │   ├── expression.py        #   expressão facial (MobileNetV3, AffectNet)
│   │   └── monitoring.py        #   FPS, latência, CPU e memória
│   ├── tools/enrol.py           # Gerir a galeria a partir de fotos
│   ├── evaluation/              # Avaliação de cada componente
│   │   ├── eval_identity.py
│   │   ├── eval_age_gender.py
│   │   └── eval_expression_btfer.py
│   └── training/
│       └── train_expression.py  # Fine-tuning da expressão no AffectNet
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
| `torch`, `torchvision`, `onnx`, `onnxscript` | Só para treinar o modelo de expressão e exportá-lo para ONNX |

## Modelos

Os pesos **não estão incluídos no repositório** (por tamanho e por termos de licença). Devem ser colocados em `models/`, na raiz do repositório:

| Ficheiro | Modelo | Uso |
|----------|--------|-----|
| `scrfd.onnx` | SCRFD (variante leve, com keypoints) | Deteção de caras (por omissão) |
| `det_10g.onnx` | SCRFD-10GF | Deteção (alternativa mais precisa, `--detector det_10g.onnx`) |
| `genderage.onnx` | InsightFace gender-age | Idade e género |
| `w600k_r50.onnx` | ArcFace ResNet-50 (WebFace600K) | Embeddings de identidade |
| `2d106det.onnx`, `1k3d68.onnx` | Landmarks 2D/3D | Não usados |
| `expression_best.pth` | MobileNetV3-Small com fine-tuning no AffectNet | Checkpoint gerado por `src/training/train_expression.py` |
| `expression_best.onnx` | O mesmo modelo exportado para ONNX | Expressão na aplicação (script de exportação por adicionar). Sem ele, a expressão fica desligada. |

Os ficheiros `det_10g.onnx`, `genderage.onnx`, `w600k_r50.onnx`, `2d106det.onnx` e `1k3d68.onnx` vêm do pack **buffalo_l** do [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package):

```bash
mkdir -p models
curl -L -o /tmp/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o /tmp/buffalo_l.zip -d models
```

> **Licença:** os modelos pré-treinados do InsightFace estão disponíveis apenas para **investigação não comercial**. Não redistribuir os pesos.

## Datasets

Nenhum dataset é versionado. As localizações estão em `src/config.py`:

| Dataset | Pasta | Uso |
|---|---|---|
| UTKFace | `data/UTKFace/` | Avaliação de idade e género. Labels no nome: `[idade]_[género]_[etnia]_[data].jpg.chip.jpg` |
| WIDER Face (val) | `data/WIDER/` (`WIDER_val/`, `wider_face_split/`, `eval_tools/`) | Avaliação do detetor |
| Fotos da galeria | `data/dados_caras/<pessoa>/` | Avaliação da identidade (voluntários com consentimento) |
| AffectNet | `AffectNetCustom/` (`train/`, `val/`, `test/`) | Treino e avaliação intra-dataset da expressão |
| BTFER | `BTFER/` | Avaliação cross-dataset da expressão |

## Execução

Todos os comandos correm **a partir da raiz do repositório**, com `python -m`:

```bash
source .venv/bin/activate
python -m src.main                                  # aplicação (webcam 0)
python -m src.main --source 0 --source video.mp4    # 's' alterna entre as fontes

python -m src.tools.enrol --all data/dados_caras --images frontal frontal2   # registar a galeria
python -m src.tools.enrol --list | --remove NOME | --clear

python -m src.evaluation.eval_identity              # resultados em results/identity/
python -m src.evaluation.eval_age_gender            # resultados em results/age_gender/
python -m src.evaluation.eval_expression_btfer      # resultados em results/expression/
python -m src.training.train_expression             # checkpoint em models/, métricas em results/expression/
```

Na aplicação, carregar em **`h`** para ver as teclas: parar/retomar a câmara, mudar de fonte, ligar/desligar módulos, registar/apagar pessoas e sair (**`q`**).

No macOS, na primeira execução, é preciso dar permissão de acesso à câmara ao terminal/IDE (Definições do Sistema → Privacidade e Segurança → Câmara).

## Limitações conhecidas

- Os comandos têm de ser corridos a partir da raiz do repositório (`python -m src...`).
- Não há tracking (excluído pelo enunciado): os atributos são calculados em cada frame, por isso as labels podem oscilar.
- Os nomes ao registar/apagar pessoas escrevem-se no terminal; enquanto isso, a janela fica parada.
- A utilização da GPU não é medida; o painel mostra só a CPU e a memória do processo.
- Idade: MAE de cerca de 10 anos no UTKFace, com tendência para a média (sobrestima crianças, subestima idosos).
- Género: falha mais em crianças, idosos e no grupo Asian do UTKFace.
- O threshold da identidade foi calibrado com só 4 pessoas e 10 casos de impostor na validação (ver MILESTONES.md).

## Considerações éticas

- **Idade** é uma estimativa de idade *aparente*, não a idade real.
- **Género** é uma classificação visual limitada às categorias do dataset de treino; não é uma determinação da identidade de género de ninguém.
- **Expressão** descreve a configuração visível da cara; não é evidência do estado emocional da pessoa.
- A **galeria de identidades** deve conter apenas voluntários que deram consentimento, guardar só os dados necessários, ter mecanismo de remoção e nunca ser incluída no repositório (a pasta `gallery/` está no `.gitignore`). O sistema é um demonstrador académico e não deve ser usado para controlo de acessos, presenças ou vigilância.
