# Milestones

Entrega: **29 de novembro de 2026**.

| # | Milestone | Datas | Responsável | Estado (22 nov, noite) |
|---|-----------|-------|-------------|-----------------|
| M1 | Pipeline base + avaliação do detetor | 12–16 nov | Samuel | 🟡 Pipeline e entrada por vídeo feitos; WIDER val descarregado, falta a avaliação |
| M2 | Idade/género e expressão | 12–21 nov | Luciana | 🟡 Idade/género avaliado; expressão sem `.onnx` nem resultados no repositório |
| M3 | Galeria de identidades | 17–22 nov | Samuel | 🟢 Implementado e avaliado; falta o consentimento escrito |
| M4 | Interface, desempenho e testes de sistema | 22–25 nov | Ambos | 🟡 Interface e monitorização feitas; falta testar ao vivo, otimizar e fazer os testes 6.2 |
| M5 | Relatório, vídeo e entrega | 25–29 nov | Ambos | ⚪ Por começar |

## Situação a 22 nov

**Bloqueios e problemas em aberto**
- **`expression_best.onnx`:** só existe no computador da Luciana. Sem ele, o `main.py` arranca com a expressão desligada. Também não estão no repositório o script de exportação para ONNX nem os resultados do AffectNet e do BTFER.
- **Reorganização do `src/` (Luciana):** antes de mexer nos ficheiros dela, tem de:
  1. fazer pull;
  2. mover os pesos para `models/` na raiz;
  3. correr os scripts com `python -m src.training.train_expression` e `python -m src.evaluation.eval_expression_btfer`, a partir da raiz.

  As métricas dela passam a ir para `results/expression/`.
- **`main.py` não foi testado ao vivo.** Foi testado sem janela, com um vídeo feito de imagens do WIDER. Falta confirmar com a webcam: pausa, registo (`e`), apagar (`d`, `x`) e fechar a janela.
- **Threshold da identidade (0.193) encostado aos impostores:** o critério FAR = 0 na validação põe o threshold no score mais alto de um impostor de validação. Só houve 10 casos de impostor. Na validação e no teste, os genuínos ficaram entre 0.63 e 0.93 e os impostores entre 0.01 e 0.19, por isso qualquer valor entre 0.19 e 0.63 dava os mesmos resultados. Ao vivo, um desconhecido pode passar de 0.19. **Rever nos testes de sistema (M4)** e discutir no relatório.

**Resolvido hoje:**
- WIDER val descarregado em `data/WIDER/`
- recorte do `genderage` corrigido
- janela duplicada eliminada
- `CAP_AVFOUNDATION` usado só em macOS
- `src/` reorganizado:
  - pastas `pipeline/`, `tools/`, `evaluation/` e `training/`
  - caminhos e limiares em `src/config.py`
  - pesos em `models/`
  - comandos com `python -m` a partir da raiz (ver README)

---

## M1 — Pipeline base + avaliação do detetor (12–16 nov)

- [x] Captura da webcam, deteção SCRFD, bounding boxes, confiança e keypoints
- [x] Alinhamento por landmarks (112×112 e 224×224)
- [x] Ficheiro de vídeo como entrada alternativa (`capture.py`, `--source`)
- [x] Estruturas de dados entre módulos: `FaceAnalysis`, `IdentityResult`, `AgeGenderResult`, `ExpressionResult`
- [ ] Avaliação no WIDER Face (val): AP easy/medium/hard e tempo de inferência. Os dados estão em `data/WIDER/` e o script ainda não existe.
- [ ] Análise qualitativa: caras pequenas, pose, oclusão, várias caras

**Feito quando:** o pipeline corre com webcam ou vídeo e os resultados do WIDER Face estão numa tabela.

## M2 — Idade/género e expressão (12–21 nov)

**Idade e género** (`age_gender.py`, avaliação em `eval_age_gender.py`, resultados em `results/age_gender/`)
- [x] Integrar o modelo pré-treinado (`genderage.onnx`) no `main.py`. O recorte usado é o do InsightFace (`attribute_crop`: caixa com 1.5× o tamanho da cara, 96 px). Nas mesmas 500 imagens, o género passou de 81% para 85% de accuracy com este recorte.
- [x] Avaliar no UTKFace (23 688 imagens): MAE 9.8, RMSE 13.1, viés +4.3 anos, faixa etária certa 37%.
  - O modelo puxa as idades extremas para o meio: +14 anos nos 0–9, −15 anos nos 70+.
  - Há resultados por faixa etária, etnia e género. É um teste cross-dataset, porque o modelo não foi treinado no UTKFace.
- [x] Género: accuracy 85.8%, F1 0.86 nas duas classes, com matriz de confusão.
  - Falha mais em crianças (60%), em idosos (75%) e no grupo Asian (79%).
  - Cobertura de 95.9% com o limiar de incerteza de 0.65.
- [x] Mostrar a idade como intervalo e marcar previsões com pouca confiança como incertas (a cinzento, com "?")

**Expressão**
- [x] Fazer fine-tuning de uma CNN compacta no AffectNet (`train_expression.py`, MobileNetV3-Small, 8 classes)
- [ ] Avaliação intra-dataset na partição de teste do AffectNet. O `train_expression.py` já a faz; faltam os resultados no repositório.
- [ ] Avaliação cross-dataset no BTFER sem re-treino, só com as 7 classes comuns. O `eval_expression_btfer.py` (antigo `btfer.py`) já a faz; faltam os resultados no repositório.
- [ ] Métricas: accuracy, balanced accuracy, precisão/recall/F1 macro, matriz de confusão. Os scripts já as calculam e guardam em JSON e CSV.
- [ ] Exportar para ONNX. Falta o script de exportação e o ficheiro `.onnx`. A inferência já está em `expression.py` (`ExpressionEstimator`, incerta abaixo de 0.5) e o overlay já está pronto.

**Feito quando:** ambos os modelos correm em tempo real no `main.py` e todas as métricas estão calculadas.

## M3 — Galeria de identidades (17–22 nov)

- [x] Fotos de registo recolhidas em `data/dados_caras/`: 4 pessoas (André, Inês, João, Samuel), 7 a 9 fotos cada (frontal, lado, expressão, distância, oclusão)
- [x] Embeddings ArcFace (`w600k_r50.onnx`) normalizados a partir do crop 112×112 (`identity.py`)
- [x] Registo (várias imagens por pessoa, template = média), remoção e limpeza da galeria. Pode fazer-se com o `enrol.py` (fotos) ou com as teclas `e`, `d` e `x` no `main.py` (webcam). A galeria guarda só embeddings, em `gallery/gallery.npz`.
- [x] Similaridade cosseno e "Unknown" abaixo do threshold
- [x] Calibrar o threshold num conjunto de validação separado do de teste (`eval_identity.py`):
  - registo com `frontal` + `frontal2`; as restantes fotos divididas entre validação e teste (seed 0)
  - impostores por leave-one-out: em cada ronda uma pessoa fica fora da galeria
  - critério FAR = 0 na validação → threshold 0.193
- [x] Avaliar: rank-1, FAR/FRR, sensibilidade ao threshold, efeito do número de imagens de registo. No teste deu rank-1 100%, FRR 0% e FAR 0%, com 2 fotos e com 1 foto de registo. Os resultados estão em `results/identity/`; os gráficos fazem-se a partir dos CSV.
- [x] Galeria guardada só localmente; o registo pela webcam pede confirmação de consentimento
- [ ] Consentimento escrito dos voluntários (< 20)

**Feito quando:** pessoas registadas são reconhecidas e uma pessoa não registada aparece como "Unknown".

## M4 — Interface, desempenho e testes de sistema (22–25 nov)

- [x] Controlos por teclas no `main.py`:
  - `espaço`: parar/retomar a câmara
  - `s`: fonte seguinte
  - `1`–`4`: ligar/desligar módulos
  - `e` / `d` / `x`: registar, apagar uma pessoa, apagar a galeria
  - `h`: ajuda; `q`: sair
- [x] Overlay final:
  - identidade em destaque (verde = conhecida, vermelho = "Unknown")
  - idade, género e expressão por baixo; as previsões incertas a cinzento com "?"
  - confiança da deteção
  - caras com menos de 40 px não são analisadas
- [x] Medir FPS, latência por fase e end-to-end, CPU e memória (`monitoring.py`, painel no ecrã, `results/system/run_<data>.csv` no fim). A GPU não é medida.
- [ ] Testar a aplicação ao vivo com a webcam: pausa, registo, apagar, fechar a janela
- [ ] Otimizar: frequência de cada modelo por frame e resolução de entrada. No teste com vídeo, sem expressão: deteção 15 ms, identidade 4 ms, idade/género 1 ms, cerca de 50 FPS.
- [ ] Testes da secção 6.2: uma e várias pessoas, entrar/sair do campo de visão, pose/distância/luz, duas pessoas registadas em simultâneo, pessoa não registada
- [ ] Rever o threshold da identidade (0.193) com uma pessoa não registada ao vivo: anotar os scores que ela tem. Se passar de 0.19, decidir entre acrescentar uma margem e usar o ponto médio entre genuínos e impostores.

**Feito quando:** a aplicação corre estável no portátil com todos os módulos e os testes estão documentados.

## M5 — Relatório, vídeo e entrega (25–29 nov)

- [ ] Relatório: arquitetura, modelos e licenças, protocolo, resultados, profiling, limitações, ética
- [ ] README final: instalação, obtenção dos modelos, execução, limitações
- [ ] Vídeo de demonstração: várias caras, atributos, identidades e "Unknown"
- [ ] Verificar que a submissão não tem pesos nem dados pessoais
- [ ] **28 nov:** revisão final em conjunto; **29 nov:** entrega

---

## Extra (só se houver tempo)

Comparar o SCRFD leve (`scrfd.onnx`) com o SCRFD-10GF (`det_10g.onnx`) no WIDER Face: AP vs. tempo de inferência. O enunciado sugere esta comparação "if feasible". O `main.py` já aceita `--detector det_10g.onnx`.
