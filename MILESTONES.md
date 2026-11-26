# Milestones

Entrega: **29 de novembro de 2026**.

| # | Milestone | Datas | Responsável | Estado (26 nov) |
|---|-----------|-------|-------------|-----------------|
| M1 | Pipeline base + avaliação do detetor | 12–16 nov | Samuel | 🟢 Código, avaliação no WIDER (com a comparação extra) e licenças verificados; falta a secção do relatório |
| M2 | Idade/género e expressão | 12–21 nov | Luciana | 🟢 Ambos avaliados, integrados e testados ao vivo; falta a idade por condição da imagem |
| M3 | Galeria de identidades | 17–22 nov | Samuel | 🟢 Implementado e avaliado; falta o consentimento escrito e fotos com outra iluminação |
| M4 | Interface, desempenho e testes de sistema | 22–25 nov | Ambos | 🟡 Interface e monitorização feitas, app testada com a webcam com todos os módulos; falta testar `d`/`x`, uma sessão longa com tudo ligado, otimizar e fazer os testes 6.2 |
| M5 | Relatório, vídeo e entrega | 25–29 nov | Ambos | ⚪ Por começar |

## Plano até à entrega

| Dia | O quê | Estado |
|---|---|---|
| **24 nov** | Escrever o relatório que der (deteção, idade/género, identidade, arquitetura) | ⚠️ Não feito: relatório por começar |
| **25 nov** | Passou de 23 nov. Receber da Luciana tudo o que falta da expressão (`.onnx`, `.pth`, script de exportação, métricas do AffectNet e do BTFER) e integrar. Testar a aplicação com a webcam. Script de gráficos, por partes. Revisão completa do código. | Expressão recebida, corrigida, integrada e testada ao vivo; app corrida com a webcam. Revisão do código feita em parte a 26 nov. Falta: gráficos |
| **26 nov** | Testes ao vivo na aula: testes 6.2, threshold da identidade com pessoas não registadas, sessão longa com tudo ligado, fotos com outra iluminação, gravar o vídeo de demonstração | Hoje |
| **28 nov** | Revisão final em conjunto | |
| **29 nov** | Entrega | |

## Situação a 26 nov

**Bloqueios e problemas em aberto**
- **Lacunas face às métricas pedidas no enunciado (6.1):**
  - **idade por condição da imagem:** o UTKFace não tem estas labels. Ou se degrada um subconjunto (blur, resolução, brilho) e se mede o MAE, ou fica como limitação no relatório;
  - **identidade com outra iluminação:** nenhuma das fotos atuais varia a luz. A amostra também é pequena (13 genuínos e 13 impostores no teste);
  - **sistema:** a única sessão com os 4 módulos tem 69 frames. Falta uma sessão longa. A GPU não é medida (CoreML no macOS), o que tem de ser explicado no relatório.
- **Ordem das classes da expressão (resolvido; a Luciana tem de saber e o relatório tem de o explicar):** as pastas `0`–`7` do AffectNet usado no treino seguem a ordem `Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt`, não a ordem oficial do AffectNet (0 = Neutral), que era a que o código assumia.
  - **Efeitos:** a app mostrava a classe errada (uma cara neutra aparecia como "Anger") e o BTFER dava accuracy 7.3%, abaixo do acaso.
  - **Como se confirmou:**
    - na matriz do BTFER, cada classe ia quase toda para uma única coluna;
    - nas fotos da galeria, a `surpresa.jpeg` saía como "Disgust".
  - **Correção:** `CLASSES` em `expression.py` passa a ser a única fonte da ordem. O BTFER foi recalculado trocando as linhas da matriz, porque as previsões não mudam.
  - **Falta confirmar com a Luciana:** de onde vem esta numeração das pastas.
- **Antes de mexer nos ficheiros da expressão, a Luciana tem de fazer pull** e correr os scripts a partir da raiz, com `python -m src.…`. Os pesos ficam em `models/` e as métricas em `results/expression/`.
- **Teclas por testar ao vivo:** falta apagar uma pessoa (`d`) e apagar a galeria (`x`).
- **Threshold da identidade (0.193) encostado aos impostores:** o critério FAR = 0 na validação põe o threshold no score mais alto de um impostor de validação. Só houve 10 casos de impostor. Na validação e no teste, os genuínos ficaram entre 0.63 e 0.93 e os impostores entre 0.01 e 0.19, por isso qualquer valor entre 0.19 e 0.63 dava os mesmos resultados. Ao vivo, um desconhecido pode passar de 0.19. **Rever nos testes de sistema (M4)** e discutir no relatório.

---

## M1 — Pipeline base + avaliação do detetor (12–16 nov)

- [x] Captura da webcam, deteção SCRFD, bounding boxes, confiança e keypoints
- [x] Alinhamento por landmarks (112×112 e 224×224)
- [x] Ficheiro de vídeo como entrada alternativa (`capture.py`, `--source`)
- [x] Estruturas de dados entre módulos: `FaceAnalysis`, `IdentityResult`, `AgeGenderResult`, `ExpressionResult`
- [x] Avaliação no WIDER Face (val, 3 226 imagens) com `eval_detection_wider.py`. Resultados em `results/detection/`.
  - **SCRFD-2.5GF (`scrfd.onnx`):** AP 93.2 / 91.3 / 75.2 (easy/medium/hard), 30 ms por imagem.
  - Com o limiar da aplicação (0.4): precisão 0.92 e recall 0.52. O recall é baixo porque 46% das caras do WIDER têm menos de 16 px.
  - Os valores estão próximos dos publicados pelo InsightFace para o `scrfd_2.5g_bnkps` (~93.8 / 92.0 / 77.1). A pequena diferença deve-se ao redimensionamento *nearest* da biblioteca `scrfd`, a explicar no relatório. Isto confirma a variante do modelo e o avaliador.
- [x] Robustez (recall a 0.4, em `robustness.csv`), também calculada só com caras ≥ 32 px para isolar o efeito do tamanho:
  - **tamanho:** 18% (<16 px) → 69% (16–32) → 88% (32–64) → 96% (≥128)
  - **oclusão** (caras ≥ 32 px): nenhuma 96% → parcial 83% → forte 60%
  - **pose** (≥ 32 px): típica 92%, atípica 72%
  - **várias caras:** o recall desce de 93% (1 cara) para 41% (>20 caras), mas com caras ≥ 32 px fica em 94% → 89%. As multidões são difíceis sobretudo por terem caras pequenas.
  - 6 imagens de exemplo em `results/detection/scrfd/examples/`
- [x] **Origem e licença do `scrfd.onnx`.**
  - Vem de [cospectrum/scrfd](https://github.com/cospectrum/scrfd/tree/main/models). O código desse repositório é Apache 2.0, mas segundo o README dele *"the license applied only to code. The models are a derivative work of InsightFace and are used for testing purposes only"*.
  - Por isso segue as condições do InsightFace, tal como os outros modelos: uso académico/investigação não comercial é permitido, mas os pesos **não podem ser redistribuídos**. Não entram no repositório nem na entrega; o README indica onde descarregá-los.
- [ ] **Relatório, secção de deteção.** O enunciado pede que se perceba e explique:
  - a regressão das caixas, os scores de confiança, os landmarks e o NMS;
  - o papel do alinhamento.

  Inclui uma figura frame → recortes (112 px ArcFace, 224 px expressão, 96 px idade/género), como a da secção 4.1 do enunciado.
- [ ] Funcionamento estável no portátil (fase 1 do enunciado): coberto pelo teste ao vivo do M4

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
- [ ] Idade por condição da imagem (ver "Situação a 26 nov")

**Expressão**
- [x] Fazer fine-tuning de uma CNN compacta no AffectNet (`train_expression.py`, MobileNetV3-Small, 8 classes)
- [x] Métricas: accuracy, balanced accuracy, precisão/recall/F1 macro e matriz de confusão, em `results/expression/`
- [x] Avaliação intra-dataset no teste do AffectNet (8 classes, equilibrado): accuracy 53.8%, F1 macro 0.54.
  - Recall por classe: Happy 0.69 e Contempt 0.61 são as melhores; Surprise 0.42, Anger 0.48 e Fear 0.49 as piores.
- [x] Avaliação cross-dataset no BTFER (2 100 imagens, 7 classes comuns, sem re-treino): accuracy 63.7%, F1 macro 0.63.
  - Happy fica quase toda certa (recall 0.96); Fear é a pior (0.39), confundida com Sad e Surprise.
  - Dá mais do que no AffectNet: há menos uma classe e o BTFER deve ter expressões mais marcadas (a confirmar no relatório).
- [x] Exportar para ONNX: `src/tools/export_expression_onnx.py` gera `expression_best.onnx` + `.onnx.data` e compara com o PyTorch (diferença 1e-5). Inferência em `expression.py` (`ExpressionEstimator`, incerta abaixo de 0.5).
- [x] Corrigir a ordem das classes (ver "Situação a 26 nov"). Nas fotos da galeria, a expressão sai agora coerente: a `surpresa.jpeg` dá Surprise (0.89) e as fotos frontais dão sobretudo Neutral ou Happy.
- [x] Ver a expressão ao vivo no `main.py` e medir a latência dela (25 nov): labels coerentes, 8–11 ms por frame
- [x] O modelo de expressão passou a ser obrigatório (26 nov): sem o `.onnx` a aplicação não arranca, como acontece com os outros modelos

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
- [ ] Fotos com outra iluminação, para avaliar o efeito da luz como pede o enunciado

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
- [x] Correr a aplicação com a webcam: 2 sessões a 25 nov, sem expressão. Média de 26 FPS (provavelmente o limite da câmara), 16–17 ms end-to-end (deteção 12–13 ms), CPU de 160–170% e até 543 MB de memória.
  - Mais 2 sessões a 25 nov com a expressão ligada: 13–17 FPS, 24–31 ms end-to-end (expressão 8–11 ms), CPU de 113–124% e até 850 MB. A sessão com todos os módulos tem só 69 frames.
- [ ] Sessão longa (vários minutos) com os 4 módulos ligados, para os números finais do relatório
- [x] Testar ao vivo a pausa, o registo (`e`) e o fechar da janela (25 nov)
- [ ] Testar ao vivo o apagar de uma pessoa (`d`) e da galeria (`x`)
- [ ] Otimizar: frequência de cada modelo por frame e resolução de entrada. No teste com vídeo, sem expressão: deteção 15 ms, identidade 4 ms, idade/género 1 ms, cerca de 50 FPS. Com a expressão ao vivo o FPS desce para 13–17; a primeira hipótese é correr a expressão e a idade/género de N em N frames.
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

## Extra

- [x] Comparar o SCRFD-2.5GF com o SCRFD-10GF (`det_10g.onnx`) no WIDER Face (`results/detection/comparison.csv`):

  | Detetor | AP easy | AP medium | AP hard | ms/imagem (média / p95) | Recall a 0.4 |
  |---|---|---|---|---|---|
  | SCRFD-2.5GF | 93.2 | 91.3 | 75.2 | 30 / 74 | 0.52 |
  | SCRFD-10GF | 94.7 | 93.2 | 80.5 | 73 / 106 | 0.57 |

  O 10GF ganha +5 pontos de AP no hard (caras pequenas), mas é 2.4× mais lento. Por isso a aplicação usa o 2.5GF por omissão (`--detector det_10g.onnx` para trocar). O 10GF tem entrada de tamanho variável, o que provavelmente impede o CoreML de o acelerar; isto ajuda a explicar a diferença de tempo.
