# Milestones

Entrega: **29 de novembro de 2026**.

| # | Milestone | Responsável | Estado (28 nov) |
|---|-----------|-------------|------------------------|
| M1 | Pipeline base + avaliação do detetor | Samuel | 🟢 Feito, com comparação 2.5GF vs 10GF; falta a secção do relatório |
| M2 | Idade/género e expressão | Luciana | 🟢 Avaliados e integrados, também por condição da imagem |
| M3 | Galeria de identidades | Samuel | 🟢 Avaliada; iluminação testada ao vivo |
| M4 | Interface, desempenho e testes de sistema | Ambos | 🟢 Interface, monitorização e testes 6.2 feitos; `d` e `e` ao vivo ficam no vídeo |
| M5 | Relatório, vídeo e entrega | Ambos | 🟡 Relatório escrito em `relatorio/` (falta compilar no Overleaf e os testes de pose, entrada/saída e distância); vídeo e zip por fazer |
| M6 | Modelo multitarefa (extra) | Samuel | 🟢 MobileNetV3 a 112/160/224 e ResNet-50 avaliadas; ResNet-50 na app |

## Plano até à entrega

| Dia | O quê | Estado |
|---|---|---|
| **26 nov** | Testes ao vivo na aula; modelo multitarefa | Multitarefa feito (M6), ResNet-50 avaliada, escolhida para a app e testada ao vivo (22.7 FPS). Faltam os testes 6.2, desconhecido, `d`/`x` e o vídeo |
| **27 nov** | Testes ao vivo que faltem; **relatório** | Testes 6.2 feitos na aula |
| **28 nov** | Lacunas da avaliação; **relatório**; vídeo | Género por classe e idade por condição feitos; README com galeria, datasets e pesos |
| **29 nov** | Revisão final em conjunto; zip; entrega | |

## Situação a 28 nov

**Em aberto**
- **Relatório:** escrito em `relatorio/` (`rel.tex` + `secoes/`), com a estrutura combinada (~10 páginas). Cada tabela e figura tem a fonte num comentário `% Fonte:`; as figuras geram-se com `python -m src.graphs` (`results/images/`) e a latência dos modelos com `eval_latency.py`. A frase dos 81→85% do recorte do genderage saiu (sem resultados guardados). Falta compilar no Overleaf, rever e preencher os `% TODO` da secção 6 com os testes de pose, entrada/saída e distância (e o CSV com 1 e 3 caras).
- **Vídeo:** apagar o próprio registo (`d`) → "Unknown" → registar de novo (`e`) → nome; com uma segunda pessoa, para mostrar várias caras e uma identidade reconhecida ao lado de outra.
- **Zip da entrega** (ver M5).

**Decidido**
- **AffectNetCustom:** é o do link do enunciado (`yakhyokhuja/affectnetaligned`), já alinhado a 112 px e com a divisão feita; a ordem das pastas `0`–`7` foi identificada por nós a partir das imagens.
- **Atributos na app:** multitarefa `r50_224_full` (`--attributes separate` para a alternativa).
- **Pesos treinados por nós** (`r50_224_full` e `expression_best`) vão no zip da entrega; os do InsightFace não (README: Modelos).
- **GPU não medida:** limitação (o CoreML reparte o modelo entre CPU, GPU e Neural Engine e não mostra a utilização por processo).
- **Identidade com amostra pequena** (13 genuínos e 13 impostores no teste): limitação. A iluminação foi testada ao vivo, não nas fotos.
- **Threshold da identidade (0.193) encostado aos impostores:** é o score mais alto dos 10 impostores de validação (FAR = 0). Genuínos entre 0.63 e 0.93, impostores entre 0.01 e 0.19: qualquer valor entre 0.19 e 0.63 dava o mesmo. Ao vivo, a pessoa não registada deu "Unknown"; discutir no relatório.
- **Ordem das classes da expressão (resolvido; explicar no relatório):** as pastas `0`–`7` do AffectNetCustom são `Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt`, não a ordem oficial (0 = Neutral). Detetado pelo BTFER a 7.3% e pela `surpresa.jpeg` a sair "Disgust"; `CLASSES` em `expression.py` é a única fonte da ordem.

---

## M1 — Pipeline base + avaliação do detetor

- [x] Webcam ou vídeo (`--source`), SCRFD (caixas, confiança, 5 landmarks), alinhamento por landmarks
- [x] WIDER Face val (3 226 imagens, `eval_detection_wider.py`): SCRFD-2.5GF com AP 93.2 / 91.3 / 75.2, 30 ms por imagem; a 0.4, precisão 0.92 e recall 0.52 (46% das caras do WIDER têm < 16 px). Perto do publicado (~93.8 / 92.0 / 77.1); a diferença vem do redimensionamento *nearest* da biblioteca `scrfd`.
- [x] Robustez (recall a 0.4, labels oficiais do WIDER por cara: blur, iluminação, oclusão, pose; tamanho e nº de caras calculados das caixas). Só caras ≥ 32 px: oclusão nenhuma 96% → forte 60%, pose típica 92% → atípica 72%.
- [x] Comparação com o SCRFD-10GF (`--compare`, `results/detection/comparison.md`): +5 AP no hard e +4–8 pontos em oclusão, pose e blur, mas 5× os parâmetros e 2.4× o tempo. A app usa o 2.5GF.
- [x] Licença do `scrfd.onnx` (cospectrum/scrfd): derivado do InsightFace, só investigação não comercial, sem redistribuição dos pesos.
- [ ] Relatório: regressão das caixas, confiança, landmarks, NMS e alinhamento; figura frame → crops

## M2 — Idade/género e expressão

- [x] genderage (InsightFace, `attribute_crop` 96 px). UTKFace completo (23 688 caras, cross-dataset): MAE 9.82, viés +4.3, género 85.8%. Puxa as idades para o meio (crianças +14, 70+ −15); género pior em crianças, idosos e no grupo Asian (78%).
- [x] Expressão: MobileNetV3-Small com fine-tuning no AffectNet (`train_expression.py`), exportada para ONNX. AffectNet test 54.0% (F1 0.54), BTFER 63.7% (F1 0.63), pelo ONNX (`eval_attributes.py --model separate`).
- [x] Previsões incertas a cinzento com "?" (género < 0.65, expressão < 0.5); idade como intervalo
- [x] Expressão vista ao vivo com labels coerentes
- [x] Género por classe (precisão, recall, F1) em `results/comparison/` a partir das matrizes de confusão. ResNet-50: Male 89.3 / 93.9 / 91.6%, Female 92.9 / 87.7 / 90.3%.
- [x] Idade e género por condição da imagem (`eval_conditions.py`, teste do UTKFace degradado). A escuridão é a pior: na ResNet-50 o MAE passa de 5.4 para 8.6 e o género de 91.0% para 86.5%; no `separate` o MAE passa de 10.1 para 17.2 (viés +10) e o género para 73.7%. A desfocagem, a cara a 40 px e a oclusão de boca e queixo custam à ResNet-50 1–1.5 anos de MAE; a oclusão tira-lhe ~4 pontos no género. Os géneros "Uncertain" (< 0.65) sobem com a degradação (ResNet-50: 7% → 14% no escuro) e os restantes mantêm 91–94%: o limiar funciona.

## M3 — Galeria de identidades

- [x] 4 voluntários (`data/dados_caras/`), 7 a 9 fotos cada (frontal, lado, expressão, distância, oclusão)
- [x] ArcFace R50 (112 px alinhado), template = média dos embeddings, cosseno, "Unknown" abaixo do threshold; registo, remoção e limpeza (`enrol.py` ou teclas `e`/`d`/`x`); a galeria guarda só embeddings
- [x] `eval_identity.py`: registo com 2 fotos, validação/teste separados, impostores por leave-one-out, threshold FAR = 0 na validação (0.193). Teste: rank-1 100%, FRR 0%, FAR 0%, com 1 e com 2 fotos de registo.
- [x] Ao vivo (aula): duas pessoas registadas reconhecidas ao mesmo tempo, uma não registada como "Unknown", com mudanças de luz

## M4 — Interface, desempenho e testes de sistema

- [x] Teclas: `espaço` câmara, `s` fonte, `1`–`4` módulos, `e`/`d`/`x` galeria, `h` ajuda, `q` sair
- [x] Overlay: identidade (verde/vermelho), idade, género e expressão, confiança da deteção; caras < 40 px não analisadas
- [x] `monitoring.py`: FPS, latência por fase, CPU e memória; CSV por sessão em `results/system/`
- [x] Sessões a 25 nov: sem expressão ~26 FPS (16–17 ms); com expressão 13–17 FPS (24–31 ms), sessão curta
- [x] Sessões com todos os módulos (`results/system/`): `separate` a 26 nov às 11h20 (4 638 frames) com 13.4 FPS e 31.1 ms por frame (deteção 14.7, expressão 10.8, idade/género 0.9, identidade 4.4); ResNet-50 a 26 nov às 22h46 (3 782 frames, ~3 min) com **22.7 FPS** e 21.1 ms (deteção 13.6, multitarefa 3.2, identidade 3.9), CPU 160% e 481 MB. A deteção passa a ser 65% do tempo. Expressões ao vivo aceitáveis, não perfeitas.
- [x] Testados ao vivo: pausa, registo (`e`), fechar a janela
- [x] Testes 6.2 na aula: várias pessoas, distância e luz, duas pessoas registadas ao mesmo tempo e uma não registada ("Unknown"); tudo a funcionar
- [x] Teste em casa com 3 pessoas (1 registada): as não registadas dão "Unknown"; género, idade e expressão certos (Anger a pior). Identificação com e sem óculos. Com a cara tapada ou pouca luz a similaridade desce de ~0.8 para 0.4–0.5 (acima do threshold); sem luz não há deteção. A partir de ~2.5 m a caixa continua, mas sem identidade nem atributos: a cara fica abaixo dos 40 px (`MIN_FACE_SIZE`).
- [ ] `d` e `e` ao vivo: no vídeo
- Nas sessões gravadas a partir das 23h de 26 nov, a latência "total" é só a análise (antes incluía o desenho das caixas, < 1 ms)

## M5 — Relatório, vídeo e entrega

- [ ] Relatório: arquitetura, modelos e licenças, protocolo, resultados, profiling, limitações, ética
- [ ] Vídeo de demonstração: várias caras, atributos, identidades e "Unknown"
- [ ] Zip da entrega: o código (git), os nossos pesos e o PDF do relatório; sem `data/`, `gallery/`, `results/` nem os pesos do InsightFace:
  ```bash
  git archive --format=zip -o entrega.zip HEAD
  zip -r entrega.zip models/multitask/r50_224_full/{multitask.onnx,multitask.onnx.data,config.json} \
      models/expression/expression_best.onnx{,.data}
  ```

## M6 — Modelo multitarefa (extra)

Um backbone do ImageNet partilhado pela expressão, género e idade, com uma cabeça e uma loss por tarefa (CrossEntropy, CrossEntropy, L1), treinado no AffectNet (expressão) + 70% do UTKFace (idade, género) com losses mascaradas. A identidade continua no ArcFace. Uma inferência por cara em vez de duas.

- [x] `prepare_multitask.py`, `multitask_model.py`, `train_multitask.py` (MPS ou CUDA), `pipeline/multitask.py`, `--attributes multitask` na app
- [x] Crop da app = quadrado da caixa do SCRFD: é o enquadramento das imagens do AffectNet e do UTKFace (medido), sem bordas artificiais
- [x] Avaliação de todas as opções com o mesmo código (`eval_attributes.py`) e tabelas em `results/comparison/` (`compare_models.py`)

| Opção | AffectNet | BTFER | Idade MAE | Género | ms/cara |
|---|---|---|---|---|---|
| separate | **54.0%** | 63.7% | 10.06 | 85.8% | 11.7 |
| mnet112 | 49.5% | 55.8% | 6.31 | 89.8% | 5.0 |
| mnet160 | 52.3% | 61.0% | 6.03 | 90.6% | 6.7 |
| mnet224 | 50.3% | 63.5% | 5.59 | **92.0%** | 9.3 |
| r50_224_full (**na app**) | 51.1% | **67.8%** | **5.44** | 91.0% | **2.9** |

Idade e género no teste do UTKFace (3 554 caras): os multitarefa treinaram no mesmo dataset e o genderage não, o que tem de ser dito no relatório.

- **Resolução:** a expressão melhora de 112 para 160; a 224 o BTFER (256 px nativos) sobe mas o AffectNet (112 px nativos) desce, sobretudo no Contempt (recall 23%).
- **Paragem:** a melhor época é escolhida pela soma das losses de validação, dominada pela expressão. No `mnet224` o F1 da expressão continuou a subir depois da época escolhida (a loss subiu por excesso de confiança). Critério mantido; discutir no relatório.
- **ResNet-50** (`r50_224_full`, Colab, sem camadas congeladas): melhor na idade e no BTFER, empata no género, 3 pontos abaixo do `separate` no AffectNet. Sobreajusta cedo: melhor época 4 de 8 (loss de treino 1.65 → 0.99, validação a subir); 23.5 M parâmetros para ~54 mil imagens.
- **Latência e CoreML:** a coluna ms/cara é o tempo real neste M1, não o custo do modelo. O CoreML corre a ResNet-50 quase inteira (2 blocos, 1.7 ms; 38 ms só com CPU) e parte a MobileNetV3 em 21 blocos (as 19 hard-swish não são suportadas), que fica mais lenta do que só com a CPU (mnet160 4.9 vs 1.4 ms; mnet224 7.8 vs 2.2 ms). Medido com `eval_latency.py` (`results/system/model_latency.csv`), mediana de 5 medições; entre execuções variou até 2× no CoreML da MobileNetV3.
- [x] ResNet-50 no Colab (`colab/train_multitask.ipynb`), avaliada (`r50_224_full`)
- [x] Opção de atributos da app: `r50_224_full` (`ATTRIBUTE_MODELS = "multitask"`)
- [x] ResNet-50 testada ao vivo: 22.7 FPS contra 13.4 do `separate` (ver M4)
