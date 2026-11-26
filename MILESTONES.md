# Milestones

Entrega: **29 de novembro de 2026**.

| # | Milestone | Responsável | Estado (26 nov, noite) |
|---|-----------|-------------|------------------------|
| M1 | Pipeline base + avaliação do detetor | Samuel | 🟢 Feito, com comparação 2.5GF vs 10GF; falta a secção do relatório |
| M2 | Idade/género e expressão | Luciana | 🟢 Avaliados e integrados; falta a idade por condição da imagem |
| M3 | Galeria de identidades | Samuel | 🟢 Avaliada; falta o consentimento escrito e fotos com outra iluminação |
| M4 | Interface, desempenho e testes de sistema | Ambos | 🟡 Interface e monitorização feitas; sessão de ~3 min com a ResNet-50 registada; faltam os testes 6.2 e `d`/`x` ao vivo |
| M5 | Relatório, vídeo e entrega | Ambos | 🔴 Por começar |
| M6 | Modelo multitarefa (extra) | Samuel | 🟢 MobileNetV3 a 112/160/224 e ResNet-50 avaliadas; ResNet-50 na app |

## Plano até à entrega

| Dia | O quê | Estado |
|---|---|---|
| **26 nov** | Testes ao vivo na aula; modelo multitarefa | Multitarefa feito (M6), ResNet-50 avaliada, escolhida para a app e testada ao vivo (22.7 FPS). Faltam os testes 6.2, desconhecido, `d`/`x` e o vídeo |
| **27 nov** | Testes ao vivo que faltem; **relatório** | |
| **28 nov** | Revisão final em conjunto; vídeo de demonstração | |
| **29 nov** | Entrega | |

## Situação a 26 nov

**Em aberto**
- **Relatório por começar** (o maior risco). As tabelas estão prontas em `results/detection/comparison.md` e `results/comparison/comparison.md`.
- **Opção de atributos da app:** decidida a 26 nov: multitarefa `r50_224_full` por omissão (`--attributes separate` para a alternativa). Falta testá-la ao vivo (ver M6).
- **Lacunas face ao enunciado (6.1):** idade por condição da imagem (o UTKFace não tem estas labels: degradar imagens ou assumir como limitação); identidade sem fotos com outra iluminação e com amostra pequena (13 genuínos e 13 impostores no teste); GPU não medida (CoreML).
- **Threshold da identidade (0.193) encostado aos impostores:** é o score mais alto dos 10 impostores de validação (FAR = 0). Genuínos entre 0.63 e 0.93, impostores entre 0.01 e 0.19: qualquer valor entre 0.19 e 0.63 dava o mesmo. Testar com um desconhecido ao vivo e discutir no relatório.
- **Ordem das classes da expressão (resolvido; explicar no relatório):** as pastas `0`–`7` do AffectNetCustom são `Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral, Contempt`, não a ordem oficial (0 = Neutral). Detetado pelo BTFER a 7.3% e pela `surpresa.jpeg` a sair "Disgust"; `CLASSES` em `expression.py` é a única fonte da ordem. Falta saber com a Luciana de onde vem esta numeração.
- **Luciana, depois do pull:** passar `expression_best.*` (os três ficheiros, juntos) para `models/expression/`.

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
- [ ] Idade por condição da imagem (ver "Em aberto")

## M3 — Galeria de identidades

- [x] 4 voluntários (`data/dados_caras/`), 7 a 9 fotos cada (frontal, lado, expressão, distância, oclusão)
- [x] ArcFace R50 (112 px alinhado), template = média dos embeddings, cosseno, "Unknown" abaixo do threshold; registo, remoção e limpeza (`enrol.py` ou teclas `e`/`d`/`x`); a galeria guarda só embeddings
- [x] `eval_identity.py`: registo com 2 fotos, validação/teste separados, impostores por leave-one-out, threshold FAR = 0 na validação (0.193). Teste: rank-1 100%, FRR 0%, FAR 0%, com 1 e com 2 fotos de registo.
- [ ] Consentimento escrito dos voluntários
- [ ] Fotos com outra iluminação

## M4 — Interface, desempenho e testes de sistema

- [x] Teclas: `espaço` câmara, `s` fonte, `1`–`4` módulos, `e`/`d`/`x` galeria, `h` ajuda, `q` sair
- [x] Overlay: identidade (verde/vermelho), idade, género e expressão, confiança da deteção; caras < 40 px não analisadas
- [x] `monitoring.py`: FPS, latência por fase, CPU e memória; CSV por sessão em `results/system/`
- [x] Sessões a 25 nov: sem expressão ~26 FPS (16–17 ms); com expressão 13–17 FPS (24–31 ms), sessão curta
- [x] Sessões com todos os módulos (`results/system/`): `separate` a 26 nov às 11h20 (4 638 frames) com 13.4 FPS e 31.1 ms por frame (deteção 14.7, expressão 10.8, idade/género 0.9, identidade 4.4); ResNet-50 a 26 nov às 22h46 (3 782 frames, ~3 min) com **22.7 FPS** e 21.1 ms (deteção 13.6, multitarefa 3.2, identidade 3.9), CPU 160% e 481 MB. A deteção passa a ser 65% do tempo. Expressões ao vivo aceitáveis, não perfeitas.
- [x] Testados ao vivo: pausa, registo (`e`), fechar a janela
- [ ] Testes 6.2: uma e várias pessoas, entrar/sair, pose/distância/luz, duas pessoas registadas, pessoa não registada
- [ ] `d` e `x` ao vivo
- Nas sessões gravadas a partir das 23h de 26 nov, a latência "total" é só a análise (antes incluía o desenho das caixas, < 1 ms)

## M5 — Relatório, vídeo e entrega

- [ ] Relatório: arquitetura, modelos e licenças, protocolo, resultados, profiling, limitações, ética
- [ ] Vídeo de demonstração: várias caras, atributos, identidades e "Unknown"
- [ ] Verificar que a submissão não tem pesos nem dados pessoais

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
- **Latência e CoreML:** a coluna ms/cara é o tempo real neste M1, não o custo do modelo. O CoreML corre a ResNet-50 quase inteira (1.8 ms; 34 ms só com CPU) e parte a MobileNetV3 em ~20 blocos (hard-swish e squeeze-excitation não suportados), que fica mais lenta do que no CPU (mnet160 4.8 vs 1.3 ms; mnet224 8.0 vs 2.2 ms). Pôr as duas colunas no relatório.
- [x] ResNet-50 no Colab (`colab/train_multitask.ipynb`), avaliada (`r50_224_full`)
- [x] Opção de atributos da app: `r50_224_full` (`ATTRIBUTE_MODELS = "multitask"`)
- [x] ResNet-50 testada ao vivo: 22.7 FPS contra 13.4 do `separate` (ver M4)
