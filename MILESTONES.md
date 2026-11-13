# Milestones

Entrega: **29 de novembro de 2026**.

| # | Milestone | Datas | Responsável |
|---|-----------|-------|-------------|
| M1 | Pipeline base + avaliação do detetor | 12–16 nov | Samuel |
| M2 | Idade/género e expressão | 12–21 nov | Luciana |
| M3 | Galeria de identidades | 17–22 nov | Samuel |
| M4 | Interface, desempenho e testes de sistema | 22–25 nov | Ambos |
| M5 | Relatório, vídeo e entrega | 25–29 nov | Ambos |

---

## M1 — Pipeline base + avaliação do detetor (12–16 nov)

- [x] Captura da webcam, deteção SCRFD, bounding boxes, confiança e keypoints
- [x] Alinhamento por landmarks (112×112 e 224×224)
- [ ] Ficheiro de vídeo como entrada alternativa
- [ ] Estruturas de dados entre módulos (frame, deteção, crop, embedding, atributos)
- [ ] Avaliação no WIDER Face (val): AP easy/medium/hard e tempo de inferência
- [ ] Análise qualitativa: caras pequenas, pose, oclusão, várias caras

**Feito quando:** o pipeline corre com webcam ou vídeo e os resultados do WIDER Face estão numa tabela.

## M2 — Idade/género e expressão (12–21 nov)

**Idade e género**
- [ ] Integrar o modelo pré-treinado (`genderage.onnx`) sobre o crop alinhado
- [ ] Avaliar no UTKFace: MAE, RMSE, MAE por grupo etário
- [ ] Género: precisão, recall, F1 por classe e matriz de confusão
- [ ] Mostrar a idade como intervalo e marcar previsões com pouca confiança como incertas

**Expressão**
- [ ] Fazer fine-tuning de uma CNN compacta no AffectNet
- [ ] Avaliação intra-dataset: partição de validação/teste oficial do AffectNet, sem sobreposição com o treino
- [ ] Avaliação cross-dataset: BTFER sem re-treino, só com as classes comuns
- [ ] Métricas: accuracy, balanced accuracy, precisão/recall/F1 macro, matriz de confusão
- [ ] Exportar para ONNX e mostrar a expressão mais provável com a confiança

**Feito quando:** ambos os modelos correm em tempo real no `main.py` e todas as métricas estão calculadas.

## M3 — Galeria de identidades (17–22 nov)

- [ ] Embeddings ArcFace (`w600k_r50.onnx`) normalizados a partir do crop 112×112
- [ ] Registo (várias imagens por pessoa, template = média), remoção e limpeza da galeria
- [ ] Similaridade cosseno e "Unknown" abaixo do threshold
- [ ] Calibrar o threshold num conjunto de validação separado do de teste
- [ ] Avaliar: rank-1, FAR/FRR, sensibilidade ao threshold, efeito do número de imagens de registo
- [ ] Consentimento escrito dos voluntários (< 20); galeria guardada só localmente

**Feito quando:** pessoas registadas são reconhecidas e uma pessoa não registada aparece como "Unknown".

## M4 — Interface, desempenho e testes de sistema (22–25 nov)

- [ ] Controlos: iniciar/parar câmara, escolher a fonte, ligar/desligar módulos, registar/remover identidades
- [ ] Overlay final: identidade, idade, expressão, género e confiança; previsões incertas assinaladas
- [ ] Medir FPS, latência end-to-end, CPU e memória
- [ ] Otimizar: frequência de cada modelo por frame e resolução de entrada
- [ ] Testes da secção 6.2: uma e várias pessoas, entrar/sair do campo de visão, pose/distância/luz, duas pessoas registadas em simultâneo, pessoa não registada

**Feito quando:** a aplicação corre estável no portátil com todos os módulos e os testes estão documentados.

## M5 — Relatório, vídeo e entrega (25–29 nov)

- [ ] Relatório: arquitetura, modelos e licenças, protocolo, resultados, profiling, limitações, ética
- [ ] README final: instalação, obtenção dos modelos, execução, limitações
- [ ] Vídeo de demonstração: várias caras, atributos, identidades e "Unknown"
- [ ] Verificar que a submissão não tem pesos nem dados pessoais
- [ ] **28 nov:** revisão final em conjunto; **29 nov:** entrega

---

## Extra (só se houver tempo)

Comparar o SCRFD leve (`scrfd.onnx`) com o SCRFD-10GF (`det_10g.onnx`) no WIDER Face: AP vs. tempo de inferência. O enunciado sugere esta comparação "if feasible".
