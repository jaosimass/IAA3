# Relatório automático da CNN (fire / nofire)

Gerado por `src/cnn_analise.py` a partir dos resultados desta execução. Todos os números abaixo foram calculados, nenhum foi digitado à mão.

- Imagem de entrada: 128x128 RGB; parâmetros: 102321; épocas executadas: 25 (melhor época: 17); semente: 42; aumento de dados: True; TensorFlow: 2.21.0.

## Métricas no conjunto de teste

Limiar de decisão: 0.5; classe positiva: `fire`; amostras: 284.

| accuracy | precision | recall | f1_score | taxa_falsos_positivos | taxa_falsos_negativos |
|---|---|---|---|---|---|
| 0.9859 | 0.9859 | 0.9859 | 0.9859 | 0.0141 | 0.0141 |

Matriz de confusão (linhas = real, colunas = previsto; ordem: nofire, fire):

|  | previsto nofire | previsto fire |
|---|---|---|
| real nofire | 140 | 2 |
| real fire | 2 | 140 |

Falsos positivos (alarmes falsos): **2**. Falsos negativos (incêndios não detectados): **2**.

Tempo de treino: 109.1 s; inferência: 0.74 ms/imagem (CPU/GPU da máquina que executou).

## Generalização (treino x validação x teste)

| conjunto | n_amostras | accuracy | precision | recall | f1_score | log_loss |
|---|---|---|---|---|---|---|
| treino | 1330 | 0.9782 | 0.9789 | 0.9774 | 0.9782 | 0.0597 |
| validacao | 286 | 0.9860 | 0.9929 | 0.9790 | 0.9859 | 0.0573 |
| teste | 284 | 0.9859 | 0.9859 | 0.9859 | 0.9859 | 0.0406 |

- Lacuna de accuracy treino - teste: -0.0077; treino - validação: -0.0078.
- Diagnóstico (limiares heuristicos {'lacuna_pequena': 0.03, 'lacuna_alta': 0.08, 'acc_treino_baixa': 0.8}): Lacuna treino-teste pequena: sem sinal relevante de sobreajuste nesta execução.

- `log_loss` aqui não inclui a regularização L2, por isso difere da `loss` exibida pelo Keras.

## Exemplos (imagens reais do teste)

Quantidade de exemplos mostrados por tipo: {'VP': 8, 'VN': 8, 'FP': 2, 'FN': 2}. Figuras: `exemplos_acertos.png` e `exemplos_erros.png` (os erros mostrados são os de maior confiança).

## Desempenho por faixa de luminosidade (verificação simples de viés)

| faixa_luminosidade | n_imagens | n_fire | n_nofire | accuracy | falsos_positivos | falsos_negativos | taxa_falsos_positivos | taxa_falsos_negativos |
|---|---|---|---|---|---|---|---|---|
| 11.7 a 78.7 | 95 | 66 | 29 | 0.9895 | 1 | 0 | 0.0345 | 0.0000 |
| 78.7 a 105.9 | 94 | 43 | 51 | 0.9787 | 1 | 1 | 0.0196 | 0.0233 |
| 105.9 a 194.5 | 95 | 33 | 62 | 0.9895 | 0 | 1 | 0.0000 | 0.0303 |

Faixas com poucas imagens geram taxas instáveis; use como indício, não como prova.

## Arquivos

`matriz_confusao.png`, `historico_treinamento.png`, `analise_generalizacao.png`, `exemplos_acertos.png`, `exemplos_erros.png`, `metricas_cnn.json/.csv`, `predicoes_todos_conjuntos.csv`, `divisao_dados.csv`.
