# Plano experimental da CNN (fire / nofire)

**Hipótese:** uma CNN pequena treinada do zero em 1900 imagens RGB classifica `fire` x `nofire` em imagens
novas com boa generalização; o aumento de dados reduz a lacuna entre treino e teste.

| Item | Definição |
|---|---|
| Dados | `data/raw/g20gli_dataset` (950 fire + 950 nofire), sem dados externos |
| Divisão | 70/15/15 estratificada, agrupada por quase-duplicatas, semente 42 (`results/cnn/divisao_dados.csv`) |
| Entrada | 128x128 RGB, valores em [0,1] |
| Modelo | CNN de 4 blocos (ver `CNN_arquitetura_e_limitacoes.md`) |
| Treino | Adam 1e-3, lote 32, até 40 épocas, EarlyStopping (paciência 8) por `val_loss` |
| Métricas | accuracy, precision, recall, F1 (classe positiva `fire`), matriz de confusão, taxas de FP e FN, tempo de treino e de inferência |
| Generalização | treino x validação x teste (`analise_generalizacao.json`) |
| Análise qualitativa | exemplos de acertos e de erros; erros por faixa de luminosidade |

**Experimentos:**
1. **E1, linha de base:** `python src\cnn_treino.py`.
2. **E2, ablação sem aumento de dados (opcional):** `python src\cnn_treino.py --sem-aumento --modelo-saida models\cnn_sem_aumento.keras --resultados-dir results\cnn_sem_aumento`; comparar a lacuna treino-teste de E1 e E2.
3. **E3, sensibilidade ao limiar (opcional, sem retreinar):** `python src\cnn_analise.py --limiar 0.3 --saida-dir results\cnn_limiar_0.3` reaplica o limiar às probabilidades de `predicoes_todos_conjuntos.csv` e grava, na pasta de saída, `relatorio_cnn.md` (com as métricas e a matriz de confusão do teste em tabelas), `exemplos_acertos.png`, `exemplos_erros.png`, `exemplos_selecionados.csv`, `analise_generalizacao.json/.png` e `analise_luminosidade.csv` (limiar menor = mais recall e mais falsos alarmes). Esse comando **não** grava `metricas_cnn.json/.csv`, `matriz_confusao.png`, `relatorio_classificacao.txt` nem `predicoes_*.csv`: esses arquivos só são gerados pelo treino. Use sempre `--saida-dir` com uma pasta nova. Sem ele, os arquivos de análise listados acima são sobrescritos em `results\cnn` com o novo limiar, enquanto `metricas_cnn.json` e `matriz_confusao.png` continuam com o limiar do treino (0,5), e a pasta fica inconsistente.
4. **E4, tamanho da imagem (opcional):** `--tamanho 96` x `--tamanho 128`, comparando métricas e tempo.

Critérios de leitura: o teste é usado uma única vez por execução; não se escolhe hiperparâmetro olhando o teste.
Para comparação com os demais modelos do grupo, avaliar todos nas imagens `conjunto == test` de `divisao_dados.csv`.
