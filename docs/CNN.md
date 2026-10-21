# CNN - Deep Learning e reconhecimento de imagens (fire / nofire)

Parte independente do projeto IAA3. Classifica imagens RGB em `fire` (incêndio, rótulo 1) e `nofire`
(sem incêndio, rótulo 0), usando só o dataset `data/raw/g20gli_dataset`. Não depende de RNA, Árvore de Decisão,
K-Means, SOM, fuzzy ou AG, e não lê o CSV de atributos.

Documentação detalhada: [arquitetura, pré-processamento, treino, limitações, vieses e riscos](CNN_arquitetura_e_limitacoes.md) |
[plano experimental](CNN_plano_experimental.md) | [seção de README](README_CNN.md).

## Arquivos da parte CNN

| Arquivo | Função |
|---|---|
| `src/cnn_dados.py` | Listagem pelas pastas, RGB, redimensionamento/normalização, agrupamento de duplicatas (dHash), divisão treino/val/teste e verificação de vazamento. Sem TensorFlow. |
| `src/cnn_metricas.py` | Accuracy, precision, recall, F1, taxas de FP/FN, matriz de confusão, gráficos e arquivos de métricas. Sem TensorFlow. |
| `src/cnn_analise.py` | Exemplos reais de acertos e erros, generalização treino x val x teste, erros por luminosidade e relatório automático. Sem TensorFlow; regenera tudo a partir de `predicoes_todos_conjuntos.csv`. |
| `src/cnn_treino.py` | Arquitetura, treino, salvamento, recarga, avaliação e chamada das análises (TensorFlow/Keras). |
| `src/cnn_prever.py` | Previsão de imagens novas (arquivo ou pasta), com probabilidade. |
| `tests/test_cnn_dados.py`, `test_cnn_metricas.py`, `test_cnn_analise.py`, `test_cnn_modelo.py` | Testes automatizados (`unittest`); os que exigem TensorFlow são pulados se ele não estiver instalado. |
| `notebooks/cnn_experimentos.ipynb` | Notebook de experimentação (chama o código de `src/`; saídas só aparecem ao executar). |
| `requirements.txt` (raiz) | Dependências do projeto, incluindo as usadas pela CNN (tensorflow, numpy, pandas, pillow, scikit-learn, matplotlib). O notebook também usa o IPython/Jupyter, que não consta nessa lista. |
| `results/cnn/divisao_dados.csv`, `inspecao_dataset.json` | Resultado real da inspeção do dataset e da divisão (gerados antes do treino). |

Gerados ao treinar: `models/cnn_fire_nofire.keras` e `_config.json`; em `results/cnn/`: `metricas_cnn.json/.csv`,
`metricas_cnn_validacao.*`, `matriz_confusao.png`, `historico_treinamento.png/.csv`, `relatorio_classificacao.txt`,
`predicoes_teste.csv`, `predicoes_todos_conjuntos.csv`, `exemplos_acertos.png`, `exemplos_erros.png`,
`exemplos_selecionados.csv`, `analise_generalizacao.json/.png`, `analise_luminosidade.csv`, `relatorio_cnn.md`, `resumo_execucao.json`.

No repositório estão versionados o modelo `models/cnn_fire_nofire.keras` e os arquivos acima em `results/cnn/`. O arquivo
`models/cnn_fire_nofire_config.json` não está versionado; sem ele, `cnn_prever.py` usa o tamanho de entrada do próprio modelo e o limiar 0,5.

## Execução no Windows (PowerShell, na raiz do projeto, ex.: `IAA3-main`)

```powershell
py -3.12 -m venv .venv          # ou py -3.11 / py -3.10
.venv\Scripts\Activate.ps1      # no CMD: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m unittest discover -s tests -v     # testes automatizados
python src\cnn_treino.py --smoke            # teste rápido do treino (pasta temporária)
python src\cnn_treino.py                    # treino completo + avaliação + análises
python src\cnn_analise.py                   # (opcional) regenera figuras/relatório sem retreinar
python src\cnn_prever.py caminho\da\imagem.jpg
```
Se o PowerShell bloquear a ativação: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
Em PC com pouca memória: `python src\cnn_treino.py --tamanho 96 --batch 16`.
Experimento opcional sem aumento de dados: `python src\cnn_treino.py --sem-aumento --modelo-saida models\cnn_sem_aumento.keras --resultados-dir results\cnn_sem_aumento`.

## Conformidade com o PDF da atividade (partes da CNN)

Esta tabela registra quais artefatos existem no repositório e onde estão. Os valores de desempenho não são repetidos aqui:
estão em `results/cnn/` (por exemplo `metricas_cnn.json` e `relatorio_cnn.md`), e `resumo_execucao.json` registra a configuração da
execução que os gerou (TensorFlow, épocas, semente, divisão). A tabela não substitui a execução dos testes automatizados
(`python -m unittest discover -s tests -v`) nem de um novo treino; **G** = depende do grupo.

| Seção | Exigência | Situação |
|---|---|---|
| 14 | dataset, classes, divisão treino/val/teste, pré-processamento | Presente: `cnn_dados.py`, `inspecao_dataset.json`, `divisao_dados.csv` |
| 14 | arquitetura, treinamento | Presente: `cnn_treino.py`; modelo treinado em `models/cnn_fire_nofire.keras`; execução registrada em `resumo_execucao.json` e `historico_treinamento.csv` |
| 14 / 18 | accuracy, precision, recall, F1, matriz de confusão | Presente: `metricas_cnn.json/.csv`, `metricas_cnn_validacao.*`, `matriz_confusao.png`, `relatorio_classificacao.txt` |
| 14 / 17 | exemplos de acertos e de erros | Presente: `exemplos_acertos.png`, `exemplos_erros.png`, `exemplos_selecionados.csv` |
| 18 | loss, accuracy, curvas de treinamento, validação | Presente: `historico_treinamento.png/.csv` |
| 19 | análise treino x validação x teste e overfitting | Presente: `analise_generalizacao.json/.png` |
| 20 | viés, FP/FN, riscos, explicabilidade, responsabilidade | Texto em `CNN_arquitetura_e_limitacoes.md` (hipóteses, não resultados); verificação por luminosidade em `analise_luminosidade.csv` |
| 22 | /data /src /models /notebooks /tests /results /docs, README, requirements | Pastas presentes; dependências no `requirements.txt` da raiz; texto de README da CNN em `README_CNN.md` (o `README.md` da raiz ainda contém só o título) |
| 26 | código, notebook, plano experimental, README, análise ética | Presentes. O notebook não tem saídas salvas: elas só aparecem ao executar as células |
| 26 | modelo treinado, resultados e métricas | Presentes em `models/` e `results/cnn/` |
| 15 / 26 | comparação entre modelos, apresentação, demonstração | **G**: a CNN fornece Accuracy/Precision/Recall/F1 e tempos em `metricas_cnn.json/.csv` |

## Integração futura
- Métricas em `results/cnn/metricas_cnn.json` / `.csv` (accuracy, precision, recall, f1_score, TN/FP/FN/TP, tempos; classe positiva `fire`).
- Para uma comparação justa, avaliar os demais modelos nas imagens com `conjunto == test` em `results/cnn/divisao_dados.csv`.
- Não há padrão comum de avaliação no código existente (`data_processing.py` e `fuzzy_ga.py` não calculam métricas);
  o formato acima é genérico e pode ser adaptado quando o grupo definir um padrão.
