# Monitoramento Florestal com K-Means

Projeto acadêmico de aprendizado não supervisionado que extrai características simples de imagens RGB e utiliza **K-Means** para agrupar imagens visualmente semelhantes.

> Importante: o projeto é experimental. O K-Means não entende semanticamente o que é um incêndio. Os grupos precisam ser avaliados e interpretados; este projeto não deve ser usado como sistema operacional de alerta de incêndio.

## Requisitos

- Python 3.10 ou superior
- Git (opcional, para versionamento)
- Dataset de imagens RGB com pastas `fire` e `nofire`

## Estrutura esperada do dataset

Coloque as imagens nestas pastas:

```text
data/
├── train/
│   ├── fire/
│   └── nofire/
└── test/
    ├── fire/
    └── nofire/
```

Use imagens distintas entre treino e teste. Não coloque a mesma imagem nos dois conjuntos. Se o dataset já vier separado, preserve a divisão original.

## Instalação no Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se a ativação do ambiente virtual for bloqueada, você pode instalar as dependências usando o interpretador do ambiente diretamente:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Executar

Na pasta raiz do projeto:

```powershell
python src/main.py
```

## Características extraídas

- Médias e desvios-padrão dos canais RGB
- Média de matiz, saturação e brilho (HSV)
- Proporção de pixels que satisfazem uma heurística simples de cor semelhante a fogo

A proporção de cor semelhante a fogo é apenas uma característica visual heurística. Não é uma detecção confiável de fogo por si só.

## Como o K-Means é utilizado

1. O código extrai características numéricas de cada imagem.
2. `StandardScaler` padroniza as características, usando estatísticas do conjunto de treino.
3. O K-Means é ajustado apenas com as características do treino, sem usar os rótulos.
4. As imagens de teste são atribuídas aos clusters já aprendidos.
5. Para interpretar os clusters, cada cluster recebe o nome da classe mais frequente no treino. Essa etapa usa rótulos apenas depois do agrupamento e não altera o ajuste do K-Means.

O valor inicial de `K` é 2 e pode ser alterado em `src/main.py`. Isso é uma escolha experimental, não uma garantia de que existam exatamente dois grupos naturais.

## Resultados gerados

A execução cria a pasta `results/` com:

- `train_clusters.csv`: características, rótulos de referência e clusters do treino
- `test_clusters.csv`: características, rótulos de referência, clusters e interpretação do teste
- `metrics.json`: métricas e relatório por classe
- `confusion_matrix.csv` e `confusion_matrix.png`
- `clusters_rgb.png`: visualização dos clusters pelas médias de vermelho e verde
- `cluster_counts.png`: quantidade de imagens em cada cluster
- `cluster_examples/`: exemplos visuais de cada cluster

## Avaliação

O projeto calcula silhouette quando aplicável, Adjusted Rand Index (ARI) e Normalized Mutual Information (NMI). Também gera métricas por classe após interpretar os clusters com base no conjunto de treino.

- Silhouette avalia a separação e coesão dos grupos.
- ARI e NMI comparam os agrupamentos com os rótulos de referência, sem exigir que os números dos clusters correspondam diretamente às classes.
- As métricas por classe dependem do mapeamento de clusters para classes aprendido no treino; não significam que o K-Means se tornou um classificador supervisionado.

## Limitações

- Estatísticas globais de cor podem perder informações espaciais e detalhes importantes.
- Imagens RGB não fornecem bandas espectrais necessárias para calcular NDVI, NBR ou dNBR.
- Fumaça, iluminação, solo, pôr do sol e outras cenas podem confundir características de cor.
- Resultados dependem do dataset, da qualidade das imagens e da seleção de características.
- O projeto é uma demonstração acadêmica, não uma ferramenta de detecção validada em campo.
