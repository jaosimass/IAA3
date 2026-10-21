
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv(
    "data/raw/ibama/ROI.csv",
    sep=";",
    low_memory=False
)

print("\nValores de TIPO DETECCAO:")
print(df["TIPO DETECCAO"].value_counts().head(15))

print("\nExemplos de horários:")
print(
    df[
        [
            "DATA/HORA INICIO FOGO",
            "DATA/HORA DETECCAO",
            "DATA/HORA PRIMEIRO ATAQUE"
        ]
    ].head(10).to_string(index=False)
)

id_col = "SEQ OCORRENCIA INCENDIO"
alvo = "NIVEL ACIONAMENTO"

categoricas = [
    "TIPO AREA",
    "VEGETACAO ATINGIDA",
    "DENTRO ENTORNO"
    "TIPO DETECCAO"
]

df["QTD AREA ATINGIDA"] = pd.to_numeric(
    df["QTD AREA ATINGIDA"]
    .astype(str)
    .str.replace(",", ".", regex=False),
    errors="coerce"
)

# Agrupar as linhas que pertencem à mesma ocorrência
agrupado = df.groupby(id_col).agg({
    alvo: lambda s: (
        s.dropna().iloc[0] if not s.dropna().empty else None
    ),
    "QTD AREA ATINGIDA": "median",
    **{
        coluna: lambda s: sorted(s.dropna().astype(str).unique())
        for coluna in categoricas
    }
})

# Transformar cada categoria em uma coluna independente
partes = []

for coluna in categoricas:
    listas = agrupado[coluna].apply(
        lambda valores: valores if valores else ["NAO_INFORMADO"]
    )

    codificador = MultiLabelBinarizer()
    matriz = codificador.fit_transform(listas)

    nomes = [
        f"{coluna}_{categoria}"
        .replace(" ", "_")
        .replace("/", "_")
        for categoria in codificador.classes_
    ]

    parte = pd.DataFrame(
        matriz,
        index=agrupado.index,
        columns=nomes
    )

    partes.append(parte)

# Montar a tabela final
resultado = pd.concat(
    [
        agrupado[[alvo, "QTD AREA ATINGIDA"]],
        *partes
    ],
    axis=1
).reset_index()

# Remover ocorrências sem o alvo, se houver
resultado = resultado.dropna(subset=[alvo])

saida = Path("data/processed")
saida.mkdir(parents=True, exist_ok=True)

resultado.to_csv(
    saida / "dados_arvore.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Total de ocorrências:", len(resultado))
print("Total de atributos:", len(resultado.columns) - 2)
print("\nDistribuição dos níveis:")
print(resultado[alvo].value_counts())
print("\nColunas geradas:")
print(resultado.columns.tolist())
print("\nArquivo atualizado: data/processed/dados_arvore.csv")


print("\nPreenchimento das colunas originais:")

colunas_relevantes = [
    "TIPO DETECCAO",
    "QTD COMBATENTE",
    "QTD BRIGADISTA",
    "QTD AEREO",
    "FORMA EXTINCAO",
    "DATA/HORA INICIO FOGO",
    "DATA/HORA DETECCAO",
    "DATA/HORA PRIMEIRO ATAQUE"
]

for coluna in colunas_relevantes:
    preenchidos = df[coluna].notna().sum()
    percentual = preenchidos / len(df) * 100

    print(
        f"{coluna}: {preenchidos} registros "
        f"({percentual:.1f}%)"
    )
