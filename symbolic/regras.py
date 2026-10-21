import pandas as pd

df = pd.read_csv("data/raw/ibama/ROI.csv", sep=";")

def classificar_ocorrencia(
        nivel,
        tipo_area,
        vegetacoes,
        dentro_entorno,
        causa
):
    if (
        nivel == "Nível 3"
        and tipo_area == "Área Protegida"
        and "MATA OU FLORESTA NATIVA" in vegetacoes
    ):
        return "ALTA"

    if (
        nivel == "Nível 3"
        and tipo_area == "Área Protegida"
    ):
        return "ALTA"

    if (
        nivel == "Nível 2"
        and tipo_area == "Área Protegida"
        and "MATA OU FLORESTA NATIVA" in vegetacoes 
    ):
        return "MÉDIA"

    if (
        nivel == "Nível 2"
        and tipo_area == "Área Protegida"
    ):
        return "MÉDIA"

    return "BAIXA"

def preparar_ocorrencia(ocorrencia):

    nivel = (
        ocorrencia["NIVEL ACIONAMENTO"].dropna().iloc[0]
        if not ocorrencia["NIVEL ACIONAMENTO"].dropna().empty
        else "NÃO INFORMADO"
    )

    tipo_area = (
        ocorrencia["TIPO AREA"].dropna().iloc[0]
        if not ocorrencia["TIPO AREA"].dropna().empty
        else "NÃO INFORMADO"
    )

    vegetacoes = set(
        ocorrencia["VEGETACAO ATINGIDA"]
        .dropna()
        .unique()
    )

    dentro_entorno = set(
        ocorrencia["DENTRO ENTORNO"]
        .dropna()
        .unique()
    )

    causas = set(
        ocorrencia["PROVAVEL CAUSA"]
        .dropna()
        .unique()
    )

    causa = next(iter(causas), "INDETERMINADO")

    return {
        "nivel": nivel,
        "tipo_area": tipo_area,
        "vegetacoes": vegetacoes,
        "dentro_entorno": dentro_entorno,
        "causa": causa
    }

resultados = []

for numero_ocorrencia, ocorrencia in df.groupby("SEQ OCORRENCIA INCENDIO"):

    caracteristicas = preparar_ocorrencia(ocorrencia)

    classificacao = classificar_ocorrencia(
        caracteristicas["nivel"],
        caracteristicas["tipo_area"],
        caracteristicas["vegetacoes"],
        caracteristicas["dentro_entorno"],
        caracteristicas["causa"]
    )

    resultados.append({
        "SEQ OCORRENCIA INCENDIO": numero_ocorrencia,
        "NIVEL ACIONAMENTO": caracteristicas["nivel"],
        "TIPO AREA": caracteristicas["tipo_area"],
        "PROVAVEL CAUSA": caracteristicas["causa"],
        "CLASSIFICACAO": classificacao
    })

resultados_df = pd.DataFrame(resultados)

print("\n==============================")
print("RESULTADO DO SISTEMA SIMBÓLICO")
print("==============================")

print("\nTotal de ocorrências classificadas:")
print(len(resultados_df))

print("\nQuantidade por classificação:")
print(resultados_df["CLASSIFICACAO"].value_counts())

print("\nPrimeiras classificações:")
print(resultados_df.head(10))


# Salvar resultados do sistema simbolico
resultados_df.to_csv(
    "data/processed/resultados_simbolico.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nArquivo salvo em: "
    "data/processed/resultados_simbolico.csv"
)
