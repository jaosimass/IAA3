
import os
import pandas as pd

# Criar a pasta de resultados, caso não exista
os.makedirs("data/processed", exist_ok=True)

# 1. Carregar os dados do IBAMA
df = pd.read_csv("data/raw/ibama/ROI.csv", sep=";")

# 2. Verificar as colunas necessárias
colunas_obrigatorias = [
    "SEQ OCORRENCIA INCENDIO",
    "NIVEL ACIONAMENTO",
    "TIPO AREA"
]

colunas_faltantes = [
    coluna for coluna in colunas_obrigatorias
    if coluna not in df.columns
]

if colunas_faltantes:
    raise ValueError(
        f"Colunas não encontradas: {colunas_faltantes}"
    )


# 3. Definir as regras do sistema simbólico
def classificar_ocorrencia(nivel, tipo_area):

    # Regra 1: Nível 3 em Área Protegida
    if nivel == "Nível 3" and tipo_area == "Área Protegida":
        return "ALTA"

    # Regra 2: Nível 2 em Área Protegida
    if nivel == "Nível 2" and tipo_area == "Área Protegida":
        return "MÉDIA"

    # Regra 3: Demais situações
    return "BAIXA"


# 4. Obter o primeiro valor não nulo de uma coluna
def primeiro_valor(serie, padrao):
    valores = serie.dropna()

    if valores.empty:
        return padrao

    return valores.iloc[0]


# 5. Agrupar os registros por ocorrência
resultados = []

for numero, ocorrencia in df.groupby("SEQ OCORRENCIA INCENDIO"):

    nivel = primeiro_valor(
        ocorrencia["NIVEL ACIONAMENTO"],
        "NÃO INFORMADO"
    )

    tipo_area = primeiro_valor(
        ocorrencia["TIPO AREA"],
        "NÃO INFORMADO"
    )

    classificacao = classificar_ocorrencia(
        nivel,
        tipo_area
    )

    resultados.append({
        "SEQ OCORRENCIA INCENDIO": numero,
        "NIVEL ACIONAMENTO": nivel,
        "TIPO AREA": tipo_area,
        "CLASSIFICACAO": classificacao
    })


# 6. Organizar os resultados
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


# 7. Salvar os resultados
caminho_saida = "data/processed/resultados_simbolico.csv"

resultados_df.to_csv(
    caminho_saida,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nArquivo salvo em: {caminho_saida}")