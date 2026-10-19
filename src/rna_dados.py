from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


RAIZ = Path(__file__).resolve().parent.parent

ARQUIVO_ATRIBUTOS = RAIZ / "data" / "processed" / "atributos_g20gli.csv"
ARQUIVO_DIVISAO = RAIZ / "results" / "cnn" / "divisao_dados.csv"


FEATURES = [
    "media_R",
    "media_G",
    "media_B",
    "std_R",
    "std_G",
    "luminosidade"
]


def carregar_dados():
    atributos = pd.read_csv(ARQUIVO_ATRIBUTOS)
    divisao = pd.read_csv(ARQUIVO_DIVISAO)

    divisao["id_imagem"] = divisao["arquivo"].apply(
        lambda caminho: Path(caminho).name
    )

    # Junta os atributos com a divisão train/val/test
    dados = atributos.merge(
        divisao[["id_imagem", "conjunto"]],
        on="id_imagem",
        how="inner"
    )

    # Separa os três conjuntos
    treino = dados[dados["conjunto"] == "train"]
    validacao = dados[dados["conjunto"] == "val"]
    teste = dados[dados["conjunto"] == "test"]

    # Features = entradas da RNA
    # target = resposta correta
    x_train = treino[FEATURES].values
    y_train = treino["target"].values

    x_val = validacao[FEATURES].values
    y_val = validacao["target"].values

    x_test = teste[FEATURES].values
    y_test = teste["target"].values

    scaler = StandardScaler()

    # O scaler aprende SOMENTE com os dados de treino
    x_train = scaler.fit_transform(x_train)

    # Validação e teste usam o que foi aprendido no treino
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        scaler
    )


if __name__ == "__main__":
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        scaler
    ) = carregar_dados()

    print("Treino:", x_train.shape, y_train.shape)
    print("Validação:", x_val.shape, y_val.shape)
    print("Teste:", x_test.shape, y_test.shape)