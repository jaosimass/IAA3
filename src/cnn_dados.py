"""Carregamento, pré-processamento e divisão do dataset RGB (fire / nofire) para a CNN.

Este módulo NÃO depende de TensorFlow: pode ser usado e testado sozinho.

Convenções (iguais às de src/data_processing.py):
    - dataset em data/raw/g20gli_dataset/{fire,nofire}/
    - rótulo: fire = 1 (incêndio), nofire = 0 (sem incêndio)

Prevenção de vazamento de dados:
    Imagens idênticas ou quase idênticas (mesmo quadro de vídeo, recortes muito
    parecidos) são agrupadas por um hash perceptual (dHash). A divisão em
    treino/validação/teste é feita por GRUPO, então um grupo nunca é dividido
    entre conjuntos diferentes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ_PROJETO / "data" / "raw" / "g20gli_dataset"

# A posição na tupla é o rótulo numérico: nofire = 0, fire = 1.
CLASSES = ("nofire", "fire")
ROTULO = {nome: indice for indice, nome in enumerate(CLASSES)}
EXTENSOES = (".png", ".jpg", ".jpeg", ".tif")

TAMANHO_PADRAO = 128
SEMENTE = 42
PROPORCOES = (0.70, 0.15, 0.15)  # treino, validação, teste
CONJUNTOS = ("train", "val", "test")
LIMIAR_DHASH = 4  # distância de Hamming (de 64 bits) até a qual duas imagens são "quase iguais"


# --------------------------------------------------------------------------- #
# Listagem das imagens
# --------------------------------------------------------------------------- #
def listar_imagens(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Lista as imagens pelas pastas das classes.

    Retorna DataFrame com: arquivo (caminho relativo a data_dir, com '/'),
    classe ('fire'/'nofire') e rotulo (1/0). Ordenado, para ser determinístico.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Pasta do dataset não encontrada: {data_dir}. "
            "Coloque as imagens em data/raw/g20gli_dataset/fire e /nofire."
        )
    linhas = []
    for classe in CLASSES:
        pasta = data_dir / classe
        if not pasta.is_dir():
            raise FileNotFoundError(f"Pasta da classe '{classe}' não encontrada: {pasta}")
        for caminho in sorted(pasta.iterdir()):
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES:
                linhas.append({"arquivo": f"{classe}/{caminho.name}",
                               "classe": classe,
                               "rotulo": ROTULO[classe]})
    df = pd.DataFrame(linhas, columns=["arquivo", "classe", "rotulo"])
    if df.empty:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {data_dir}")
    return df


# --------------------------------------------------------------------------- #
# Carregamento / pré-processamento
# --------------------------------------------------------------------------- #
def carregar_imagem(origem, tamanho: int = TAMANHO_PADRAO) -> np.ndarray:
    """Abre uma imagem, converte para RGB, redimensiona e normaliza para [0, 1].

    Esta MESMA função é usada no treino, na avaliação e na previsão de imagens
    novas, garantindo o mesmo pré-processamento em todas as etapas.

    Retorna array float32 com shape (tamanho, tamanho, 3).
    """
    with Image.open(origem) as img:
        img = img.convert("RGB").resize((tamanho, tamanho), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def carregar_lote(arquivos, data_dir: Path | str = DATA_DIR,
                  tamanho: int = TAMANHO_PADRAO) -> np.ndarray:
    """Carrega vários arquivos (caminhos relativos a data_dir). Shape (n, t, t, 3)."""
    data_dir = Path(data_dir)
    arquivos = list(arquivos)
    lote = np.empty((len(arquivos), tamanho, tamanho, 3), dtype=np.float32)
    for i, arquivo in enumerate(arquivos):
        lote[i] = carregar_imagem(data_dir / arquivo, tamanho)
    return lote


# --------------------------------------------------------------------------- #
# Detecção de duplicatas (hash perceptual) e agrupamento
# --------------------------------------------------------------------------- #
def calcular_dhash(origem) -> int:
    """dHash de 64 bits: compara pixels vizinhos de uma versão 9x8 em tons de cinza."""
    with Image.open(origem) as img:
        cinza = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        a = np.asarray(cinza, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    valor = 0
    for b in bits:
        valor = (valor << 1) | int(b)
    return valor


def _distancias_hamming(h: np.ndarray, i: int) -> np.ndarray:
    """Distância de Hamming entre o hash i e todos os hashes de índice > i."""
    xor = np.ascontiguousarray(h[i] ^ h[i + 1:])
    if xor.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unpackbits(xor.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def agrupar_duplicatas(hashes, limiar: int = LIMIAR_DHASH) -> np.ndarray:
    """Agrupa imagens iguais/quase iguais (união transitiva por distância <= limiar).

    Retorna um array com o id do grupo de cada imagem (ids de 0 a n_grupos-1).
    """
    h = np.asarray([int(x) for x in hashes], dtype=np.uint64)
    n = len(h)
    pai = list(range(n))

    def achar(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    for i in range(n - 1):
        dist = _distancias_hamming(h, i)
        for j in np.nonzero(dist <= limiar)[0]:
            ra, rb = achar(i), achar(i + 1 + int(j))
            if ra != rb:
                pai[rb] = ra

    raizes = [achar(i) for i in range(n)]
    mapa = {r: k for k, r in enumerate(dict.fromkeys(raizes))}
    return np.array([mapa[r] for r in raizes], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Divisão treino / validação / teste sem vazamento
# --------------------------------------------------------------------------- #
def dividir_dados(df: pd.DataFrame, grupos, proporcoes=PROPORCOES,
                  semente: int = SEMENTE) -> pd.DataFrame:
    """Divide em treino/validação/teste, estratificado por classe e SEM separar grupos.

    Cada grupo (imagens iguais/quase iguais) vai inteiro para um único conjunto.
    Retorna cópia de df com as colunas 'grupo' e 'conjunto' ('train'/'val'/'test').
    """
    if abs(sum(proporcoes) - 1.0) > 1e-9:
        raise ValueError("As proporções devem somar 1.")
    df = df.reset_index(drop=True).copy()
    df["grupo"] = np.asarray(grupos, dtype=np.int64)
    rng = np.random.default_rng(semente)

    # Rótulo do grupo = rótulo majoritário (na prática os grupos são de uma só classe).
    info = df.groupby("grupo").agg(n=("rotulo", "size"), media=("rotulo", "mean"))
    info["rotulo"] = (info["media"] >= 0.5).astype(int)

    destino = {}
    for rotulo in sorted(info["rotulo"].unique()):
        ids = info.index[info["rotulo"] == rotulo].to_numpy().copy()
        rng.shuffle(ids)
        total = int(info.loc[ids, "n"].sum())
        alvo = np.array(proporcoes, dtype=float) * total
        atual = np.zeros(len(proporcoes), dtype=float)
        for g in ids:
            k = int(np.argmax(alvo - atual))  # conjunto mais "atrasado" em relação à meta
            destino[g] = CONJUNTOS[k]
            atual[k] += info.loc[g, "n"]

    df["conjunto"] = df["grupo"].map(destino)
    return df


def verificar_vazamento(manifesto: pd.DataFrame) -> dict:
    """Confere que não há vazamento entre os conjuntos. Levanta ValueError se houver.

    Verifica: (1) nenhum arquivo em mais de um conjunto; (2) nenhum grupo de
    duplicatas dividido entre conjuntos; (3) as duas classes existem em cada conjunto.
    """
    problemas = []
    n_conj_arquivo = manifesto.groupby("arquivo")["conjunto"].nunique()
    if (n_conj_arquivo > 1).any():
        problemas.append(f"{int((n_conj_arquivo > 1).sum())} arquivo(s) em mais de um conjunto")
    n_conj_grupo = manifesto.groupby("grupo")["conjunto"].nunique()
    if (n_conj_grupo > 1).any():
        problemas.append(f"{int((n_conj_grupo > 1).sum())} grupo(s) de duplicatas divididos entre conjuntos")
    if manifesto["arquivo"].duplicated().any():
        problemas.append("arquivos repetidos no manifesto")
    for conjunto in CONJUNTOS:
        classes = set(manifesto.loc[manifesto["conjunto"] == conjunto, "classe"])
        if classes != set(CLASSES):
            problemas.append(f"conjunto '{conjunto}' não contém as duas classes: {sorted(classes)}")
    if problemas:
        raise ValueError("Vazamento/inconsistência na divisão: " + "; ".join(problemas))
    return {"arquivos_em_mais_de_um_conjunto": 0,
            "grupos_divididos_entre_conjuntos": 0,
            "classes_em_todos_os_conjuntos": True}


def construir_manifesto(data_dir: Path | str = DATA_DIR, proporcoes=PROPORCOES,
                        semente: int = SEMENTE, limiar_dhash: int = LIMIAR_DHASH) -> pd.DataFrame:
    """Lista as imagens, agrupa duplicatas e divide em treino/val/teste (verificado)."""
    data_dir = Path(data_dir)
    df = listar_imagens(data_dir)
    hashes = [calcular_dhash(data_dir / a) for a in df["arquivo"]]
    grupos = agrupar_duplicatas(hashes, limiar_dhash)
    manifesto = dividir_dados(df, grupos, proporcoes, semente)
    verificar_vazamento(manifesto)
    return manifesto


def resumo_dataset(manifesto: pd.DataFrame) -> dict:
    """Contagem de imagens por conjunto e classe, e quantidade de grupos de duplicatas."""
    tabela = pd.crosstab(manifesto["conjunto"], manifesto["classe"]).reindex(list(CONJUNTOS))
    por_conjunto = {c: {k: int(v) for k, v in tabela.loc[c].items()} for c in tabela.index}
    tam_grupo = manifesto.groupby("grupo").size()
    return {
        "total_imagens": int(len(manifesto)),
        "por_classe": {k: int(v) for k, v in manifesto["classe"].value_counts().items()},
        "por_conjunto": por_conjunto,
        "grupos_com_duplicatas": int((tam_grupo > 1).sum()),
        "imagens_em_grupos_com_duplicatas": int(tam_grupo[tam_grupo > 1].sum()),
    }


if __name__ == "__main__":  # inspeção rápida do dataset
    m = construir_manifesto()
    print(resumo_dataset(m))
