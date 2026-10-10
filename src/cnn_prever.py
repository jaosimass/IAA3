"""Previsão de novas imagens com a CNN treinada (fire / nofire).

Uso (a partir da raiz do projeto):
    python src/cnn_prever.py caminho/da/imagem.jpg
    python src/cnn_prever.py img1.jpg img2.png pasta_com_imagens/
    python src/cnn_prever.py imagem.jpg --modelo models/cnn_fire_nofire.keras --limiar 0.5

Uso como função:
    from cnn_prever import prever_imagem
    resultado = prever_imagem("minha_foto.jpg")
    # {'arquivo': ..., 'classe': 'fire', 'prob_fire': 0.97, 'prob_nofire': 0.03, 'limiar': 0.5}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cnn_dados as dados  # noqa: E402

RAIZ = dados.RAIZ_PROJETO
MODELO_PADRAO = RAIZ / "models" / "cnn_fire_nofire.keras"


def carregar_modelo(caminho_modelo: Path | str = MODELO_PADRAO):
    """Carrega o modelo .keras e sua configuração (tamanho da imagem, limiar, classes)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from tensorflow import keras
    except ImportError as erro:
        raise ImportError("TensorFlow não está instalado. Execute: pip install -r requirements-cnn.txt") from erro

    caminho_modelo = Path(caminho_modelo)
    if not caminho_modelo.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho_modelo}. Treine antes com: python src/cnn_treino.py")
    config_path = caminho_modelo.with_name(caminho_modelo.stem + "_config.json")
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:  # sem config: usa o tamanho de entrada do próprio modelo
        config = {}
    modelo = keras.models.load_model(caminho_modelo)
    config.setdefault("tamanho_imagem", int(modelo.input_shape[1]))
    config.setdefault("limiar", 0.5)
    return modelo, config


def prever_imagem(origem, modelo=None, config=None, limiar: float | None = None,
                  caminho_modelo: Path | str = MODELO_PADRAO) -> dict:
    """Classifica UMA imagem e devolve a classe e as probabilidades."""
    return prever_varias([origem], modelo, config, limiar, caminho_modelo)[0]


def prever_varias(origens, modelo=None, config=None, limiar: float | None = None,
                  caminho_modelo: Path | str = MODELO_PADRAO) -> list[dict]:
    """Classifica várias imagens de uma vez (mesmo pré-processamento do treino)."""
    if modelo is None or config is None:
        modelo, config = carregar_modelo(caminho_modelo)
    limiar = float(config["limiar"] if limiar is None else limiar)
    tamanho = int(config["tamanho_imagem"])

    origens = list(origens)
    lote = np.stack([dados.carregar_imagem(o, tamanho) for o in origens])
    prob_fire = modelo.predict(lote, verbose=0).ravel()

    resultados = []
    for origem, p in zip(origens, prob_fire):
        p = float(p)
        resultados.append({
            "arquivo": str(origem),
            "classe": "fire" if p >= limiar else "nofire",
            "prob_fire": p,
            "prob_nofire": 1.0 - p,
            "limiar": limiar,
        })
    return resultados


def _expandir_entradas(entradas) -> list[Path]:
    arquivos: list[Path] = []
    for entrada in entradas:
        caminho = Path(entrada)
        if caminho.is_dir():
            arquivos += sorted(p for p in caminho.rglob("*")
                               if p.is_file() and p.suffix.lower() in dados.EXTENSOES)
        elif caminho.is_file():
            arquivos.append(caminho)
        else:
            raise FileNotFoundError(f"Arquivo ou pasta não encontrado: {caminho}")
    if not arquivos:
        raise FileNotFoundError("Nenhuma imagem encontrada nas entradas informadas.")
    return arquivos


def main(argv=None):
    p = argparse.ArgumentParser(description="Prevê se imagens contêm incêndio (fire) ou não (nofire).")
    p.add_argument("imagens", nargs="+", help="arquivo(s) de imagem ou pasta(s)")
    p.add_argument("--modelo", default=str(MODELO_PADRAO))
    p.add_argument("--limiar", type=float, default=None,
                   help="limiar para 'fire' (padrão: o salvo na configuração do modelo)")
    a = p.parse_args(argv)

    arquivos = _expandir_entradas(a.imagens)
    modelo, config = carregar_modelo(a.modelo)
    for r in prever_varias(arquivos, modelo, config, a.limiar):
        prob_classe = r["prob_fire"] if r["classe"] == "fire" else r["prob_nofire"]
        print(f"{r['arquivo']}: {r['classe'].upper()} "
              f"(confiança {prob_classe * 100:.1f}% | P(fire)={r['prob_fire']:.4f})")


if __name__ == "__main__":
    main()
