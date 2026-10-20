"""Treino, validação e avaliação (em teste) da CNN fire / nofire.

Uso (a partir da raiz do projeto):
    python src/cnn_treino.py              # treinamento completo
    python src/cnn_treino.py --smoke      # teste rápido (poucas imagens, 2 épocas)
    python src/cnn_treino.py --help

A CNN é independente dos demais modelos do grupo: usa apenas as imagens RGB de
data/raw/g20gli_dataset e grava somente em:
    models/cnn_fire_nofire.keras  e  models/cnn_fire_nofire_config.json
    results/cnn/...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cnn_analise as analise  # noqa: E402
import cnn_dados as dados  # noqa: E402
import cnn_metricas as metricas_mod  # noqa: E402

RAIZ = dados.RAIZ_PROJETO
MODELO_PADRAO = RAIZ / "models" / "cnn_fire_nofire.keras"
CONFIG_PADRAO = RAIZ / "models" / "cnn_fire_nofire_config.json"
RESULTADOS_PADRAO = RAIZ / "results" / "cnn"


def _importar_tf():
    """Importa o TensorFlow só quando necessário (mensagem clara se não estiver instalado)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as erro:
        raise ImportError(
            "TensorFlow não está instalado neste ambiente. "
            "Ative o ambiente virtual e execute: pip install -r requirements.txt"
        ) from erro
    return tf, tf.keras


# --------------------------------------------------------------------------- #
# Arquitetura
# --------------------------------------------------------------------------- #
def construir_cnn(tamanho: int = dados.TAMANHO_PADRAO, aumento: bool = True):
    """CNN pequena (~100 mil parâmetros), adequada para ~1900 imagens e CPU comum.

    Entrada: imagem RGB (tamanho x tamanho x 3) já normalizada em [0, 1].
    Saída: probabilidade de 'fire' (sigmoid): >= limiar -> fire, senão nofire.

    Contra overfitting: aumento de dados geométrico (ativo só no treino), BatchNorm,
    GlobalAveragePooling (poucos parâmetros), Dropout, regularização L2 e
    EarlyStopping + redução da taxa de aprendizado (no treino).
    Não há aumento de cor/brilho de propósito: a cor (laranja/vermelho) é informativa.
    `aumento=False` remove o aumento de dados (útil para experimento de ablação / análise de overfitting).
    """
    _, keras = _importar_tf()
    L = keras.layers

    def bloco(x, filtros):
        x = L.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)
        return L.MaxPooling2D(2)(x)

    entrada = keras.Input(shape=(tamanho, tamanho, 3), name="imagem_rgb")
    x = entrada
    if aumento:
        x = L.RandomFlip("horizontal")(x)
        x = L.RandomRotation(0.05)(x)
        x = L.RandomZoom(0.1)(x)
        x = L.RandomTranslation(0.08, 0.08)(x)
    for filtros in (16, 32, 64, 128):
        x = bloco(x, filtros)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.4)(x)
    x = L.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = L.Dropout(0.3)(x)
    saida = L.Dense(1, activation="sigmoid", name="prob_fire")(x)

    modelo = keras.Model(entrada, saida, name="cnn_fire_nofire")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                   loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #
def subamostrar_manifesto(manifesto: pd.DataFrame, por_conjunto_e_classe: dict, semente: int) -> pd.DataFrame:
    """Reduz o manifesto (usado no --smoke) mantendo conjuntos e classes separados."""
    partes = []
    for conjunto, n in por_conjunto_e_classe.items():
        for classe in dados.CLASSES:
            sub = manifesto[(manifesto["conjunto"] == conjunto) & (manifesto["classe"] == classe)]
            partes.append(sub.sample(n=min(n, len(sub)), random_state=semente))
    return pd.concat(partes).sort_values("arquivo").reset_index(drop=True)


def _carregar_conjunto(manifesto, conjunto, data_dir, tamanho):
    sub = manifesto[manifesto["conjunto"] == conjunto].reset_index(drop=True)
    x = dados.carregar_lote(sub["arquivo"], data_dir, tamanho)
    y = sub["rotulo"].to_numpy(dtype=np.float32)
    return sub, x, y


# --------------------------------------------------------------------------- #
# Treinamento completo
# --------------------------------------------------------------------------- #
def treinar(epocas: int = 40, batch: int = 32, tamanho: int = dados.TAMANHO_PADRAO,
            semente: int = dados.SEMENTE, paciencia: int = 8, limiar: float = 0.5,
            data_dir: Path | str = dados.DATA_DIR,
            modelo_saida: Path | str = MODELO_PADRAO,
            config_saida: Path | str | None = None,
            resultados_dir: Path | str = RESULTADOS_PADRAO,
            smoke: bool = False, aumento: bool = True) -> dict:
    tf, keras = _importar_tf()
    keras.utils.set_random_seed(semente)

    data_dir = Path(data_dir)
    modelo_saida = Path(modelo_saida)
    config_saida = Path(config_saida) if config_saida else modelo_saida.with_name(
        modelo_saida.stem + "_config.json")
    resultados_dir = Path(resultados_dir)
    resultados_dir.mkdir(parents=True, exist_ok=True)
    modelo_saida.parent.mkdir(parents=True, exist_ok=True)

    # 1) Dataset, grupos de duplicatas e divisão treino/val/teste (com verificação de vazamento)
    print("[1/6] Lendo o dataset e dividindo em treino/validação/teste...")
    manifesto = dados.construir_manifesto(data_dir, semente=semente)
    if smoke:
        manifesto = subamostrar_manifesto(
            manifesto, {"train": 60, "val": 15, "test": 15}, semente)
    verificacao = dados.verificar_vazamento(manifesto)
    resumo = dados.resumo_dataset(manifesto)
    manifesto.to_csv(resultados_dir / "divisao_dados.csv", index=False)
    print("      ", json.dumps(resumo, ensure_ascii=False))
    print("       verificação de vazamento:", verificacao)

    # 2) Carregamento (redimensiona e normaliza para [0, 1])
    print(f"[2/6] Carregando imagens ({tamanho}x{tamanho}, RGB, normalizadas em [0,1])...")
    sub_tr, x_tr, y_tr = _carregar_conjunto(manifesto, "train", data_dir, tamanho)
    sub_val, x_val, y_val = _carregar_conjunto(manifesto, "val", data_dir, tamanho)
    sub_te, x_te, y_te = _carregar_conjunto(manifesto, "test", data_dir, tamanho)
    print(f"       treino={x_tr.shape}  validação={x_val.shape}  teste={x_te.shape}")

    # 3) Modelo
    print("[3/6] Construindo a CNN...")
    modelo = construir_cnn(tamanho, aumento)
    modelo.summary()

    # 4) Treinamento (usa só treino e validação; o teste NÃO é usado aqui)
    print("[4/6] Treinando...")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(2, paciencia // 3), min_lr=1e-6),
    ]
    t0 = time.perf_counter()
    hist = modelo.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=epocas,
                      batch_size=batch, callbacks=callbacks, shuffle=True, verbose=2)
    tempo_treino = time.perf_counter() - t0
    historico = pd.DataFrame(hist.history)
    melhor_epoca = int(np.argmin(historico["val_loss"])) + 1

    # 5) Salvar e recarregar o modelo; a avaliação usa o modelo RECARREGADO do disco
    print("[5/6] Salvando o modelo e testando o carregamento...")
    modelo.save(modelo_saida)
    carregado = keras.models.load_model(modelo_saida)
    dif = float(np.max(np.abs(
        modelo.predict(x_te, batch_size=64, verbose=0) -
        carregado.predict(x_te, batch_size=64, verbose=0))))
    print(f"       modelo salvo em {modelo_saida} | diferença máx. entre modelo original e recarregado: {dif:.2e}")
    if dif > 1e-4:
        print("       AVISO: diferença alta entre o modelo original e o recarregado.")

    # 6) Avaliação: teste (principal), validação e treino (este último só para medir generalização)
    print("[6/6] Avaliando (teste, validação e treino) e gerando análises...")
    t0 = time.perf_counter()
    prob_te = carregado.predict(x_te, batch_size=64, verbose=0).ravel()
    tempo_inf = (time.perf_counter() - t0) / max(1, len(x_te)) * 1000.0  # ms por imagem
    prob_val = carregado.predict(x_val, batch_size=64, verbose=0).ravel()
    prob_tr = carregado.predict(x_tr, batch_size=64, verbose=0).ravel()  # modo inferência (sem aumento)

    extras = {"tempo_treino_s": float(tempo_treino), "tempo_inferencia_ms_por_imagem": float(tempo_inf)}
    artefatos = metricas_mod.gerar_artefatos_avaliacao(
        y_te, prob_te, historico, resultados_dir, limiar, nome_modelo="cnn", extras=extras)
    metricas_val = metricas_mod.calcular_metricas(y_val, prob_val, limiar)
    metricas_mod.salvar_metricas(metricas_val, resultados_dir, nome_modelo="cnn_validacao")

    def _tabela(sub, prob, conjunto):
        t = pd.DataFrame({"conjunto": conjunto, "arquivo": sub["arquivo"], "classe_real": sub["classe"],
                          "rotulo_real": sub["rotulo"], "prob_fire": prob,
                          "rotulo_previsto": (prob >= limiar).astype(int)})
        t["classe_prevista"] = t["rotulo_previsto"].map(dict(enumerate(dados.CLASSES)))
        return t

    todos = pd.concat([_tabela(sub_tr, prob_tr, "train"), _tabela(sub_val, prob_val, "val"),
                       _tabela(sub_te, prob_te, "test")], ignore_index=True)
    todos.to_csv(resultados_dir / "predicoes_todos_conjuntos.csv", index=False)
    pred = todos[todos["conjunto"] == "test"].drop(columns="conjunto")
    pred.to_csv(resultados_dir / "predicoes_teste.csv", index=False)

    config = {
        "modelo": "cnn_fire_nofire", "arquivo_modelo": modelo_saida.name,
        "tamanho_imagem": tamanho, "classes": list(dados.CLASSES), "classe_positiva": "fire",
        "limiar": limiar, "normalizacao": "pixel/255 -> [0, 1]",
        "semente": semente, "batch": batch, "epocas_maximas": epocas,
        "epocas_executadas": int(len(historico)), "melhor_epoca": melhor_epoca,
        "parametros": int(modelo.count_params()), "tensorflow": tf.__version__,
        "proporcoes_treino_val_teste": list(dados.PROPORCOES), "smoke": bool(smoke),
        "aumento_de_dados": bool(aumento), "tempo_treino_s": float(tempo_treino),
        "tempo_inferencia_ms_por_imagem": float(tempo_inf),
    }
    config_saida.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (resultados_dir / "resumo_execucao.json").write_text(json.dumps(
        {"config": config, "dataset": resumo, "verificacao_vazamento": verificacao},
        indent=2, ensure_ascii=False), encoding="utf-8")

    analises = analise.executar_analises(todos, historico, resultados_dir, data_dir, config, limiar, extras)
    print("Generalização:", analises["generalizacao"]["diagnostico"])

    m = artefatos["metricas"]
    print("\n=== RESULTADOS NO CONJUNTO DE TESTE (imagens não usadas no treino) ===")
    print(f"accuracy : {m['accuracy']:.4f}\nprecision: {m['precision']:.4f}\n"
          f"recall   : {m['recall']:.4f}\nf1-score : {m['f1_score']:.4f}")
    print("matriz de confusão [linhas=real, colunas=previsto; ordem: nofire, fire]:")
    print(np.array(m["matriz_confusao"]))
    print(f"\nArquivos gerados em: {resultados_dir}")
    return {"metricas_teste": m, "metricas_validacao": metricas_val, "analises": analises,
            "config": config, "resultados_dir": resultados_dir, "modelo": modelo_saida}


def main(argv=None):
    p = argparse.ArgumentParser(description="Treina e avalia a CNN fire/nofire.")
    p.add_argument("--epocas", type=int, default=40, help="máximo de épocas (padrão 40)")
    p.add_argument("--batch", type=int, default=32, help="tamanho do lote (padrão 32)")
    p.add_argument("--tamanho", type=int, default=dados.TAMANHO_PADRAO,
                   help="lado da imagem em pixels (padrão 128; use 96 em PC fraco)")
    p.add_argument("--semente", type=int, default=dados.SEMENTE)
    p.add_argument("--paciencia", type=int, default=8, help="paciência do EarlyStopping")
    p.add_argument("--limiar", type=float, default=0.5, help="limiar de decisão para 'fire'")
    p.add_argument("--data-dir", default=str(dados.DATA_DIR))
    p.add_argument("--modelo-saida", default=str(MODELO_PADRAO))
    p.add_argument("--resultados-dir", default=str(RESULTADOS_PADRAO))
    p.add_argument("--sem-aumento", action="store_true",
                   help="experimento de ablação: treina sem aumento de dados (use com --resultados-dir e --modelo-saida diferentes)")
    p.add_argument("--smoke", action="store_true",
                   help="teste rápido: poucas imagens, 2 épocas, grava em pasta temporária")
    a = p.parse_args(argv)

    if a.smoke:
        tmp = Path(tempfile.mkdtemp(prefix="cnn_smoke_"))
        print(f"[SMOKE] teste rápido; saídas em {tmp} (não altera models/ nem results/)")
        treinar(epocas=2, batch=16, tamanho=min(a.tamanho, 64), semente=a.semente,
                paciencia=a.paciencia, limiar=a.limiar, data_dir=a.data_dir,
                modelo_saida=tmp / "cnn_smoke.keras", resultados_dir=tmp / "results", smoke=True,
                aumento=not a.sem_aumento)
    else:
        treinar(epocas=a.epocas, batch=a.batch, tamanho=a.tamanho, semente=a.semente,
                paciencia=a.paciencia, limiar=a.limiar, data_dir=a.data_dir,
                modelo_saida=a.modelo_saida, resultados_dir=a.resultados_dir,
                aumento=not a.sem_aumento)


if __name__ == "__main__":
    main()
