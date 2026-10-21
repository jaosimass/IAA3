"""Métricas e gráficos de avaliação da CNN (sem dependência do TensorFlow).

Classe positiva: fire (rótulo 1). Ordem das classes na matriz de confusão:
[nofire, fire] -> linhas = classe real, colunas = classe prevista:

                 previsto nofire   previsto fire
    real nofire        TN                FP
    real fire          FN                TP

O formato do JSON/CSV gerado é simples e genérico (accuracy, precision, recall,
f1_score, matriz de confusão) para facilitar a comparação futura com os outros
modelos do grupo.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de janela/interface gráfica
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

CLASSES = ("nofire", "fire")  # índice = rótulo


def calcular_metricas(y_true, y_prob, limiar: float = 0.5) -> dict:
    """Calcula as métricas a partir dos rótulos reais e das probabilidades de 'fire'."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) != len(y_prob):
        raise ValueError("y_true e y_prob devem ter o mesmo tamanho.")
    if len(y_true) == 0:
        raise ValueError("Nenhuma amostra para avaliar.")
    y_pred = (y_prob >= limiar).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)

    p_clip = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return {
        "classe_positiva": "fire",
        "limiar": float(limiar),
        "n_amostras": int(len(y_true)),
        # log_loss = entropia cruzada binária SEM o termo de regularização L2 (a "loss" do Keras o inclui)
        "log_loss": float(log_loss(y_true, p_clip, labels=[0, 1])),
        "taxa_falsos_positivos": float(fp / (fp + tn)) if (fp + tn) else 0.0,  # nofire previsto como fire
        "taxa_falsos_negativos": float(fn / (fn + tp)) if (fn + tp) else 0.0,  # fire previsto como nofire
        "especificidade": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec[1]),
        "recall": float(rec[1]),
        "f1_score": float(f1[1]),
        "ordem_classes": list(CLASSES),
        "matriz_confusao": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "por_classe": {
            CLASSES[i]: {"precision": float(prec[i]), "recall": float(rec[i]),
                         "f1_score": float(f1[i]), "suporte": int(sup[i])}
            for i in (0, 1)
        },
        "macro": {"precision": float(prec.mean()), "recall": float(rec.mean()),
                  "f1_score": float(f1.mean())},
    }


def relatorio_texto(y_true, y_prob, limiar: float = 0.5) -> str:
    y_pred = (np.asarray(y_prob, dtype=float).ravel() >= limiar).astype(int)
    return classification_report(np.asarray(y_true).astype(int).ravel(), y_pred,
                                 labels=[0, 1], target_names=list(CLASSES),
                                 digits=4, zero_division=0)


def plotar_matriz_confusao(cm, caminho: Path | str, titulo: str = "Matriz de confusão (teste)") -> Path:
    """Salva a matriz de confusão (contagens e % por linha) como PNG."""
    cm = np.asarray(cm, dtype=int)
    soma = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, soma, out=np.zeros_like(cm, dtype=float), where=soma > 0)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=[f"previsto\n{c}" for c in CLASSES])
    ax.set_yticks([0, 1], labels=[f"real\n{c}" for c in CLASSES])
    limite = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j] * 100:.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > limite else "black", fontsize=11)
    ax.set_title(titulo)
    fig.tight_layout()
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def plotar_historico(historico: pd.DataFrame | dict, caminho: Path | str,
                     titulo: str = "Histórico de treinamento") -> Path:
    """Salva as curvas de loss e accuracy (treino x validação) como PNG."""
    h = pd.DataFrame(historico)
    epocas = np.arange(1, len(h) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, chave, rotulo in ((ax1, "loss", "Loss"), (ax2, "accuracy", "Acurácia")):
        if chave in h:
            ax.plot(epocas, h[chave], label="treino")
        if f"val_{chave}" in h:
            ax.plot(epocas, h[f"val_{chave}"], label="validação")
        ax.set_xlabel("Época")
        ax.set_ylabel(rotulo)
        ax.set_title(rotulo)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho


def salvar_metricas(metricas: dict, pasta: Path | str, nome_modelo: str = "cnn") -> dict:
    """Salva metricas_<modelo>.json e metricas_<modelo>.csv (uma linha por modelo)."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_json = pasta / f"metricas_{nome_modelo}.json"
    caminho_csv = pasta / f"metricas_{nome_modelo}.csv"
    caminho_json.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    colunas = ("accuracy", "precision", "recall", "f1_score", "tn", "fp", "fn", "tp",
               "n_amostras", "limiar", "tempo_treino_s", "tempo_inferencia_ms_por_imagem")
    linha = {"modelo": nome_modelo, **{k: metricas[k] for k in colunas if k in metricas}}
    pd.DataFrame([linha]).to_csv(caminho_csv, index=False)
    return {"json": caminho_json, "csv": caminho_csv}


def gerar_artefatos_avaliacao(y_true, y_prob, historico, pasta: Path | str,
                              limiar: float = 0.5, nome_modelo: str = "cnn",
                              extras: dict | None = None) -> dict:
    """Calcula as métricas de TESTE e salva JSON, CSV, relatório, matriz e histórico.

    `y_true`/`y_prob` devem ser do conjunto de teste (imagens nunca vistas no treino
    nem na validação). `historico` é o histórico por época do treinamento.
    """
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    metricas = calcular_metricas(y_true, y_prob, limiar)
    if extras:  # ex.: tempos medidos de treino e de inferência (seção 15 do PDF)
        metricas.update(extras)
    arquivos = salvar_metricas(metricas, pasta, nome_modelo)
    (pasta / "relatorio_classificacao.txt").write_text(
        relatorio_texto(y_true, y_prob, limiar), encoding="utf-8")
    plotar_matriz_confusao(metricas["matriz_confusao"], pasta / "matriz_confusao.png")
    if historico is not None and len(pd.DataFrame(historico)) > 0:
        h = pd.DataFrame(historico)
        h.index = range(1, len(h) + 1)
        h.to_csv(pasta / "historico_treinamento.csv", index_label="epoca")
        plotar_historico(historico, pasta / "historico_treinamento.png")
    return {"metricas": metricas, "arquivos": arquivos}
