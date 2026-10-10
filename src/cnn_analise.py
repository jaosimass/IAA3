"""Análises complementares da CNN (seções 14, 17, 19 e 20 do PDF da atividade).

Sem dependência do TensorFlow: trabalha a partir das probabilidades salvas pelo
treino em results/cnn/predicoes_todos_conjuntos.csv. Pode ser executado depois
do treino, sem retreinar:

    python src/cnn_analise.py                 # regenera exemplos, generalização, viés e relatório

Gera em results/cnn/:
    exemplos_acertos.png, exemplos_erros.png     imagens reais de teste (seção 14/17)
    exemplos_selecionados.csv                    quais imagens aparecem nas figuras
    analise_generalizacao.json/.png              treino x validação x teste (seção 19)
    analise_luminosidade.csv                     erros por faixa de luminosidade (seção 20, viés)
    relatorio_cnn.md                             relatório preenchido SÓ com valores calculados
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cnn_dados as dados  # noqa: E402
import cnn_metricas as met  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (backend Agg já definido em cnn_metricas)

RESULTADOS_PADRAO = dados.RAIZ_PROJETO / "results" / "cnn"

# Limiares HEURÍSTICOS (convenção, não regra universal) para o diagnóstico da lacuna treino-teste.
LACUNA_PEQUENA = 0.03
LACUNA_ALTA = 0.08
ACC_TREINO_BAIXA = 0.80


# --------------------------------------------------------------------------- #
# Exemplos de acertos e erros
# --------------------------------------------------------------------------- #
def _tipo_resultado(real: int, previsto: int) -> str:
    if real == 1 and previsto == 1:
        return "VP"  # fire corretamente detectado
    if real == 0 and previsto == 0:
        return "VN"  # nofire corretamente reconhecido
    if real == 0 and previsto == 1:
        return "FP"  # alarme falso
    return "FN"      # incêndio não detectado


def selecionar_exemplos(pred: pd.DataFrame, n_por_tipo: int = 8, semente: int = dados.SEMENTE) -> pd.DataFrame:
    """Escolhe imagens de TESTE para ilustrar VP, VN, FP e FN.

    - Acertos (VP/VN): amostra aleatória reprodutível (não só os mais fáceis).
    - Erros (FP/FN): os erros mais "confiantes" primeiro (maior probabilidade atribuída à classe errada).
    Espera as colunas: arquivo, rotulo_real, prob_fire, rotulo_previsto.
    """
    p = pred.copy()
    p["tipo"] = [_tipo_resultado(int(r), int(q)) for r, q in zip(p["rotulo_real"], p["rotulo_previsto"])]
    p["confianca_na_previsao"] = np.where(p["rotulo_previsto"] == 1, p["prob_fire"], 1 - p["prob_fire"])
    partes = []
    for tipo in ("VP", "VN"):
        sub = p[p["tipo"] == tipo]
        partes.append(sub.sample(n=min(n_por_tipo, len(sub)), random_state=semente))
    for tipo in ("FP", "FN"):
        sub = p[p["tipo"] == tipo].sort_values("confianca_na_previsao", ascending=False)
        partes.append(sub.head(n_por_tipo))
    return pd.concat(partes).reset_index(drop=True)


_NOMES_TIPO = {
    "VP": "Acerto: incêndio detectado (verdadeiro positivo)",
    "VN": "Acerto: sem incêndio reconhecido (verdadeiro negativo)",
    "FP": "Erro: alarme falso (falso positivo: era nofire, previsto fire)",
    "FN": "Erro: incêndio não detectado (falso negativo: era fire, previsto nofire)",
}


def _grade_de_imagens(exemplos: pd.DataFrame, tipos, data_dir, caminho, titulo_geral, colunas=4, lado=3.0):
    """Uma sub-figura por tipo (cabeçalho próprio) com uma grade de imagens reais."""
    secoes = [(t, exemplos[exemplos["tipo"] == t]) for t in tipos]
    linhas_por_secao = [max(1, int(np.ceil(len(df) / colunas))) for _, df in secoes]
    altura = sum(n * (lado + 0.7) + 0.6 for n in linhas_por_secao) + 0.8
    fig = plt.figure(figsize=(colunas * lado, altura), layout="constrained")
    fig.suptitle(titulo_geral, fontsize=13)
    subfigs = fig.subfigures(len(secoes), 1, height_ratios=[n + 0.4 for n in linhas_por_secao])
    subfigs = np.atleast_1d(subfigs)
    for sf, (tipo, df), n_linhas in zip(subfigs, secoes, linhas_por_secao):
        sf.suptitle(_NOMES_TIPO[tipo], fontsize=10, fontweight="bold", x=0.01, ha="left")
        eixos = sf.subplots(n_linhas, colunas, squeeze=False)
        for k in range(n_linhas * colunas):
            ax = eixos[k // colunas][k % colunas]
            ax.axis("off")
            if k < len(df):
                r = df.iloc[k]
                with Image.open(Path(data_dir) / r["arquivo"]) as img:
                    ax.imshow(img.convert("RGB"))
                real = dados.CLASSES[int(r["rotulo_real"])]
                prev = dados.CLASSES[int(r["rotulo_previsto"])]
                ax.set_title(f"{Path(r['arquivo']).name}\nreal={real} | prev={prev}\nP(fire)={r['prob_fire']:.3f}",
                             fontsize=8)
            elif k == 0:
                ax.text(0.5, 0.5, "nenhum caso\nnesta categoria", ha="center", va="center", fontsize=9)
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def gerar_figuras_exemplos(pred_teste: pd.DataFrame, data_dir, pasta, n_por_tipo: int = 8) -> dict:
    """Salva exemplos_acertos.png e exemplos_erros.png com imagens reais do conjunto de teste."""
    pasta = Path(pasta)
    ex = selecionar_exemplos(pred_teste, n_por_tipo)
    ex.to_csv(pasta / "exemplos_selecionados.csv", index=False)
    acertos = _grade_de_imagens(ex, ("VP", "VN"), data_dir, pasta / "exemplos_acertos.png",
                                "Exemplos de imagens de teste classificadas CORRETAMENTE")
    erros = _grade_de_imagens(ex, ("FP", "FN"), data_dir, pasta / "exemplos_erros.png",
                              "Exemplos de imagens de teste classificadas INCORRETAMENTE")
    return {"acertos": acertos, "erros": erros, "contagem": ex["tipo"].value_counts().to_dict()}


# --------------------------------------------------------------------------- #
# Generalização (treino x validação x teste)
# --------------------------------------------------------------------------- #
def diagnosticar_generalizacao(acc_treino: float, acc_val: float, acc_teste: float) -> str:
    lacuna_teste = acc_treino - acc_teste
    lacuna_val = acc_treino - acc_val
    if acc_treino < ACC_TREINO_BAIXA:
        return ("Acurácia de treino baixa: possível subajuste (o modelo ainda não aprendeu bem os padrões "
                "nem nos dados de treino).")
    if max(lacuna_teste, lacuna_val) > LACUNA_ALTA:
        return ("Lacuna treino-teste alta: indício de sobreajuste (overfitting): bom desempenho no treino "
                "que não se mantém em imagens novas.")
    if max(lacuna_teste, lacuna_val) > LACUNA_PEQUENA:
        return "Lacuna treino-teste moderada: algum sobreajuste, mas a generalização ainda é razoável."
    return "Lacuna treino-teste pequena: sem sinal relevante de sobreajuste nesta execução."


def analisar_generalizacao(m_treino: dict, m_val: dict, m_teste: dict, historico, pasta) -> dict:
    """Compara métricas nos três conjuntos e salva analise_generalizacao.json e .png."""
    pasta = Path(pasta)
    linhas = {}
    for nome, m in (("treino", m_treino), ("validacao", m_val), ("teste", m_teste)):
        linhas[nome] = {k: m[k] for k in ("accuracy", "precision", "recall", "f1_score", "log_loss", "n_amostras")}
    h = pd.DataFrame(historico) if historico is not None else pd.DataFrame()
    extra = {}
    if len(h) and "val_loss" in h:
        melhor = int(np.argmin(h["val_loss"].to_numpy())) + 1
        extra = {"epocas_executadas": int(len(h)), "melhor_epoca_val_loss": melhor,
                 "val_loss_minima": float(h["val_loss"].min()),
                 "val_loss_na_ultima_epoca": float(h["val_loss"].iloc[-1])}
    resultado = {
        "metricas_por_conjunto": linhas,
        "lacuna_accuracy_treino_menos_teste": linhas["treino"]["accuracy"] - linhas["teste"]["accuracy"],
        "lacuna_accuracy_treino_menos_validacao": linhas["treino"]["accuracy"] - linhas["validacao"]["accuracy"],
        "diagnostico": diagnosticar_generalizacao(linhas["treino"]["accuracy"], linhas["validacao"]["accuracy"],
                                                  linhas["teste"]["accuracy"]),
        "limiares_heuristicos": {"lacuna_pequena": LACUNA_PEQUENA, "lacuna_alta": LACUNA_ALTA,
                                 "acc_treino_baixa": ACC_TREINO_BAIXA},
        "historico": extra,
    }
    (pasta / "analise_generalizacao.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    nomes = ["treino", "validacao", "teste"]
    larg = 0.2
    for i, chave in enumerate(("accuracy", "precision", "recall", "f1_score")):
        ax.bar(np.arange(3) + (i - 1.5) * larg, [linhas[n][chave] for n in nomes], larg, label=chave)
    ax.set_xticks(range(3), labels=nomes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("valor")
    ax.set_title("Treino x validação x teste")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(pasta / "analise_generalizacao.png", dpi=150)
    plt.close(fig)
    return resultado


# --------------------------------------------------------------------------- #
# Possível viés: desempenho por faixa de luminosidade
# --------------------------------------------------------------------------- #
def luminosidade_media(origem) -> float:
    """Luminosidade média (mesma fórmula de src/data_processing.py): 0,299 R + 0,587 G + 0,114 B."""
    with Image.open(origem) as img:
        a = np.asarray(img.convert("RGB"), dtype=np.float64)
    return float(np.mean(0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]))


def analisar_por_luminosidade(pred_teste: pd.DataFrame, data_dir, pasta, faixas: int = 3) -> pd.DataFrame:
    """Taxas de acerto, falsos positivos e falsos negativos por faixa (quantis) de luminosidade.

    Serve para verificar se o modelo erra mais em imagens escuras/claras (ex.: cenas noturnas,
    pôr do sol). É uma verificação de viés SIMPLES: não cobre outras variáveis (região, estação,
    tipo de vegetação), que o dataset não traz.
    """
    p = pred_teste.copy()
    p["luminosidade"] = [luminosidade_media(Path(data_dir) / a) for a in p["arquivo"]]
    p["faixa"] = pd.qcut(p["luminosidade"], q=faixas, duplicates="drop")
    linhas = []
    for faixa, g in p.groupby("faixa", observed=True):
        real, prev = g["rotulo_real"].to_numpy(), g["rotulo_previsto"].to_numpy()
        n_nofire, n_fire = int((real == 0).sum()), int((real == 1).sum())
        linhas.append({
            "faixa_luminosidade": f"{faixa.left:.1f} a {faixa.right:.1f}",
            "n_imagens": int(len(g)), "n_fire": n_fire, "n_nofire": n_nofire,
            "accuracy": float((real == prev).mean()),
            "falsos_positivos": int(((real == 0) & (prev == 1)).sum()),
            "falsos_negativos": int(((real == 1) & (prev == 0)).sum()),
            "taxa_falsos_positivos": float(((real == 0) & (prev == 1)).sum() / n_nofire) if n_nofire else np.nan,
            "taxa_falsos_negativos": float(((real == 1) & (prev == 0)).sum() / n_fire) if n_fire else np.nan,
        })
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(Path(pasta) / "analise_luminosidade.csv", index=False)
    return tabela


# --------------------------------------------------------------------------- #
# Relatório automático (só valores calculados)
# --------------------------------------------------------------------------- #
def _md_tabela(df: pd.DataFrame, formato="{:.4f}") -> str:
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    corpo = ["| " + " | ".join(formato.format(v) if isinstance(v, float) else str(v) for v in linha) + " |"
             for linha in df.itertuples(index=False)]
    return "\n".join([cab, sep, *corpo])


def gerar_relatorio_markdown(m_teste: dict, generalizacao: dict, luminosidade: pd.DataFrame,
                             contagem_exemplos: dict, config: dict | None, pasta) -> Path:
    """Escreve results/cnn/relatorio_cnn.md com os números reais da execução."""
    pasta = Path(pasta)
    cm = m_teste["matriz_confusao"]
    g = generalizacao["metricas_por_conjunto"]
    tab_gen = pd.DataFrame([{"conjunto": n, **{k: v[k] for k in ("n_amostras", "accuracy", "precision", "recall",
                                                               "f1_score", "log_loss")}} for n, v in g.items()])
    L = []
    L.append("# Relatório automático da CNN (fire / nofire)\n")
    L.append("Gerado por `src/cnn_analise.py` a partir dos resultados desta execução. "
             "Todos os números abaixo foram calculados, nenhum foi digitado à mão.\n")
    if config:
        L.append(f"- Imagem de entrada: {config.get('tamanho_imagem')}x{config.get('tamanho_imagem')} RGB; "
                 f"parâmetros: {config.get('parametros')}; épocas executadas: {config.get('epocas_executadas')} "
                 f"(melhor época: {config.get('melhor_epoca')}); semente: {config.get('semente')}; "
                 f"aumento de dados: {config.get('aumento_de_dados')}; TensorFlow: {config.get('tensorflow')}.\n")
    L.append("## Métricas no conjunto de teste\n")
    L.append(f"Limiar de decisão: {m_teste['limiar']}; classe positiva: `fire`; amostras: {m_teste['n_amostras']}.\n")
    L.append(_md_tabela(pd.DataFrame([{k: m_teste[k] for k in (
        "accuracy", "precision", "recall", "f1_score", "taxa_falsos_positivos", "taxa_falsos_negativos")}])) + "\n")
    L.append("Matriz de confusão (linhas = real, colunas = previsto; ordem: nofire, fire):\n")
    L.append(f"|  | previsto nofire | previsto fire |\n|---|---|---|\n| real nofire | {cm[0][0]} | {cm[0][1]} |\n"
             f"| real fire | {cm[1][0]} | {cm[1][1]} |\n")
    L.append(f"Falsos positivos (alarmes falsos): **{m_teste['fp']}**. "
             f"Falsos negativos (incêndios não detectados): **{m_teste['fn']}**.\n")
    if "tempo_treino_s" in m_teste:
        L.append(f"Tempo de treino: {m_teste['tempo_treino_s']:.1f} s; "
                 f"inferência: {m_teste['tempo_inferencia_ms_por_imagem']:.2f} ms/imagem (CPU/GPU da máquina que executou).\n")
    L.append("## Generalização (treino x validação x teste)\n")
    L.append(_md_tabela(tab_gen) + "\n")
    L.append(f"- Lacuna de accuracy treino - teste: {generalizacao['lacuna_accuracy_treino_menos_teste']:.4f}; "
             f"treino - validação: {generalizacao['lacuna_accuracy_treino_menos_validacao']:.4f}.\n"
             f"- Diagnóstico (limiares heuristicos {generalizacao['limiares_heuristicos']}): "
             f"{generalizacao['diagnostico']}\n")
    L.append("- `log_loss` aqui não inclui a regularização L2, por isso difere da `loss` exibida pelo Keras.\n")
    L.append("## Exemplos (imagens reais do teste)\n")
    L.append(f"Quantidade de exemplos mostrados por tipo: {contagem_exemplos}. "
             "Figuras: `exemplos_acertos.png` e `exemplos_erros.png` (os erros mostrados são os de maior confiança).\n")
    L.append("## Desempenho por faixa de luminosidade (verificação simples de viés)\n")
    L.append(_md_tabela(luminosidade) + "\n")
    L.append("Faixas com poucas imagens geram taxas instáveis; use como indício, não como prova.\n")
    L.append("## Arquivos\n")
    L.append("`matriz_confusao.png`, `historico_treinamento.png`, `analise_generalizacao.png`, "
             "`exemplos_acertos.png`, `exemplos_erros.png`, `metricas_cnn.json/.csv`, "
             "`predicoes_todos_conjuntos.csv`, `divisao_dados.csv`.\n")
    caminho = pasta / "relatorio_cnn.md"
    caminho.write_text("\n".join(L), encoding="utf-8")
    return caminho


# --------------------------------------------------------------------------- #
# Orquestração a partir de predições salvas
# --------------------------------------------------------------------------- #
def executar_analises(pred_todos: pd.DataFrame, historico, pasta, data_dir=dados.DATA_DIR,
                      config: dict | None = None, limiar: float = 0.5, extras_teste: dict | None = None) -> dict:
    """Roda exemplos, generalização, luminosidade e relatório a partir das predições salvas."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    metricas = {}
    for conj, nome in (("train", "treino"), ("val", "validacao"), ("test", "teste")):
        sub = pred_todos[pred_todos["conjunto"] == conj]
        if sub.empty:
            raise ValueError(f"Sem predições para o conjunto '{conj}'.")
        metricas[nome] = met.calcular_metricas(sub["rotulo_real"], sub["prob_fire"], limiar)
    if extras_teste:
        metricas["teste"].update(extras_teste)
    teste = pred_todos[pred_todos["conjunto"] == "test"].reset_index(drop=True)
    exemplos = gerar_figuras_exemplos(teste, data_dir, pasta)
    ger = analisar_generalizacao(metricas["treino"], metricas["validacao"], metricas["teste"], historico, pasta)
    lum = analisar_por_luminosidade(teste, data_dir, pasta)
    relatorio = gerar_relatorio_markdown(metricas["teste"], ger, lum, exemplos["contagem"], config, pasta)
    return {"metricas": metricas, "exemplos": exemplos, "generalizacao": ger,
            "luminosidade": lum, "relatorio": relatorio}


def main(argv=None):
    p = argparse.ArgumentParser(description="Regenera as análises da CNN a partir de results/cnn (sem TensorFlow).")
    p.add_argument("--resultados-dir", default=str(RESULTADOS_PADRAO),
                   help="pasta com predicoes_todos_conjuntos.csv (gerada pelo treino)")
    p.add_argument("--data-dir", default=str(dados.DATA_DIR))
    p.add_argument("--limiar", type=float, default=None,
                   help="recalcula tudo com outro limiar de decisão, SEM retreinar (padrão: o usado no treino)")
    p.add_argument("--saida-dir", default=None,
                   help="onde gravar as análises (padrão: a própria pasta de resultados; use outra para não sobrescrever)")
    a = p.parse_args(argv)
    pasta = Path(a.resultados_dir)
    saida = Path(a.saida_dir) if a.saida_dir else pasta
    arq = pasta / "predicoes_todos_conjuntos.csv"
    if not arq.is_file():
        raise FileNotFoundError(f"{arq} não existe. Treine antes: python src/cnn_treino.py")
    hist = pasta / "historico_treinamento.csv"
    cfg = pasta / "resumo_execucao.json"
    config = json.loads(cfg.read_text(encoding="utf-8")).get("config") if cfg.is_file() else None
    limiar = a.limiar if a.limiar is not None else (config or {}).get("limiar", 0.5)
    todos = pd.read_csv(arq)
    todos["rotulo_previsto"] = (todos["prob_fire"] >= limiar).astype(int)  # reaplica o limiar às probabilidades
    mj = pasta / "metricas_cnn.json"
    extras = None
    if mj.is_file():
        antigo = json.loads(mj.read_text(encoding="utf-8"))
        extras = {k: antigo[k] for k in ("tempo_treino_s", "tempo_inferencia_ms_por_imagem") if k in antigo}
    r = executar_analises(todos, pd.read_csv(hist) if hist.is_file() else None,
                          saida, a.data_dir, config, limiar, extras)
    print(f"Limiar usado: {limiar}")
    print("Diagnóstico de generalização:", r["generalizacao"]["diagnostico"])
    print("Relatório:", r["relatorio"])


if __name__ == "__main__":
    main()
