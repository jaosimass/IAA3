"""Testes das análises (exemplos de acertos/erros, generalização, luminosidade, relatório).

Sem TensorFlow. As probabilidades usadas aqui são SINTÉTICAS (só para testar o código, em pasta
temporária); os resultados reais da CNN vêm do treino (src/cnn_treino.py).
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cnn_analise as a  # noqa: E402
import cnn_dados as d  # noqa: E402

DATASET_OK = (d.DATA_DIR / "fire").is_dir() and (d.DATA_DIR / "nofire").is_dir()


def _pred_sintetica(arquivos, rotulos, probs):
    probs = np.asarray(probs, dtype=float)
    return pd.DataFrame({"arquivo": arquivos, "rotulo_real": rotulos, "prob_fire": probs,
                         "rotulo_previsto": (probs >= 0.5).astype(int)})


class TestSelecaoEDiagnostico(unittest.TestCase):
    def test_selecionar_exemplos_tipos_e_ordem_dos_erros(self):
        rotulos = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        probs = [0.9, 0.8, 0.4, 0.1, 0.1, 0.2, 0.7, 0.95, 0.05, 0.3]
        df = _pred_sintetica([f"x{i}.jpg" for i in range(10)], rotulos, probs)
        ex = a.selecionar_exemplos(df, n_por_tipo=2)
        contagem = ex["tipo"].value_counts().to_dict()
        self.assertEqual(contagem, {"VP": 2, "VN": 2, "FP": 2, "FN": 2})
        fp = ex[ex["tipo"] == "FP"]["arquivo"].tolist()
        self.assertEqual(fp, ["x7.jpg", "x6.jpg"])     # o erro mais confiante (0.95) vem primeiro
        fn = ex[ex["tipo"] == "FN"]["arquivo"].tolist()
        self.assertEqual(fn, ["x3.jpg", "x2.jpg"])     # P(fire)=0.1 é mais confiante que 0.4

    def test_diagnostico_generalizacao(self):
        self.assertIn("pequena", a.diagnosticar_generalizacao(0.95, 0.94, 0.94))
        self.assertIn("moderada", a.diagnosticar_generalizacao(0.97, 0.95, 0.93))
        self.assertIn("sobreajuste", a.diagnosticar_generalizacao(0.99, 0.90, 0.70))
        self.assertIn("subajuste", a.diagnosticar_generalizacao(0.60, 0.58, 0.57))


class TestLuminosidade(unittest.TestCase):
    def test_faixas_com_imagens_de_luminosidade_conhecida(self):
        with tempfile.TemporaryDirectory() as tmp:
            nomes, rot, prob = [], [], []
            for nivel, base in (("escura", 20), ("media", 128), ("clara", 235)):
                for k in range(3):
                    nome = f"{nivel}{k}.png"
                    Image.fromarray(np.full((16, 16, 3), base + k, np.uint8)).save(Path(tmp) / nome)
                    nomes.append(nome)
                    rot.append(1 if k < 2 else 0)
                    prob.append(0.9 if k == 0 else (0.1 if k == 1 else 0.8))
            # escura/media/clara: k=0 acerta (fire,0.9); k=1 erra (fire,0.1 -> FN); k=2 erra (nofire,0.8 -> FP)
            pred = _pred_sintetica(nomes, rot, prob)
            tab = a.analisar_por_luminosidade(pred, tmp, tmp, faixas=3)
            self.assertEqual(len(tab), 3)
            self.assertEqual(tab["n_imagens"].tolist(), [3, 3, 3])
            self.assertTrue(np.allclose(tab["accuracy"], 1 / 3))
            self.assertEqual(tab["falsos_positivos"].tolist(), [1, 1, 1])
            self.assertEqual(tab["falsos_negativos"].tolist(), [1, 1, 1])
            self.assertTrue((Path(tmp) / "analise_luminosidade.csv").is_file())


@unittest.skipUnless(DATASET_OK, "dataset não encontrado")
class TestFluxoCompletoComImagensReais(unittest.TestCase):
    def test_figuras_generalizacao_e_relatorio(self):
        df = d.listar_imagens()
        rng = np.random.default_rng(0)
        amostra = pd.concat([df[df["classe"] == "fire"].sample(60, random_state=0),
                             df[df["classe"] == "nofire"].sample(60, random_state=0)]).reset_index(drop=True)
        amostra["conjunto"] = ["train", "val", "test"] * 40
        # probabilidades sintéticas: majoritariamente corretas, com alguns erros em cada tipo
        prob = np.where(amostra["rotulo"] == 1, rng.uniform(0.55, 1, len(amostra)), rng.uniform(0, 0.45, len(amostra)))
        flip = rng.random(len(amostra)) < 0.12
        prob = np.where(flip, 1 - prob, prob)
        todos = pd.DataFrame({"conjunto": amostra["conjunto"], "arquivo": amostra["arquivo"],
                              "rotulo_real": amostra["rotulo"], "prob_fire": prob,
                              "rotulo_previsto": (prob >= 0.5).astype(int)})
        hist = pd.DataFrame({"loss": [0.7, 0.5], "val_loss": [0.7, 0.6], "accuracy": [0.5, 0.8], "val_accuracy": [0.5, 0.7]})
        with tempfile.TemporaryDirectory() as tmp:
            r = a.executar_analises(todos, hist, tmp, d.DATA_DIR, config={"tamanho_imagem": 128},
                                    extras_teste={"tempo_treino_s": 1.0, "tempo_inferencia_ms_por_imagem": 2.0})
            for nome in ("exemplos_acertos.png", "exemplos_erros.png", "exemplos_selecionados.csv",
                         "analise_generalizacao.json", "analise_generalizacao.png",
                         "analise_luminosidade.csv", "relatorio_cnn.md"):
                arq = Path(tmp) / nome
                self.assertTrue(arq.is_file(), nome)
                self.assertGreater(arq.stat().st_size, 0, nome)
            texto = (Path(tmp) / "relatorio_cnn.md").read_text(encoding="utf-8")
            self.assertIn("Matriz de confusão", texto)
            self.assertIn("Diagnóstico", texto)
            self.assertGreater(len(r["exemplos"]["contagem"]), 0)

    def test_figura_de_erros_sem_nenhum_erro_nao_quebra(self):
        df = d.listar_imagens().groupby("classe").head(6).reset_index(drop=True)
        prob = np.where(df["rotulo"] == 1, 0.99, 0.01)  # classificador "perfeito"
        pred = _pred_sintetica(df["arquivo"], df["rotulo"], prob)
        with tempfile.TemporaryDirectory() as tmp:
            out = a.gerar_figuras_exemplos(pred, d.DATA_DIR, tmp, n_por_tipo=3)
            self.assertTrue(out["erros"].is_file())
            self.assertNotIn("FP", out["contagem"])


@unittest.skipUnless(DATASET_OK, "dataset não encontrado")
class TestCLIAnalise(unittest.TestCase):
    def test_main_com_limiar_diferente_e_saida_separada(self):
        df = d.listar_imagens()
        amostra = pd.concat([df[df["classe"] == "fire"].head(30), df[df["classe"] == "nofire"].head(30)]).reset_index(drop=True)
        rng = np.random.default_rng(3)
        prob = np.where(amostra["rotulo"] == 1, rng.uniform(0.3, 1, 60), rng.uniform(0, 0.7, 60))  # SINTÉTICO
        todos = pd.DataFrame({"conjunto": ["train", "val", "test"] * 20, "arquivo": amostra["arquivo"],
                              "classe_real": amostra["classe"], "rotulo_real": amostra["rotulo"], "prob_fire": prob,
                              "rotulo_previsto": (prob >= 0.5).astype(int)})
        with tempfile.TemporaryDirectory() as tmp:
            res, out = Path(tmp) / "res", Path(tmp) / "out"
            res.mkdir()
            todos.to_csv(res / "predicoes_todos_conjuntos.csv", index=False)
            a.main(["--resultados-dir", str(res), "--limiar", "0.2", "--saida-dir", str(out)])
            self.assertTrue((out / "relatorio_cnn.md").is_file())
            self.assertFalse((res / "relatorio_cnn.md").exists())  # não sobrescreve a pasta original
            import json
            ger = json.loads((out / "analise_generalizacao.json").read_text(encoding="utf-8"))
            self.assertEqual(ger["metricas_por_conjunto"]["teste"]["n_amostras"], 20)
            # limiar baixo (0.2) deve prever fire com mais frequência que 0.5: recall não pode cair
            r02 = a.met.calcular_metricas(todos[todos["conjunto"] == "test"]["rotulo_real"],
                                          todos[todos["conjunto"] == "test"]["prob_fire"], 0.2)["recall"]
            r05 = a.met.calcular_metricas(todos[todos["conjunto"] == "test"]["rotulo_real"],
                                          todos[todos["conjunto"] == "test"]["prob_fire"], 0.5)["recall"]
            self.assertGreaterEqual(r02, r05)


if __name__ == "__main__":
    unittest.main()
