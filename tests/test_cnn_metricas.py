"""Testes das métricas e gráficos (sem TensorFlow). Valores conferidos à mão."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cnn_metricas as m  # noqa: E402

# 10 amostras: 4 fire (1) e 6 nofire (0).
Y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
P = [0.9, 0.8, 0.4, 0.6, 0.1, 0.2, 0.7, 0.3, 0.05, 0.5]
# limiar 0.5 -> previsto [1,1,0,1,0,0,1,0,0,1]  => TP=3 FN=1 FP=2 TN=4


class TestMetricas(unittest.TestCase):
    def test_valores_conferidos_a_mao(self):
        r = m.calcular_metricas(Y, P, 0.5)
        self.assertEqual((r["tn"], r["fp"], r["fn"], r["tp"]), (4, 2, 1, 3))
        self.assertEqual(r["matriz_confusao"], [[4, 2], [1, 3]])
        self.assertAlmostEqual(r["accuracy"], 0.7)
        self.assertAlmostEqual(r["precision"], 3 / 5)
        self.assertAlmostEqual(r["recall"], 3 / 4)
        self.assertAlmostEqual(r["f1_score"], 2 * 0.6 * 0.75 / 1.35)
        self.assertEqual(r["por_classe"]["nofire"]["suporte"], 6)
        self.assertAlmostEqual(r["taxa_falsos_positivos"], 2 / 6)
        self.assertAlmostEqual(r["taxa_falsos_negativos"], 1 / 4)
        self.assertAlmostEqual(r["especificidade"], 4 / 6)
        self.assertGreater(r["log_loss"], 0.0)

    def test_extras_aparecem_no_json_e_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            m.gerar_artefatos_avaliacao(Y, P, None, tmp, extras={"tempo_treino_s": 12.5,
                                                                 "tempo_inferencia_ms_por_imagem": 3.2})
            salvo = json.loads((Path(tmp) / "metricas_cnn.json").read_text(encoding="utf-8"))
            self.assertEqual(salvo["tempo_treino_s"], 12.5)
            csv = pd.read_csv(Path(tmp) / "metricas_cnn.csv")
            self.assertEqual(float(csv.loc[0, "tempo_inferencia_ms_por_imagem"]), 3.2)

    def test_sem_previsao_positiva_nao_quebra(self):
        r = m.calcular_metricas([1, 0, 1, 0], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(r["precision"], 0.0)
        self.assertEqual(r["recall"], 0.0)
        self.assertEqual(r["tp"], 0)

    def test_entradas_invalidas(self):
        with self.assertRaises(ValueError):
            m.calcular_metricas([1, 0], [0.5])
        with self.assertRaises(ValueError):
            m.calcular_metricas([], [])

    def test_gera_arquivos(self):
        hist = {"loss": [0.7, 0.5, 0.4], "val_loss": [0.7, 0.6, 0.5],
                "accuracy": [0.5, 0.7, 0.8], "val_accuracy": [0.5, 0.65, 0.75]}
        with tempfile.TemporaryDirectory() as tmp:
            out = m.gerar_artefatos_avaliacao(Y, P, hist, tmp)
            for nome in ("metricas_cnn.json", "metricas_cnn.csv", "matriz_confusao.png",
                         "historico_treinamento.png", "historico_treinamento.csv",
                         "relatorio_classificacao.txt"):
                arq = Path(tmp) / nome
                self.assertTrue(arq.is_file(), nome)
                self.assertGreater(arq.stat().st_size, 0, nome)
            salvo = json.loads((Path(tmp) / "metricas_cnn.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(salvo["accuracy"], 0.7)
            csv = pd.read_csv(Path(tmp) / "metricas_cnn.csv")
            self.assertAlmostEqual(float(csv.loc[0, "f1_score"]), out["metricas"]["f1_score"])
            with open(Path(tmp) / "matriz_confusao.png", "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
