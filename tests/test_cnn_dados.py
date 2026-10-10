"""Testes do carregamento, pré-processamento e divisão sem vazamento (sem TensorFlow)."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cnn_dados as d  # noqa: E402

DATASET_OK = (d.DATA_DIR / "fire").is_dir() and (d.DATA_DIR / "nofire").is_dir()


class TestDivisaoSintetica(unittest.TestCase):
    """Rápidos: não dependem do dataset real."""

    def _df(self, n=400):
        rotulos = np.array([0, 1] * (n // 2))
        return pd.DataFrame({"arquivo": [f"{'fire' if r else 'nofire'}/{i}.jpg" for i, r in enumerate(rotulos)],
                             "classe": ["fire" if r else "nofire" for r in rotulos],
                             "rotulo": rotulos})

    def test_grupos_nunca_sao_divididos_e_proporcoes_ok(self):
        df = self._df()
        grupos = np.arange(len(df))
        # junta 3 imagens "quase iguais" da mesma classe (índices pares = nofire)
        grupos[[10, 12, 14]] = 10
        grupos[[11, 13, 15]] = 11
        m = d.dividir_dados(df, grupos)
        d.verificar_vazamento(m)
        for g in (10, 11):
            self.assertEqual(m.loc[m["grupo"] == g, "conjunto"].nunique(), 1)
        frac = m["conjunto"].value_counts(normalize=True)
        for c, p in zip(d.CONJUNTOS, d.PROPORCOES):
            self.assertAlmostEqual(frac[c], p, delta=0.03)

    def test_deterministico_e_semente_muda_divisao(self):
        df, grupos = self._df(), np.arange(400)
        a = d.dividir_dados(df, grupos, semente=1)["conjunto"].tolist()
        b = d.dividir_dados(df, grupos, semente=1)["conjunto"].tolist()
        c = d.dividir_dados(df, grupos, semente=2)["conjunto"].tolist()
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_verificar_vazamento_detecta_problemas(self):
        m = d.dividir_dados(self._df(), np.arange(400))
        d.verificar_vazamento(m)  # ok
        ruim = m.copy()
        ruim.loc[ruim.index[0], "conjunto"] = "test" if ruim.loc[ruim.index[0], "conjunto"] != "test" else "train"
        ruim.loc[ruim.index[1], "grupo"] = ruim.loc[ruim.index[0], "grupo"]  # grupo dividido
        with self.assertRaises(ValueError):
            d.verificar_vazamento(ruim)
        duplicado = pd.concat([m, m.iloc[[0]].assign(conjunto="val" if m.iloc[0]["conjunto"] != "val" else "test")])
        with self.assertRaises(ValueError):
            d.verificar_vazamento(duplicado)

    def test_agrupar_duplicatas(self):
        base = 0x0F0F0F0F0F0F0F0F
        hashes = [base, base ^ 0b11, 0xFFFFFFFF00000000, base, 0x00000000FFFFFFFF]
        g = d.agrupar_duplicatas(hashes, limiar=4)
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[0], g[3])
        self.assertEqual(len({g[0], g[2], g[4]}), 3)


class TestImagens(unittest.TestCase):
    def test_carregar_imagem_rgb_redimensiona_e_normaliza(self):
        with tempfile.TemporaryDirectory() as tmp:
            # imagem em tons de cinza 'L' e outra RGBA: ambas devem virar RGB
            Image.fromarray(np.full((40, 30), 255, np.uint8), "L").save(Path(tmp) / "a.png")
            Image.fromarray(np.zeros((20, 50, 4), np.uint8), "RGBA").save(Path(tmp) / "b.png")
            for nome in ("a.png", "b.png"):
                x = d.carregar_imagem(Path(tmp) / nome, 32)
                self.assertEqual(x.shape, (32, 32, 3))
                self.assertEqual(x.dtype, np.float32)
                self.assertGreaterEqual(x.min(), 0.0)
                self.assertLessEqual(x.max(), 1.0)
            self.assertAlmostEqual(float(d.carregar_imagem(Path(tmp) / "a.png", 8).mean()), 1.0, places=5)

    def test_dhash_imagem_copiada_tem_distancia_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(0)
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(Path(tmp) / "x.png")
            shutil.copy(Path(tmp) / "x.png", Path(tmp) / "y.png")
            self.assertEqual(d.calcular_dhash(Path(tmp) / "x.png"), d.calcular_dhash(Path(tmp) / "y.png"))


@unittest.skipUnless(DATASET_OK, "dataset em data/raw/g20gli_dataset não encontrado")
class TestDatasetReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = d.listar_imagens()
        cls.manifesto = d.construir_manifesto()

    def test_classes_e_rotulos_vem_das_pastas(self):
        self.assertEqual(set(self.df["classe"]), {"fire", "nofire"})
        self.assertTrue((self.df.loc[self.df["classe"] == "fire", "rotulo"] == 1).all())
        self.assertTrue((self.df.loc[self.df["classe"] == "nofire", "rotulo"] == 0).all())
        self.assertTrue(all(a.startswith(c + "/") for a, c in zip(self.df["arquivo"], self.df["classe"])))
        contagem = self.df["classe"].value_counts()
        self.assertGreater(contagem.min(), 0)
        self.assertEqual(contagem.sum(), len(self.df))

    def test_todas_as_imagens_carregam_com_dimensoes_corretas(self):
        amostra = self.df.groupby("classe").head(5)["arquivo"]
        x = d.carregar_lote(amostra, tamanho=64)
        self.assertEqual(x.shape, (len(amostra), 64, 64, 3))
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), 1.0)

    def test_divisao_sem_vazamento_com_dataset_real(self):
        m = self.manifesto
        d.verificar_vazamento(m)
        self.assertEqual(len(m), len(self.df))
        conjuntos = {c: set(m.loc[m["conjunto"] == c, "arquivo"]) for c in d.CONJUNTOS}
        self.assertFalse(conjuntos["train"] & conjuntos["val"])
        self.assertFalse(conjuntos["train"] & conjuntos["test"])
        self.assertFalse(conjuntos["val"] & conjuntos["test"])
        frac = m["conjunto"].value_counts(normalize=True)
        for c, p in zip(d.CONJUNTOS, d.PROPORCOES):
            self.assertAlmostEqual(frac[c], p, delta=0.02)
        # estratificação: classes equilibradas em cada conjunto
        for c in d.CONJUNTOS:
            sub = m[m["conjunto"] == c]
            self.assertAlmostEqual((sub["rotulo"] == 1).mean(), (m["rotulo"] == 1).mean(), delta=0.03)

    def test_divisao_reprodutivel(self):
        again = d.construir_manifesto()
        self.assertEqual(again["conjunto"].tolist(), self.manifesto["conjunto"].tolist())


if __name__ == "__main__":
    unittest.main()
