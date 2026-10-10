"""Testes da CNN (arquitetura, treino mínimo, salvar/carregar e previsão).

Exigem TensorFlow; sem ele, são PULADOS (skipped) em vez de falharem.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import tensorflow as tf  # noqa: F401
    TF_OK = True
except ImportError:
    TF_OK = False


@unittest.skipUnless(TF_OK, "TensorFlow não instalado: pip install -r requirements-cnn.txt")
class TestCNN(unittest.TestCase):
    def test_arquitetura_treino_salvar_carregar_e_prever(self):
        import cnn_treino as t
        import cnn_prever as p

        tam = 32
        modelo = t.construir_cnn(tam)
        self.assertEqual(modelo.output_shape, (None, 1))

        # dados sintéticos separáveis: "fire" = imagens avermelhadas, "nofire" = esverdeadas
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 0.3, (48, tam, tam, 3)).astype("float32")
        y = np.array([1, 0] * 24, dtype="float32")
        x[y == 1, :, :, 0] += 0.6
        x[y == 0, :, :, 1] += 0.6
        hist = modelo.fit(x, y, epochs=2, batch_size=16, verbose=0)
        self.assertEqual(len(hist.history["loss"]), 2)
        self.assertTrue(np.isfinite(hist.history["loss"]).all())

        saida = modelo.predict(x[:8], verbose=0)
        self.assertEqual(saida.shape, (8, 1))
        self.assertTrue(((saida >= 0) & (saida <= 1)).all())

        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "m.keras"
            modelo.save(caminho)
            recarregado = tf.keras.models.load_model(caminho)
            np.testing.assert_allclose(saida, recarregado.predict(x[:8], verbose=0), atol=1e-5)

            # previsão de uma imagem nova pelo caminho público (sem arquivo de config)
            img = Path(tmp) / "nova.jpg"
            Image.fromarray((x[0] * 255).astype("uint8")).save(img)
            r = p.prever_imagem(img, caminho_modelo=caminho)
            self.assertIn(r["classe"], ("fire", "nofire"))
            self.assertAlmostEqual(r["prob_fire"] + r["prob_nofire"], 1.0, places=6)

class TestSubamostragem(unittest.TestCase):
    """Não precisa de TensorFlow."""

    def test_subamostragem_smoke(self):
        import pandas as pd
        import cnn_treino as t
        import cnn_dados as d
        linhas = [{"arquivo": f"{c}/{s}{i}", "classe": c, "rotulo": d.ROTULO[c], "grupo": i, "conjunto": s}
                  for s in d.CONJUNTOS for c in d.CLASSES for i in range(30)]
        sub = t.subamostrar_manifesto(pd.DataFrame(linhas), {"train": 10, "val": 5, "test": 5}, 0)
        self.assertEqual(len(sub), 2 * (10 + 5 + 5))
        self.assertEqual(sorted(sub["conjunto"].unique()), sorted(d.CONJUNTOS))


if __name__ == "__main__":
    unittest.main()
