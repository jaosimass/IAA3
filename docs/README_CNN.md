# README - parte CNN (para ser incorporada ao README.md do projeto)

> O `README.md` da raiz contém apenas o título do projeto; copie esta seção para ele quando o grupo integrar o README.
> As dependências da CNN (tensorflow, numpy, pandas, pillow, scikit-learn, matplotlib) constam no `requirements.txt` da raiz;
> não existe `requirements-cnn.txt`.

## Deep Learning: CNN fire / nofire

1. **Instalar** (Windows, PowerShell, na raiz do projeto):
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Preparar os dados:** as imagens devem estar em `data\raw\g20gli_dataset\fire` e `data\raw\g20gli_dataset\nofire`.
   A divisão treino/validação/teste e a checagem de vazamento são feitas automaticamente pelo treino
   (inspeção isolada: `python src\cnn_dados.py`).
3. **Executar/treinar:** `python src\cnn_treino.py` (teste rápido: `python src\cnn_treino.py --smoke`).
4. **Testar:** `python -m unittest discover -s tests -v`
5. **Reproduzir os resultados:** repetir o passo 3 com os padrões (semente 42). Saídas em `results\cnn\` e `models\`.
   Regenerar só as análises/figuras sem retreinar: `python src\cnn_analise.py`.
6. **Prever uma imagem nova:** `python src\cnn_prever.py caminho\imagem.jpg`

Notebook: `notebooks\cnn_experimentos.ipynb`. Documentação: `docs\CNN.md`,
`docs\CNN_arquitetura_e_limitacoes.md`, `docs\CNN_plano_experimental.md`.
