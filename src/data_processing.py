import os
import pandas as pd
import numpy as np
from PIL import Image

# Caminho corrigido (sem o ../ no início)
DATA_DIR = "data/raw/g20gli_dataset"

def extrair_caracteristicas_visuais():
    dados = []
    categorias = ["fire", "nofire"]
    
    for categoria in categorias:
        pasta_alvo = os.path.join(DATA_DIR, categoria)
        
        # Define a variável-alvo (1 para incêndio, 0 para sem incêndio)
        label = 1 if categoria == "fire" else 0
        
        print(f"Extraindo dados da pasta: {categoria}...")
        
        # Verifica se a pasta existe antes de tentar ler
        if not os.path.exists(pasta_alvo):
            print(f"AVISO: A pasta {pasta_alvo} não foi encontrada. Colocaste as imagens no sítio certo?")
            continue

        # Varre as imagens contidas no diretório
        for arquivo in os.listdir(pasta_alvo):
            if arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                caminho_completo = os.path.join(pasta_alvo, arquivo)
                
                try:
                    # Carrega a imagem RGB usando a biblioteca PIL
                    img = Image.open(caminho_completo).convert('RGB')
                    img_array = np.array(img)
                    
                    # 1. Estatísticas de Canais de Cor (Médias de 0 a 255)
                    media_R = np.mean(img_array[:, :, 0])
                    media_G = np.mean(img_array[:, :, 1])
                    media_B = np.mean(img_array[:, :, 2])
                    
                    # 2. Desvio padrão (indica contraste/textura na vegetação)
                    std_R = np.std(img_array[:, :, 0])
                    std_G = np.std(img_array[:, :, 1])
                    
                    # 3. Luminosidade calculada por canais ponderados
                    luminosidade = np.mean(0.299 * img_array[:, :, 0] + 
                                           0.587 * img_array[:, :, 1] + 
                                           0.114 * img_array[:, :, 2])
                    
                    dados.append({
                        "id_imagem": arquivo,
                        "media_R": media_R,
                        "media_G": media_G,
                        "media_B": media_B,
                        "std_R": std_R,
                        "std_G": std_G,
                        "luminosidade": luminosidade,
                        "target": label
                    })
                except Exception as e:
                    print(f"Erro ao processar arquivo {arquivo}: {e}")
                    
    return pd.DataFrame(dados)

# Executa o processamento estruturado
df_caracteristicas = extrair_caracteristicas_visuais()

# Garante a criação do diretório processed exigido (caminho corrigido)
os.makedirs("data/processed", exist_ok=True)

# Salva a tabela final convertida (caminho corrigido)
if not df_caracteristicas.empty:
    df_caracteristicas.to_csv("data/processed/atributos_g20gli.csv", index=False)
    print("\n[SUCESSO] Planilha contruída em data/processed/atributos_g20gli.csv!")
    print(f"Formato gerado para os modelos: {df_caracteristicas.shape}")
else:
    print("\n[ERRO] Nenhum dado foi extraído. Verifica se as imagens estão na pasta data/raw/g20gli_dataset.")