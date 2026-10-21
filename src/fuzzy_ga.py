import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pygad

print("Carregando os dados...")
caminho_dados = "data/processed/atributos_g20gli.csv"
dados = pd.read_csv(caminho_dados)
print("Colunas do CSV:", dados.columns)

print("Configurando a Lógica Fuzzy...")
# 1. Definir os universos (0-255 para as cores/luminosidade e 0-100 para o risco)
universo_cor_lum = np.arange(0, 256, 1)
universo_risco = np.arange(0, 101, 1)

# 2. Criar as variáveis de entrada (Antecedentes)
media_r = ctrl.Antecedent(universo_cor_lum, 'media_R')
media_g = ctrl.Antecedent(universo_cor_lum, 'media_G')
media_b = ctrl.Antecedent(universo_cor_lum, 'media_B')
luminosidade = ctrl.Antecedent(universo_cor_lum, 'luminosidade')

# 3. Criar a variável de saída (Consequente)
risco = ctrl.Consequent(universo_risco, 'risco')

# 4. Gerar automaticamente as categorias (baixo, médio, alto) para todas as variáveis
nomes_categorias = ['baixo', 'medio', 'alto']
media_r.automf(3, names=nomes_categorias)
media_g.automf(3, names=nomes_categorias)
media_b.automf(3, names=nomes_categorias)
luminosidade.automf(3, names=nomes_categorias)
risco.automf(3, names=nomes_categorias)

print("Criando as regras Fuzzy...")
# Regra 1: Se o vermelho é alto e a luminosidade é alta, então o risco de fogo é ALTO
regra1 = ctrl.Rule(media_r['alto'] & luminosidade['alto'], risco['alto'])

# Regra 2: Se o vermelho é baixo, então o risco de fogo é BAIXO
regra2 = ctrl.Rule(media_r['baixo'], risco['baixo'])

# Regra 3: Se o verde e o azul são altos (natureza/céu), o risco de fogo é BAIXO
regra3 = ctrl.Rule(media_g['alto'] & media_b['alto'], risco['baixo'])

# Criar o sistema de controle com essas regras
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3])
simulador = ctrl.ControlSystemSimulation(sistema_controle)

print("Testando o simulador Fuzzy...")
# Inserir valores fictícios (de 0 a 255) simulando as cores de uma chama
simulador.input['media_R'] = 220
simulador.input['media_G'] = 60
simulador.input['media_B'] = 40
simulador.input['luminosidade'] = 240

# Executar o cálculo lógico
simulador.compute()

# Mostrar o resultado final
resultado_risco = simulador.output['risco']
print(f"Risco de fogo calculado: {resultado_risco:.2f}%\n")

print("Configurando o Algoritmo Genético...")
# 5. Função de aptidão (Fitness) real usando os dados do CSV
def fitness_func(ga_instance, solution, solution_idx):
    acertos = 0
    # Pegamos uma amostra de 50 imagens para treinar mais rápido
    amostra = dados.sample(50, random_state=42) 
    
    for _, linha in amostra.iterrows():
        # Alimentar o simulador com os dados reais da imagem
        simulador.input['media_R'] = float(linha['media_R'])
        simulador.input['media_G'] = float(linha['media_G'])
        simulador.input['media_B'] = float(linha['media_B'])
        simulador.input['luminosidade'] = float(linha['luminosidade'])
        
        try:
            simulador.compute()
            risco_calculado = simulador.output['risco']
        except:
            risco_calculado = 0 
            
        previsao_fogo = risco_calculado > 50
        eh_fogo_real = str(linha['target']).strip().lower() in ['fire', '1', 'fogo']
        
        if previsao_fogo == eh_fogo_real:
            acertos += 1
            
    return acertos

# 6. Instância do GA
ga_instance = pygad.GA(
    num_generations=20,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=6,
    init_range_low=0,
    init_range_high=255,
    mutation_num_genes=1, # Corrige o aviso de 0 genes
    on_generation=lambda ga: print(f"Geração {ga.generations_completed} | Maior nota (acertos em 50): {ga.best_solution()[1]}")
)

print("Iniciando o treinamento do Algoritmo Genético...")
ga_instance.run()

print("\nTreinamento concluído com sucesso!")