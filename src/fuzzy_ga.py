import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pygad

print("Carregando os dados...")
caminho_dados = "data/processed/atributos_g20gli.csv"
dados = pd.read_csv(caminho_dados)

media_r = ctrl.Antecedent(np.arange(0, 256, 1), 'media_R')
luminosidade = ctrl.Antecedent(np.arange(0, 256, 1), 'luminosidade')

risco = ctrl.Consequent(np.arange(0, 101, 1), 'risco')

def fitness_func(ga_instance, solution, solution_idx):
    erro_simulado = 10 
    nota = 1.0 / (erro_simulado + 1e-10)
    return nota

print("Configurando o Algoritmo Genético...")
ga_instance = pygad.GA(
    num_generations=20,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=10,
    num_genes=6,
    init_range_low=0,
    init_range_high=255
)

print("Estrutura inicial pronta para rodar!")