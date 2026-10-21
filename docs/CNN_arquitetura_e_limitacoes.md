# CNN fire / nofire: arquitetura, pré-processamento, treinamento, limitações, vieses e riscos

Documento da parte de **Deep Learning e reconhecimento de imagens** (seções 14, 17, 19 e 20 do PDF da atividade).
Os números do desempenho **não estão aqui de propósito**: são gerados a cada execução em
`results/cnn/relatorio_cnn.md` (e nos JSON/CSV/PNG da mesma pasta). Este texto descreve o método e os riscos.

## 1. Problema e dados
- **Problema:** dada uma imagem RGB, decidir se há incêndio (`fire`, rótulo 1) ou não (`nofire`, rótulo 0).
- **Dataset:** `data/raw/g20gli_dataset/{fire,nofire}`, 1900 imagens (950 por classe), todas RGB, quase quadradas
  (1826 de 250x250, 73 de 256x256 e 1 de 252x252). O `archive.zip` enviado, com a pasta `Forest Fire Dataset`
  (Training/Testing), continha as mesmas 1900 imagens (idênticas por hash); foi usado apenas o dataset do projeto.
- **Variável-alvo:** a classe da pasta. **Características:** os pixels da imagem; a CNN aprende as
  características sozinha (diferente da Árvore/RNA do grupo, que usam atributos extraídos em `atributos_g20gli.csv`).
- **Pergunta de pesquisa:** uma CNN pequena, treinada do zero em CPU com 1900 imagens, reconhece incêndio em imagens
  novas com boa generalização, e a que custo de falsos negativos e falsos positivos?

## 2. Pré-processamento (`src/cnn_dados.py`)
1. Abre a imagem e converte para **RGB** (cobre tons de cinza, RGBA etc.).
2. **Redimensiona** para 128x128 (interpolação bilinear; padrão alterável por `--tamanho`).
3. **Normaliza** dividindo por 255 (valores em [0, 1]).
A mesma função (`carregar_imagem`) é usada no treino, na avaliação e na previsão de imagens novas.
Sem aumento de cor: a cor (laranja/vermelho) é informação importante para incêndio.

**Divisão:** 70% treino / 15% validação / 15% teste, estratificada por classe, semente 42
(`results/cnn/divisao_dados.csv`). **Vazamento:** hash perceptual (dHash 64 bits, distância de Hamming <= 4) agrupa imagens
iguais ou quase iguais; cada grupo vai inteiro para um único conjunto. Nenhum arquivo aparece em dois conjuntos
(verificado a cada execução; o treino aborta se houver). O teste só é usado depois do treino.

## 3. Arquitetura (`construir_cnn` em `src/cnn_treino.py`)
```
Entrada 128x128x3
 -> [somente no treino] RandomFlip horizontal, RandomRotation 5%, RandomZoom 10%, RandomTranslation 8%
 -> 4 blocos: Conv2D 3x3 (16, 32, 64, 128 filtros) -> BatchNormalization -> ReLU -> MaxPooling 2x2
 -> GlobalAveragePooling2D -> Dropout 0,4
 -> Dense 32 (ReLU, L2 1e-4) -> Dropout 0,3
 -> Dense 1 (sigmoid) = P(fire)
```
Cerca de 100 mil parâmetros (valor exato em `results/cnn/resumo_execucao.json` após o treino).
- **Convolução:** cada filtro 3x3 é um pequeno detector (bordas, cores, texturas) aplicado em toda a imagem; os
  blocos mais profundos combinam detectores simples em padrões maiores (chamas, fumaça, contraste).
- **ReLU** (`max(0, x)`): ativação não linear que permite aprender fronteiras complexas.
- **BatchNorm:** estabiliza e acelera o treino. **MaxPooling:** reduz a resolução e dá tolerância a pequenos deslocamentos.
- **GlobalAveragePooling:** substitui camadas densas grandes, reduzindo parâmetros e overfitting.
- **Sigmoid:** transforma a saída em probabilidade; `P(fire) >= limiar` (0,5) => `fire`.
- Arquitetura leve de propósito: roda em computador comum e é adequada ao tamanho do dataset.

## 4. Treinamento
- **Perda:** entropia cruzada binária. **Otimizador:** Adam, taxa de aprendizado inicial 1e-3 (a derivada da perda em
  relação aos pesos é calculada por retropropagação e usada para atualizar os pesos a cada lote).
- **Lote** 32; até **40 épocas**; semente 42 (reprodutibilidade aproximada: em CPU/GPU diferentes pode haver pequenas variações).
- **Contra overfitting:** aumento de dados geométrico, Dropout, L2, BatchNorm, GlobalAveragePooling,
  **EarlyStopping** (paciência 8, restaura os melhores pesos pela `val_loss`) e **ReduceLROnPlateau** (metade da taxa).
- O conjunto de **validação** só orienta o EarlyStopping e a redução da taxa; o de **teste** não participa de nenhuma decisão.
- Após treinar, o modelo é **salvo** (`models/cnn_fire_nofire.keras`), **recarregado** e só então avaliado.

## 5. Avaliação e análises geradas (`cnn_metricas.py`, `cnn_analise.py`)
| Exigência do PDF | Onde sai |
|---|---|
| Accuracy, Precision, Recall, F1 | `metricas_cnn.json/.csv` (teste), `metricas_cnn_validacao.*` |
| Matriz de confusão | `matriz_confusao.png` |
| Curvas de treinamento (loss/accuracy, treino x validação) | `historico_treinamento.png/.csv` |
| Overfitting/generalização (treino x validação x teste) | `analise_generalizacao.json/.png` |
| Exemplos de acertos e de erros (imagens reais) | `exemplos_acertos.png`, `exemplos_erros.png`, `exemplos_selecionados.csv` |
| Falsos positivos/negativos | `taxa_falsos_positivos`, `taxa_falsos_negativos` em `metricas_cnn.json` |
| Tempo (tabela comparativa, seção 15) | `tempo_treino_s`, `tempo_inferencia_ms_por_imagem` em `metricas_cnn.json/.csv` |
| Possível viés por luminosidade | `analise_luminosidade.csv` |
| Relatório consolidado | `relatorio_cnn.md` |

Os exemplos de acerto são amostrados aleatoriamente (semente fixa), para não mostrar só os casos fáceis; os erros mostrados
são os de **maior confiança errada**, que são os mais informativos.

**Como ler o overfitting (seção 19):** compara-se a accuracy de treino (em modo de inferência, sem aumento) com a de
validação e a de teste. Os limiares do diagnóstico automático (lacuna <= 0,03 pequena; > 0,08 alta; treino < 0,80 = possível
subajuste) são **convenções heurísticas** do código, não regras universais, e podem ser mudados em `cnn_analise.py`.
A `log_loss` do relatório não inclui a regularização L2, então difere da `loss` exibida pelo Keras.

## 6. Limitações
- **Tamanho:** só 1900 imagens; o teste tem 284 imagens, então as métricas têm incerteza considerável (uma única imagem
  muda a accuracy em ~0,35 ponto percentual). Uma única divisão e uma única execução não dão intervalo de confiança.
- **Escopo:** classifica a imagem inteira; não localiza o fogo, não estima a área nem o estágio do incêndio.
- **Distribuição:** só foi avaliada em imagens do mesmo dataset. Câmeras, resoluções, climas, regiões e horários diferentes
  (câmeras de torre, drones, satélite) podem derrubar o desempenho; nada aqui mostra o contrário.
- **Duplicatas:** o dHash pega cópias e quase-cópias, mas não garante a ausência de quadros do mesmo evento/vídeo com
  pequenas diferenças entre si; isso pode inflar levemente o desempenho em teste.
- **Pré-processamento:** o redimensionamento para quadrado distorce levemente imagens não quadradas (poucas aqui).
- **Fotos de internet:** pelas amostras que inspecionei, as imagens parecem fotografias variadas (incluindo cenas noturnas e de
  grande contraste); isso é diferente de imagens de monitoramento reais, em geral menores, com fumaça distante e fogo pequeno.
- **Explicabilidade:** a CNN é uma caixa-preta; o código não inclui Grad-CAM nem outra técnica de explicação.
  Os exemplos de erro permitem uma análise qualitativa, mas não provam *por que* o modelo errou.
- **Calibração:** `P(fire)` é a saída da sigmoid; não foi verificado que ela corresponde a uma probabilidade real.

## 7. Possíveis vieses (hipóteses a verificar, não conclusões)
| Possível viés | Por quê | Como verificar com o que há no repositório |
|---|---|---|
| **Atalho de cor/luz** | `fire` tende a ter tons laranja/escuros e `nofire` tende a ser paisagens verdes e claras; o modelo pode aprender "laranja = fogo" | olhar `exemplos_erros.png`: pôr do sol, luz alaranjada, névoa ou fumaça de neblina em `nofire` viram falsos positivos?; `analise_luminosidade.csv` |
| **Cena noturna/escura** | imagens escuras podem ser mais difíceis | taxas de erro por faixa de luminosidade (amostras pequenas por faixa) |
| **Fumaça x névoa/nuvem** | aparência parecida | inspeção qualitativa dos erros |
| **Geográfico/climático/sazonal** | o dataset não traz metadados (local, data, tipo de vegetação) | **não é possível medir** com estes dados; deve ser declarado como limitação |
| **Classes sub-representadas** | as duas classes estão equilibradas (950/950), mas subtipos (fogo pequeno, fumaça sem chama, fogo urbano) podem ser raros | contagem manual de subtipos; não automatizada aqui |
| **Equilíbrio artificial** | em campo, `nofire` é muito mais frequente que `fire` | métricas em dados equilibrados superestimam a precisão que se teria em uso real (muito mais falsos alarmes por incêndio real) |

## 8. Riscos de falsos positivos e falsos negativos
- **Falso negativo (incêndio não detectado)**: é o erro mais grave neste contexto: atraso de resposta, perda de
  vidas, floresta e patrimônio. Mede-se por `taxa_falsos_negativos` (= 1 - recall). Mitigação: baixar o `--limiar` de decisão aumenta o recall,
  ao custo de mais falsos alarmes; o limiar deve ser escolhido pelo custo relativo dos erros, não pela accuracy.
- **Falso positivo (alarme falso)**: gera mobilização desnecessária, custo, perda de confiança e "fadiga de alarme", o que
  pode levar a ignorar alertas verdadeiros. Mede-se por `taxa_falsos_positivos`; a precisão também cai muito se o `nofire` real for
  muito mais comum que no dataset.
- O trade-off entre os dois deve ser decidido por humanos responsáveis pela operação e documentado.

## 9. Ética (seção 20 do PDF), respostas da parte de CNN
| Pergunta | Resposta |
|---|---|
| Os dados possuem vieses? | Provavelmente sim (ver seção 7); a distribuição geográfica/temporal é desconhecida. |
| Existem classes sub-representadas? | As duas classes têm 950 imagens; subtipos dentro das classes não foram medidos. |
| O modelo pode discriminar grupos? | Não classifica pessoas; mas uma ferramenta de vigilância baseada nele poderia afetar de forma desigual regiões ou comunidades (p.ex., regiões com pouca cobertura de dados). Não foi testado. |
| Risco de falso positivo / falso negativo? | Ver seção 8; valores medidos em `results/cnn/metricas_cnn.json`. |
| Os dados têm informações sensíveis? | São paisagens/incêndios; não foi feita varredura de rostos, placas ou casas. Se houver pessoas ou propriedades identificáveis, devem ser tratadas conforme a licença/LGPD. A licença e a origem das imagens devem ser confirmadas pelo grupo. |
| Uso indevido? | Uso como único critério de decisão de emergência; uso em vigilância de áreas habitadas; confiar nele fora do domínio treinado. |
| Quem responde pela decisão? | O operador/organização que adota o sistema; o modelo é apoio à decisão, não decisor. |
| O modelo é explicável? | Pouco. É uma CNN sem técnica de explicação implementada; a Árvore de Decisão e as regras do grupo são mais interpretáveis. |
| Limitações a comunicar ao usuário | Só classifica imagem inteira; testado só em um dataset; taxas de erro de FP e FN; pode errar com luz alaranjada, névoa, noite; não substitui verificação humana. |

## 10. Para a defesa técnica (seção 27)
Pontos que podem ser alterados e reexecutados rapidamente: `--epocas`, `--batch`, `--tamanho`, `--limiar`, `--paciencia`,
`--semente`, `--sem-aumento` (ablação) e, no código, o número de filtros/camadas em `construir_cnn`.
Teste rápido do pipeline: `python src\cnn_treino.py --smoke`.
