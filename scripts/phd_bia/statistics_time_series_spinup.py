# Importar bibliotecas
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pedir ao usuário para inserir o acrônimo da célula [AFL, ALP, FEC, MAN, CAX]
while True:
    grd_acro = input('Gridcell acronym [AFL, ALP, FEC, MAN, CAX]: ')

    if grd_acro == 'ALP':
        grd = '188-213'
        break
    elif grd_acro == 'FEC':
        grd = '200-225'
        break
    elif grd_acro == 'MAN':
        grd = '186-239'
        break
    elif grd_acro == 'CAX':
        grd = '183-257'
        break
    elif grd_acro == 'AFL':
       grd = '199-248'
       break
    else:
        print('This acronym does not correspond')
        break

main_path = f'/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{grd_acro}/state_start/'

# Criar um dicionário para armazenar as séries temporais
dfs = {}

# Loop através dos 10 runs
for i in range(1, 10 + 1):
    run_name = f'{grd_acro}_save_spin_{i}'

    # Navegar até a pasta especificada para acessar os spins
    os.chdir(f'{main_path}{run_name}/gridcell{grd}')

    # Ler o arquivo CSV correspondente à série concatenada
    df = pd.read_csv(f'{main_path}{run_name}/gridcell{grd}/csv/concatenated_series_{run_name}.csv')

    # Converter a coluna 'Date' para o tipo de dado datetime
    df['Date'] = pd.to_datetime(df['Date'])


        # Insert a column to armazenate total carbon
    df['total_carbon'] = df['cleaf'] + df['cwood'] + df['croot'] + df['csap'] + df['cheart'] + df['csto']
    
    df.to_csv(f'{main_path}{run_name}/gridcell{grd}/csv/concatenated_series_{run_name}_totalcarbon.csv')

    df = pd.read_csv(f'{main_path}{run_name}/gridcell{grd}/csv/concatenated_series_{run_name}_totalcarbon.csv')

    # Armazenar o DataFrame no dicionário
    dfs[run_name] = df




# # Lista de variáveis para análise
variaveis_analise = ['npp', 'photo', 'evapm', 'total_carbon']

# # Loop para cada variável
for variable in variaveis_analise:
    # Lista para armazenar as séries temporais da variável atual
    series_variable = []

    # Loop através do dicionário dfs para obter cada série temporal da variável atual
    for run_name, df in dfs.items():
        # Converter a coluna 'Date' para o tipo de dado datetime, se ainda não estiver
        df['Date'] = pd.to_datetime(df['Date'])

        series_variable.append(df[variable].values)

#     # Converter a lista de listas em um array NumPy
    series_array_variable = np.array(series_variable)
    # print(series_array_variable)

    # Calcular a média ao longo do eixo 0 (média de cada ponto de dado ao longo das séries da variável)
    media_geral_variable = np.mean(series_array_variable, axis=0)

    # Calcular a diferença para cada série em relação à média geral da variável
    diferencas_variable = series_array_variable - media_geral_variable
    
    # Calcular a variância para cada série em relação à média geral da variável
    variancias_variable = np.var(diferencas_variable, axis=1)

    # Calcular o desvio padrão para cada série em relação à média geral da variável
    desvios_padrao_variable = np.std(series_array_variable, axis=1)

    # Identificar a série com a maior variância da variável
    indice_maior_variancia_variable = np.argmax(variancias_variable)
    serie_maior_variancia_variable = dfs[list(dfs.keys())[indice_maior_variancia_variable]]
    # Identificar a série com a menor variância da variável
    indice_menor_variancia_variable = np.argmin(variancias_variable)
    serie_menor_variancia_variable = dfs[list(dfs.keys())[indice_menor_variancia_variable]]

#     # Obter o valor de 'ls' para a série de maior variância da variável
#     valor_ls_maior_variancia_variable = serie_maior_variancia_variable['ls'].values[0]

#     # Obter o valor de 'ls' para a série de menor variância da variável
#     valor_ls_menor_variancia_variable = serie_menor_variancia_variable['ls'].values[0]

    # Identificar a série com o maior desvio padrão da variável
    indice_maior_desvio_padrao_variable = np.argmax(desvios_padrao_variable)
    serie_maior_desvio_padrao_variable = dfs[list(dfs.keys())[indice_maior_desvio_padrao_variable]]

    # Identificar a série com o menor desvio padrão da variável
    indice_menor_desvio_padrao_variable = np.argmin(desvios_padrao_variable)
    serie_menor_desvio_padrao_variable = dfs[list(dfs.keys())[indice_menor_desvio_padrao_variable]]

    print(f"Série com Maior Variância em Relação à Média Geral ({variable}): {list(dfs.keys())[indice_maior_variancia_variable]}")  
    print(f"Série com Menor Variância em Relação à Média Geral ({variable}): {list(dfs.keys())[indice_menor_variancia_variable]}")  
    print('')

    print(f"Série com Maior STD em Relação à Média Geral ({variable}): {list(dfs.keys())[indice_maior_desvio_padrao_variable]}")  
    print(f"Série com Menor STD em Relação à Média Geral ({variable}): {list(dfs.keys())[indice_menor_desvio_padrao_variable]}")  
    print('')

#     # Obter o valor de 'ls' para a série de maior desvio padrão da variável
    # valor_ls_maior_desvio_padrao_variable = serie_maior_desvio_padrao_variable['ls'].values[0]
    # print(valor_ls_maior_desvio_padrao_variable)
#     # Obter o valor de 'ls' para a série de menor desvio padrão da variável
#     valor_ls_menor_desvio_padrao_variable = serie_menor_desvio_padrao_variable['ls'].values[0]

#     # Plotar a série com o menor desvio padrão da variável
#     plt.plot(df['Date'], serie_menor_desvio_padrao_variable[variable], label=f"Menor Desvio Padrão ({variable}) - Original", alpha=0.3, linewidth=0.1)

#     # Plotar a série com o maior desvio padrão da variável
