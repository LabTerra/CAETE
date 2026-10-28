import bz2
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import copy

# # Defina os anos inicial e final
start_year = 1979
end_year = 2016  # Ajustado para 2016 para corresponder aos dados fornecidos

caminho_saida_csv_mensal = '/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/prec_values_monthly_grouped.csv'


# # Leia o arquivo CSV com a coluna 'Date' no formato 'ano-mes-dia'
# caminho_arquivo_csv = '/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/concatenated_series_MAN_10prec_3y.csv'  # Substitua pelo caminho real do seu arquivo CSV
# df_dates = pd.read_csv(caminho_arquivo_csv)
# dates = pd.to_datetime(df_dates['Date'], format='%Y-%m-%d')
# print(len(dates))
# # # Substitua 'seuarquivo.pbz2' e 'saida_pr.csv' pelos caminhos reais dos seus arquivos
# caminho_arquivo_pbz2 = '/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/input/central/input_data_186-239.pbz2'

caminho_saida_csv = '/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/prec_values.csv'

# with bz2.BZ2File(caminho_arquivo_pbz2, mode='r') as fh:
#     dados = pickle.load(fh)
# dados['pr'] = [valor * 86400 for valor in dados['pr']]

# # Criar DataFrame pandas
# df_pr = pd.DataFrame({'pr': dados['pr'][:13880], 'data': dates[:13880]})
# # Salvar os dados em um arquivo CSV
# df_pr.to_csv(caminho_saida_csv, index=False)
# print(f'Dados da variável "pr" (até o 13880º valor) salvos em {caminho_saida_csv}')
# abrir arquivo de valor diário de precipitação
daily_pr = pd.read_csv(caminho_saida_csv)
# Converta a coluna 'data' para o tipo datetime
daily_pr['date'] = pd.to_datetime(daily_pr['data'])

daily_npp = pd.read_csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/MAN_regularclimate/gridcell186-239/concatenated_series_MAN_regularclimate.csv")
daily_npp['date'] = pd.to_datetime(daily_npp['Date'])

# Plotando com dois eixos y
fig, ax1 = plt.subplots(figsize=(10, 6))

# Eixo y para a precipitação
ax1.plot(daily_pr['date'], daily_pr['pr'], linestyle='-', color='b', label='Precipitação')
ax1.set_xlabel('Data')
ax1.set_ylabel('Precipitação', color='b')
ax1.tick_params('y', colors='b')

# Criar um segundo eixo y
ax2 = ax1.twinx()

# Eixo y para a NPP
ax2.plot(daily_npp['date'], daily_npp['npp'], linestyle='-', color='g', label='NPP')
ax2.set_ylabel('NPP', color='g')
ax2.tick_params('y', colors='g')

# Ajustes do gráfico
fig.suptitle('Precipitação e NPP ao longo do tempo')
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Ajusta layout para evitar sobreposição de título
plt.grid(True)
plt.show()

# # Plotando
# plt.figure(figsize=(10, 6))
# plt.plot(daily_pr['date'], daily_pr['pr'], linestyle='-', color='b')
# plt.title('Precipitação ao longo do tempo')
# plt.xlabel('Data')
# plt.ylabel('Precipitação (pr)')
# plt.grid(True)
# plt.show()

plt.savefig('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/scripts/prec_daily.png')




# # Calcular a média mensal de precipitação
# df_pr['mes'] = df_pr['data'].dt.to_period('M')
# df_mensal = df_pr.groupby('mes').sum().reset_index()
# print(len(df_mensal))

# # # Salvar os dados mensais em um arquivo CSV
# caminho_saida_csv_mensal = '/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/prec_values_monthly_grouped.csv'
# # df_mensal.to_csv(caminho_saida_csv_mensal, index=False)
# # print(f'Dados mensais de precipitação salvos em {caminho_saida_csv_mensal}')

# # Carregar os dados mensais do arquivo CSV
# df_mensal = pd.read_csv(caminho_saida_csv_mensal)

# # Converter a coluna 'mes' para o formato adequado
# df_mensal['mes'] = pd.to_datetime(df_mensal['mes']).dt.to_period('M')

# # Obter valores numéricos para o eixo x (número de meses a partir de uma data de referência)
# df_mensal['mes_numerico'] = df_mensal['mes'].dt.year * 12 + df_mensal['mes'].dt.month

# # Extrair os anos da coluna 'mes' e criar uma nova coluna 'ano'
# df_mensal['ano'] = df_mensal['mes'].dt.year

# # Plotar os dados mensais
# plt.figure(figsize=(10, 6))
# plt.plot(df_mensal['mes'], df_mensal['pr'],  linestyle='-', color='b')
# plt.title('Média Mensal de Precipitação')
# plt.xlabel('Meses desde o início')
# plt.ylabel('Precipitação (mm)')
# plt.grid(True)


# # Salvar o gráfico em um arquivo PNG
# plt.savefig('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/prec_monthly_regclim.png')

# intervalo = 1
# intervalo = int(intervalo) + 1 #+1 guarantee the right interval between applications


# df_mensal = pd.read_csv(caminho_saida_csv_mensal)

# df_mensal_exp = copy.deepcopy(df_mensal)

# pr_exp_values = []

# # Loop sobre os anos
# for year in range(1979, 2017):
#     start_date = f"{year}0101"
#     end_date = f"{year}1231"

#     pr_exp_values.append(df_mensal_exp['pr'][:13880])  # Adicione os valores originais para o ano atual

#     # Aplicar a redução de precipitação conforme a frequência desejada
#     if (year % intervalo == 0) and (start_date != '19790101'):
#         print(f"Aplicando a redução em {year}")
#         df_mensal_exp['pr'] = [valor * 0.7 for valor in df_mensal_exp['pr']]