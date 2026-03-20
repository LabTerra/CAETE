import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
grd_acro = 'MAN'
base_path = "/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/"
# path_regclim = os.path.join(base_path, f"experiments/MAN_regularclimate/gridcell186-239/concatenated_series_MAN_regularclimate.csv")

# df = pd.read_csv('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/concatenated_series_MAN_allprec_3y.csv')
# df_regclim = pd.read_csv(path_regclim)

# df_regclim['perc'] = 0

# # Concatenar os DataFrames
# result_df = pd.concat([df, df_regclim], ignore_index=True)

# result_df['total_carbon'] = result_df[['cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto']].sum(axis=1)

# result_df['date_dateformat'] = pd.to_datetime(result_df['Date'])


# # # # Agora, result_df contém ambos os DataFrames concatenados
# # # # Você pode salvar result_df em um novo arquivo CSV se necessário
# result_df.to_csv(os.path.join(base_path, "experiments/concatenated_series_MAN_allprec_3y.csv"), index=False)

df = pd.read_csv(os.path.join(base_path, "experiments/concatenated_series_MAN_allprec_3y.csv"))

variables_to_plot = ['npp', 'ls', 'evapm', 'total_carbon']  # , 'cleaf', 'croot', 'csap', 'cheart', 'csto']

# Lista de porcentagens desejadas
percentages = [0, 10, 20, 30]

# Criar subplots para cada variável
plt.figure(figsize=(18, 20))

# Número de subplots
num_subplots = len(variables_to_plot)

# Número de linhas e colunas
num_rows = 4
num_cols = 1

# Criar subplots para cada variável
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows), sharex=True)

# Iterar sobre as variáveis
for idx, variable in enumerate(variables_to_plot):
    # Calcular a posição do subplot na matriz 4x1
    row = idx

    # Plotar cada porcentagem em um subplot separado
    for perc_idx, perc in enumerate(percentages):
        if perc == 0:
            linestyle = '-'
            label = 'Regular Climate'
        else:
            linestyle = '-'
            label = f'{perc}% reduction'

        # Filtrar o DataFrame para a porcentagem atual
        df_perc = df[df['perc'] == perc]
        
        # Adicionar mais uma dimensão ao acesso aos subplots
        if variable == 'ls':
            # Aumentar a linewidth para 'ls'
            axes[row].plot(df_perc['date_dateformat'], df_perc[variable], linewidth=2, alpha=0.7, linestyle=linestyle, label=label)
        else:
            axes[row].plot(df_perc['date_dateformat'], df_perc[variable], linewidth=0.5, alpha=0.7, linestyle=linestyle, label=label)

        axes[row].set_ylabel(f'{variable}', fontsize=16)
        axes[row].tick_params(axis='both', which='both', labelsize=12)

# Adicionar rótulos para o eixo x comum
axes[-1].set_xlabel('Date', fontsize=20)

# Adicionar legenda fora do subplot
handles, labels = axes[-1].get_legend_handles_labels()
# fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.02), loc='upper center')

# Ajustes na posição da legenda e espaçamento
# fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.02), loc='upper center')
fig.subplots_adjust(bottom=0.15, top=0.9)

# Salvar o gráfico como um arquivo PNG
plt.savefig(os.path.join(base_path, f'{grd_acro}_timeseries_allperc_subplots.png'), bbox_inches='tight')

# Mostrar o gráfico
plt.show()