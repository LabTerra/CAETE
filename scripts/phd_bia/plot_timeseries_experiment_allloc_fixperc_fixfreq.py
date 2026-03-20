import pandas as pd
import matplotlib.pyplot as plt

# MAN = pd.read_csv('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/30perc_reduction/concatenated_series_MAN_30prec_3y.csv')
# AFL = pd.read_csv('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/AFL/experiments/30perc_reduction/concatenated_series_AFL_30prec_3y.csv')
# ALP = pd.read_csv('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/ALP/experiments/30perc_reduction/concatenated_series_ALP_30prec_3y.csv')
# FEC = pd.read_csv('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/FEC/experiments/30perc_reduction/concatenated_series_FEC_30prec_3y.csv')
# CAX = pd.read_csv('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/CAX/experiments/30perc_reduction/concatenated_series_CAX_30prec_3y.csv')

# # Adicionando a coluna 'location' a cada dataframe
# MAN['location'] = 'MAN'
# AFL['location'] = 'AFL'
# ALP['location'] = 'ALP'
# FEC['location'] = 'FEC'
# CAX['location'] = 'CAX'

# # Concatenando os dataframes
# df_concatenado = pd.concat([MAN, AFL, ALP, FEC, CAX], ignore_index=True)

# df_concatenado['total_carbon'] = df_concatenado[['cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto']].sum(axis=1)

# df_concatenado['date_dateformat'] = pd.to_datetime(df_concatenado['Date'])

# print(df_concatenado.columns)
# df_concatenado.to_csv('/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/ALL_LOCATIONS/concatenated_series_ALL_LOCATIONS_30prec_3y.csv', index=False)

import pandas as pd
import os
import matplotlib.pyplot as plt
import glob


    


# Consolidate file paths
base_path = "/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/ALL_LOCATIONS/"

# Avoid using reserved words
dfs_list = []


# # Plotting the time series
# plt.figure(figsize=(13, 9))

# for freq in [0, 1, 3, 5, 7]:
#     file_path = get_file_path(freq)
#     if os.path.exists(file_path):
#         if freq == 0:
#             print(f'\nPlotting time series for regular climate...\n')
#             df = pd.read_csv(file_path)
#             df['date_dateformat'] = pd.to_datetime(df['Date'])
#             plt.plot(df['date_dateformat'], df['npp'], label='Regular climate', linewidth=0.6, alpha=0.8, color='black', zorder=5)
#         else:
#             print(f'\nPlotting time series for frequency {freq} years...\n')
#             df = pd.read_csv(file_path)
#             df['date_dateformat'] = pd.to_datetime(df['Date'])
#             plt.plot(df['date_dateformat'], df['npp'], label=f'Frequency {freq} years', linewidth=0.8, alpha=0.8)

# # Add labels to the axes and legend
# plt.xlabel('Date')
# plt.ylabel('NPP')
# plt.title('Time series - {perc_prec}% precipitation reduction')
# plt.legend()

# # Save the plot as a PNG file
# plt.savefig(os.path.join(path, f'{grd_acro}_timeseries_allfreq_{perc_prec}perc.png'))

# # Show the plot
# plt.show()

# # Plot freq X regular climate
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True)

# frequencies = [1, 3, 5, 7]

# for idx, freq in enumerate(frequencies, 1):
#     file_path = get_file_path(freq)
#     if os.path.exists(file_path) and os.path.exists(path_regclim):
#         print(f'\nPlotting time series for frequency {freq} years...\n')
#         df = pd.read_csv(file_path)
#         df['date_dateformat'] = pd.to_datetime(df['Date'])
#         df_regclim = pd.read_csv(path_regclim)
#         df_regclim['date_dateformat'] = pd.to_datetime(df_regclim['Date'])

#         axes[(idx-1)//2, (idx-1)%2].plot(df['date_dateformat'], df['npp'], label=f'Frequency {freq} years', linewidth=0.2, color='coral', alpha=0.8)
#         axes[(idx-1)//2, (idx-1)%2].plot(df_regclim['date_dateformat'], df_regclim['npp'], label='Regular Climate', linewidth=0.2, color='black', alpha=0.8)

#         axes[(idx-1)//2, (idx-1)%2].set_xlabel('Date')
#         axes[(idx-1)//2, (idx-1)%2].set_ylabel('NPP')
#         axes[(idx-1)//2, (idx-1)%2].set_title(f'Time series - {freq} years precipitation reduction')
#         axes[(idx-1)//2, (idx-1)%2].legend()

# plt.tight_layout()
# plt.savefig(os.path.join(path, f'{grd_acro}_timeseries_allfreq_x_regclim_{perc_prec}perc.png'))
# plt.show()


df = pd.read_csv(f'/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/ALL_LOCATIONS/concatenated_series_ALL_LOCATIONS_30prec_3y.csv')
df['date_dateformat'] = pd.to_datetime(df['Date'])



# Lista de locais desejados
locs = ['AFL', 'ALP', 'CAX', 'FEC', 'MAN']

# Lista de variáveis a serem plotadas
variables_to_plot = ['npp', 'ls', 'evapm', 'cleaf', 'croot', 'csap', 'cheart', 'csto']

# Dicionário para mapear variáveis para títulos
variable_titles = {
    'npp': 'Net primary productivity',
    'ls': 'Number of surviving strategies',
    'evapm': 'Evapotranspiration',
    'cleaf': 'Leaf carbon',
    'croot': 'Root carbon',
    'csap': 'Sapwood carbon',
    'cheart': 'Heartwood carbon',
    'csto': 'Storage carbon'
}

# Iterar sobre as variáveis
for variable in variables_to_plot:

    # Criar subplots para cada local dentro da figura da variável
    plt.figure(figsize=(15, 20))

    # Criar subplots para cada local
    fig, axes = plt.subplots(len(locs), 1, figsize=(15, 20), sharex=True, sharey=True)

    # Iterar sobre os locais
    for i, loc in enumerate(locs):
        # Filtrar o DataFrame para o local atual e a variável atual
        df_loc_var = df[(df['location'] == loc) & (df[variable].notnull())]

        # Definir o linewidth com base na variável 'ls'
        linewidth = 2 if variable == 'ls' else 1

        # Plotar a variável no subplot correspondente
        axes[i].plot(df_loc_var['date_dateformat'], df_loc_var[variable], linewidth=linewidth, alpha=1, label=f'{loc}', color='black')

        axes[i].set_ylabel(f'{variable.upper()}', fontsize=18)  # Ajustar o tamanho da fonte no eixo y

        # Adicionar anotação no canto superior direito do subplot
        axes[i].annotate(loc, xy=(1, 1), xytext=(-5, -5), ha='right', va='top', color='black', fontsize=12,
                         bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='white'))

        # Adicionar legenda dentro do subplot
        axes[i].legend(fontsize=20)  # Ajustar o tamanho da fonte na legenda

        # Ajustar o tamanho dos valores dos eixos x e y
        axes[i].tick_params(axis='both', labelsize=14)

    # Adicionar rótulo para o eixo x comum e ajuste manual do layout
    axes[-1].set_xlabel('Date', fontsize=20)  # Ajustar o tamanho da fonte no eixo x

    # Adicionar título específico com base na variável
    plt.suptitle(f'Time series - {variable_titles[variable]} precipitation reduction', y=1.02, fontsize=16)  # Ajustar o tamanho da fonte no título

    # Aumentar o tamanho dos ticks apenas no eixo x
    for ax in axes:
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=18)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.90, hspace=0.1, wspace=0.35)  # Ajustar hspace

    # Salvar o gráfico como um arquivo PNG
    plt.savefig(os.path.join(base_path, f'{variable}_subplots.png'))

    # Mostrar o gráfico
    plt.show()
