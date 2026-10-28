import pandas as pd
import os
import matplotlib.pyplot as plt
import glob

while True:
    grd_acro = 'MAN'  # input('Gridcell acronym [AFL, ALP, FEC, MAN, CAX]: ')

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
    elif grd_acro == 'NVX':
        grd = '210-249'
        break
    elif grd_acro == 'AFL':
        grd = '199-248'
        break
    else:
        print('This acronym does not correspond')
        break

perc_prec = 30  # input("What is the percentage of reduction? [0, 10, 20, 30] ")

# Consolidate file paths
base_path = "/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs"
path_regclim = os.path.join(base_path, grd_acro, f"experiments/{grd_acro}_regularclimate/gridcell{grd}/concatenated_series_{grd_acro}_regularclimate.csv")
path_csv_allfreq = os.path.join(base_path, grd_acro, f"experiments/{perc_prec}perc_reduction",
                                f"concatenated_series_{grd_acro}_{perc_prec}prec_allfreq.csv")
path = os.path.join(base_path, grd_acro, f"experiments/{perc_prec}perc_reduction/")

# Use a function for frequency logic
def get_file_path(freq):
    if freq == 0:
        return path_regclim
    return os.path.join(base_path, grd_acro, f'experiments/{perc_prec}perc_reduction/concatenated_series_{grd_acro}_{perc_prec}prec_{freq}y.csv')

# Avoid using reserved words
dfs_list = []

# Verify if the table with all frequencies for this experiment already exists
if os.path.exists(path_csv_allfreq) and os.path.exists(path_regclim):
    print('\n !!!!!!!!! The csv with all frequencies for this experiment already exists !!!!!!!!! \n')
    csv_allfreq = pd.read_csv(path_csv_allfreq)
    df_regclim = pd.read_csv(path_regclim)

else:
    print('\n !!!!!!!!! Creating the csv with all frequencies for this experiment !!!!!!!!! \n')
    df_regclim = pd.read_csv(path_regclim)
    df_regclim['frequency'] = 0
    df_regclim['prec_red_perc'] = 0.0
    df_regclim['total_carbon'] = df_regclim[['cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto']].sum(axis=1)

    path = os.path.join(base_path, grd_acro, f"experiments/{perc_prec}perc_reduction/")
    files = glob.glob(os.path.join(path, '*y.csv'))

    for file in files:
        frequency = int(file.split('_')[-1][0])
        df = pd.read_csv(file)
        df['frequency'] = frequency
        df['prec_red_perc'] = int(perc_prec)
        df['total_carbon'] = df[['cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto']].sum(axis=1)
        dfs_list.append(df)

    dfs_list.append(df_regclim)
    csv_allfreq = pd.concat(dfs_list, ignore_index=True)
    csv_allfreq['date_dateformat'] = pd.to_datetime(csv_allfreq['Date'])
    csv_allfreq.to_csv(os.path.join(path, f'concatenated_series_{grd_acro}_{perc_prec}prec_allfreq.csv'), index=False)

# Plotting other variables with 1 year frequency
df_regclim = pd.read_csv(path_regclim)
df_regclim['frequency'] = 0
df_regclim['prec_red_perc'] = 0.0
df_regclim['total_carbon'] = df_regclim[['cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto']].sum(axis=1)
df_regclim['date_dateformat'] = pd.to_datetime(df_regclim['Date'])

df = pd.read_csv(
    f'/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{grd_acro}/experiments/{perc_prec}perc_reduction/concatenated_series_{grd_acro}_{perc_prec}prec_allfreq.csv')
df['date_dateformat'] = pd.to_datetime(df['Date'])

# Lista de frequências desejadas
frequencies = [0, 1, 3, 5, 7]

variables_to_plot = ['npp', 'ls', 'evapm', 'total_carbon']  # , 'cleaf', 'croot', 'csap', 'cheart', 'csto']

# Criar subplots para cada variável
plt.figure(figsize=(18, 20))

# Número de subplots
num_subplots = len(variables_to_plot)

# Número de linhas e colunas
num_rows = 4
num_cols = 1

# Criar subplots para cada variável
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows), sharex=True)
# Dicionário para mapear variáveis para títulos

# Iterar sobre as variáveis
for idx, variable in enumerate(variables_to_plot):
    # Calcular a posição do subplot na matriz 4x1
    row = idx

    # Plotar cada frequência em um subplot separado
    for freq_idx, freq in enumerate(set(df['frequency'])):
        if freq == 0:
            linestyle = '-'
            label = 'Regular Climate'
        else:
            linestyle = '-'
            label = f'{freq} years'

        # Filtrar o DataFrame para a frequência atual
        df_freq = df[df['frequency'] == freq]
        
       # Adicionar mais uma dimensão ao acesso aos subplots
        if variable == 'ls':
            # Aumentar a linewidth para 'ls'
            axes[row].plot(df_freq['date_dateformat'], df_freq[variable], linewidth=2, alpha=0.7, linestyle=linestyle, label=label)
        else:
            axes[row].plot(df_freq['date_dateformat'], df_freq[variable], linewidth=0.5, alpha=0.7, linestyle=linestyle, label=label)

        axes[row].set_ylabel(f'{variable}', fontsize=16)
        axes[row].tick_params(axis='both', which='both', labelsize=12)

# Adicionar rótulos para o eixo x comum
axes[-1].set_xlabel('Date', fontsize=20)

# Adicionar legenda fora do subplot
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.02), loc='upper center')


# Ajustes na posição da legenda e espaçamento
fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.02), loc='upper center')
fig.subplots_adjust(bottom=0.15, top=0.9)

# Salvar o gráfico como um arquivo PNG
plt.savefig(os.path.join(path, f'{grd_acro}_timeseries_allfreq_{perc_prec}perc_subplots.png'), bbox_inches='tight')

# Mostrar o gráfico
plt.show()