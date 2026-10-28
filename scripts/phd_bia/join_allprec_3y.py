import pandas as pd
import os

# Caminhos dos arquivos CSV
base_path = "/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments"
csv_files = [
    "concatenated_series_MAN_10prec_3y.csv",
    "concatenated_series_MAN_20prec_3y.csv",
    "concatenated_series_MAN_30prec_3y.csv"
]

# Lista para armazenar os DataFrames
dfs_list = []

# Iterar sobre os arquivos CSV
for csv_file in csv_files:
    # Extrair o valor de 'perc' do nome do arquivo
    perc = int(csv_file.split('_')[3].replace('prec', ''))

    # Ler o CSV e adicionar a coluna 'perc'
    df = pd.read_csv(os.path.join(base_path, csv_file))
    df['perc'] = perc

    # Adicionar o DataFrame à lista
    dfs_list.append(df)

# Concatenar os DataFrames em um único DataFrame
result_df = pd.concat(dfs_list, ignore_index=True)

# Salvar o DataFrame resultante em um novo CSV
result_df.to_csv(os.path.join(base_path, "concatenated_series_MAN_allprec_3y.csv"), index=False)
