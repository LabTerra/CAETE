import os
import pandas as pd
import warnings
import numpy as np

folder_path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/csv_allvar/csv_allvar_original/'  # Substitua pelo caminho correto

# Pasta para salvar os novos arquivos CSV
output_folder = '/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/csv_allvar/'  # Substitua pelo caminho correto

# Lista todos os arquivos na pasta
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Loop através de cada arquivo CSV
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    result = pd.read_csv(file_path)

    if result['timeindex'].iloc[0] != 1:
        print(f'O índice de tempo não é 1 para o arquivo {csv_file}')

        additional_rows = pd.DataFrame({
            'timeindex': range(1, result['timeindex'].iloc[0]),
            'ar1': [np.nan] * (result['timeindex'].iloc[0] - 1),
            'var': [result['var'].iloc[0]] * (result['timeindex'].iloc[0] - 1),
            'frequency': [result['frequency'].iloc[0]] * (result['timeindex'].iloc[0] - 1)
        })

        # Substituir células vazias por "NA"
        additional_rows = additional_rows.replace(np.nan, 'NA')

        # Combinar o DataFrame original com as linhas adicionais
        result = pd.concat([additional_rows, result]).reset_index(drop=True)

        # Caminho para o novo arquivo CSV
        new_file_path = os.path.join(output_folder, f"new_{csv_file}")

        # Salvar o novo DataFrame em um novo arquivo CSV
        result.to_csv(new_file_path, index=False)
    else:
        print(f'O índice de tempo é 1 para o arquivo {csv_file}')


