"""

Code documentation:

The clean_run function performs the following actions:

    1. Save Current Conditions:
Before starting a new round or experiment, the function saves the current conditions associated with a save_id identifier.
    2. Check Output Directory Existence:
Checks if the output directory associated with the new experiment already exists. If the directory already exists, the code takes measures to avoid overwriting.
    3. Revert Changes in Case of Failure:
If the attempt to create the directory fails due to its preexistence, the code reverts the changes made earlier (restoring self.out_dir to its original value) and raises an exception.
    4. Record Current Conditions:
Adds a pair (save_id, self.outputs.copy()) to the realized_runs list, thus recording the current state of the simulation.
    5. Clear Attributes for the New Round:
Resets relevant attributes for the simulation, such as self.outputs and self.run_counter, preparing for a new simulation round.

This function appears prepare for a new round of simulation, ensuring the integrity of saved data and avoiding accidental overwriting of previous results.


"""
import os
import shutil
import multiprocessing as mp
from pathlib import Path
import joblib
import numpy as np
import csv
from parameters import base_run, ATTR_FILENAME, run_path, pls_path

assert run_path.exists(), "Wrong path to initial conditions"
assert pls_path.exists(), "Wrong path to Attributes Table"

# Open the binary file at 'run_path' for reading ('rb')
with open(run_path, 'rb') as fh:
    # Load data from the file using the joblib library
    init_conditions = joblib.load(fh)
    #init_conditions contais all the attributes and methods of caete.grd
    
all_attributes_and_methods = dir(init_conditions)



experiment = input("regular climate(a) or experiment(b)? ")

if experiment == 'a':
    print('')
    print('')
    print("YOU ARE RUNNING WITH REGULAR CLIMATE")
    print('')
    print('')

    application_interval = '0'
else:
    print('')
    print('')
    print("YOU ARE RUNNING AN EXPERIMENT")
    print('')
    print('')
    
    application_interval = input('Which is the interval between the applications? [1, 3, 5, 7] ')

interval = int(application_interval) + 1 #+1 guarantee the right interval between applications
while True:
    perc_prec = input("What is the percentage of reduction? [0, 10, 20, 30] ")
    
    if perc_prec == '0':
        print('You are not applying precipitation reduction')
        #10% precipitation reduction
        prec_red = 1.
        break

    if perc_prec == '10':
        print('You are applying 10% of precipitation reduction')
        #10% precipitation reduction
        prec_red = 0.9
        break

    elif perc_prec == '20':
        print('You are applying 20% of precipitation reduction')
        #20% precipitation reduction
        prec_red = 0.8
        break

    elif perc_prec == '30':
        print('You are applying 30% of precipitation reduction')
        #30% precipitation reduction
        prec_red = 0.7
        if prec_red == 0.7:
            print('')
            print('OK',prec_red)
            print('OK',prec_red)
            print('OK',prec_red)
            print('')
        else:
            print('WRONG')
        break
    
    else:
        print('Reduction out of range -- CANCELLING')
        pass
# new outputs folder
run_name = input(f"Give a name to this output: ")
dump_folder = Path(f"{run_name}")

for gridcell in init_conditions:
    gridcell.clean_run(dump_folder, "init_cond")
    

def zip_gridtime(grd_pool, interval):
    res = []
    for i, j in enumerate(grd_pool):
        res.append((j, interval[i % len(interval)]))
    return res


# pr_values = []     
# Loop para executar para cada ano de '19790101' a '20161231'
for year in range(1979, 2017):
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    # pr_values.append(gridcell.pr)
    # print(f"Ano: {year}, Comprimento de pr_values: {len(pr_values)}")


    # Application for a whole year in the set interval
    if (year % interval == 0) and (experiment =='a'):
        print(f"running the model in {year}")
        gridcell.pr = gridcell.pr * prec_red
    
    elif (year >= 1980) and (year % interval == 1980 % interval):  #garante que o primeiro ano não seja afetado
        print(f"applying the disturbance in {year}")
        gridcell.pr = gridcell.pr * prec_red

    gridcell.run_caete_allom(start_date, end_date)
        
# print("Total de Anos:", len(pr_values))

# dados_a_salvar =pr_values

# Caminho do arquivo CSV
# caminho_arquivo_csv = '/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/pr_values.csv'

#run model

# # Escrever no arquivo CSV
# with open(caminho_arquivo_csv, mode='w', newline='') as arquivo_csv:
#     escritor_csv = csv.writer(arquivo_csv)
    
#     # Escrever cabeçalho, se necessário
#     # escritor_csv.writerow(['Ano', 'Valor_PR'])

#     escritor_csv.writerow('Valor_prec')    
#     # Concatenar os valores do array em uma única lista
#     valores_concatenados = [item for sublist in pr_values for item in sublist]
    
#     # Escrever dados
#     escritor_csv.writerow(valores_concatenados)

# print(f"Valores de 'pr_values' foram salvos em '{caminho_arquivo_csv}'.")
