import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Carregar os dataframes
df_regclim = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/traits_weightedmean_regclim.csv")
df_1y = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/traits_weightedmean_1y.csv")
df_7y = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/traits/traits_weightedmean_7y.csv")

# # Variáveis de interesse
variaveis = ['weighted_mean_wd','weighted_mean_g1','weighted_mean_sla']
time = df_regclim['YEAR']

# Função para normalizar os dados
def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

# Normalizar os dados para cada DataFrame
df_regclim_norm = normalize(df_regclim[variaveis])
df_1y_norm = normalize(df_1y[variaveis])
df_7y_norm = normalize(df_7y[variaveis])

# Calcula a diferença relativa das variáveis de interesse ao longo do tempo
for var in variaveis:
    df_1y[var + '_diff_relative'] = (df_1y[var] - df_regclim[var]) / df_regclim[var] * 100
   
    df_7y[var + '_diff_relative'] = (df_7y[var] - df_regclim[var]) / df_regclim[var] * 100
 

# Define o tamanho da figura
plt.figure(figsize=(10, 6))

# Plota a diferença relativa das variáveis de interesse ao longo do tempo
for var in variaveis:
    sns.lineplot(data=df_7y, x='YEAR', y=var + '_diff_relative', label=var)
    # Imprime a diferença relativa para cada YEAR
    for year in df_7y['YEAR'].unique():
        diff_relative_year = df_7y[df_7y['YEAR'] == year][var + '_diff_relative'].iloc[0]
        # print(f"Variável: {var}, YEAR: {year}, Diferença relativa: {diff_relative_year}")
        
# Encontrar o menor e o maior valor de diferença relativa para cada variável
for var in variaveis:
    min_diff = df_7y[var + '_diff_relative'].min()
    max_diff = df_7y[var + '_diff_relative'].max()
    print(f"Freq 7: Variável: {var}, Menor diferença relativa: {min_diff}, Maior diferença relativa: {max_diff}")

    min_diff = df_1y[var + '_diff_relative'].min()
    max_diff = df_1y[var + '_diff_relative'].max()
    print(f"Freq 1: Variável: {var}, Menor diferença relativa: {min_diff}, Maior diferença relativa: {max_diff}")

plt.xlabel('Year', fontsize = 14)
plt.ylabel('Relative difference (%)',fontsize = 14)
plt.title('Reduced precipitation frequency: 7 years', fontsize = 14)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.show()

# Define o tamanho da figura
plt.figure(figsize=(10, 6))

for var in variaveis:
    sns.lineplot(data=df_1y, x='YEAR', y=var + '_diff_relative', label=var)

plt.xlabel('Year', fontsize = 14)
plt.ylabel('Relative difference (%)',fontsize = 14)
plt.title('Reduced precipitation frequency: 1 year', fontsize = 14)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=14)
# plt.show()

# Filtrar os dados para o ano de 2005
df_7y_2005 = df_7y[df_7y['YEAR'] == 2005]
df_1y_2005 = df_1y[df_1y['YEAR'] == 2005]
# Calcular a diferença relativa das variáveis de interesse para o ano de 2005
for var in variaveis:
    diff_relative_2005 = df_7y_2005[var + '_diff_relative'].iloc[0]
    print(f"7y; Variável: {var}, YEAR: 2005, Diferença relativa: {diff_relative_2005}")

    diff_relative_2005 = df_1y_2005[var + '_diff_relative'].iloc[0]
    print(f"1y; Variável: {var}, YEAR: 2005, Diferença relativa: {diff_relative_2005}")
