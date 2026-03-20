import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Definindo o estilo dos plots
sns.set_style("whitegrid")

# Carregar os dataframes
df_regclim = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_regularclimate_yearly.csv")
df_1y = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_30prec_1y_yearly.csv")
df_7y = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/yearly_mean_tables/MAN_30prec_7y_yearly.csv")

# Variáveis de interesse
variaveis = ['npp', 'ctotal', 'evapm','ls', "wue"]

# Mapear os nomes das variáveis para os rótulos desejados
variaveis_label = {
    'npp': 'NPP',
    'ctotal': 'Total Carbon',
    'evapm': 'Evapotranspiration',
    'ls': 'Num. PLSs',
    'wue': 'WUE'
}

# Calcula a diferença relativa das variáveis de interesse ao longo do tempo
for var in variaveis:
    df_1y[var + '_diff_relative'] = (df_1y[var] - df_regclim[var]) / df_regclim[var] * 100
    df_7y[var + '_diff_relative'] = (df_7y[var] - df_regclim[var]) / df_regclim[var] * 100

# Define o tamanho da figura
plt.figure(figsize=(10, 6))

# Plota a diferença relativa das variáveis de interesse ao longo do tempo
for var in variaveis:
    sns.lineplot(data=df_7y, x='date', y=var + '_diff_relative', label=variaveis_label[var])

plt.xlabel('Year', fontsize = 14)
plt.ylabel('Relative difference (%)',fontsize = 14)
plt.title('Reduced precipitation frequency: 7 years', fontsize = 14)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

# Define o tamanho da figura
plt.figure(figsize=(10, 6))

# Variáveis de interesse
variaveis = ['npp', 'ctotal', 'evapm','ls']
for var in variaveis:
    sns.lineplot(data=df_1y, x='date', y=var + '_diff_relative', label=variaveis_label[var])

plt.xlabel('Year', fontsize = 14)
plt.ylabel('Relative difference (%)',fontsize = 14)
plt.title('Reduced precipitation frequency: 1 year', fontsize = 14)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()

# Define o tamanho da figura
plt.figure(figsize=(10, 6))

# Plot para 'WUE' com cor personalizada
df_1y['wue_diff_relative'] = (df_1y['wue'] - df_regclim['wue']) / df_regclim['wue'] * 100
sns.lineplot(data=df_1y, x='date', y='wue_diff_relative', label='WUE', color='#9370DB')

plt.xlabel('Year',fontsize = 14)
plt.ylabel('Relative difference (%)',fontsize = 14)
plt.title('Reduced precipitation frequency: 1 year', fontsize = 14)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=14)
plt.show()
