
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Carregar dados
df_1y = pd.read_csv('/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/monthly_mean_tables/MAN_30prec_1y_monthly.csv')
df_1y['date'] = pd.to_datetime(df_1y['date'])

# Ordenar o DataFrame por data
df_1y = df_1y.sort_values(by='date')

# Inicializar listas para armazenar os resultados do ajuste
dates_fit = []
npp_fit_values = []

# Encontrar os índices de dezembro para o início e o fim da série temporal
start_index = df_1y[df_1y['date'].dt.month == 12].index[0]
end_index = df_1y[df_1y['date'].dt.month == 12].index[-1]

# Iterar sobre a série temporal em intervalos de dois anos
for i in range(start_index, end_index, 24):  # 24 meses correspondem a dois anos
    # Subconjunto de dados a partir de dezembro de um ano até dezembro do próximo ano
    subset = df_1y.iloc[i:i + 24]  # 24 meses correspondem a dois anos
    
    # Ajustar uma curva quadrática ao subconjunto
    coefficients = np.polyfit(np.arange(len(subset)), subset['npp'], 2)
    quadratic_fit = np.poly1d(coefficients)
    
    # Adicionar pontos ajustados à lista
    dates_fit.extend(subset['date'])
    npp_fit_values.extend(quadratic_fit(np.arange(len(subset))))

# Plotar os resultados
plt.figure(figsize=(10, 6))
plt.plot(df_1y['date'], df_1y['npp'], label='Data', color = 'grey')
plt.plot(dates_fit, npp_fit_values, label='Baseline', linestyle='--', color='black')
plt.xlabel('Data')
plt.ylabel('NPP (Net Primary Productivity)')
plt.title('prec. reduc: 30 - Frequency: 1 year - Manaus')
plt.legend()
plt.grid(True)
plt.show()


