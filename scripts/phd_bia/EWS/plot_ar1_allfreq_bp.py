import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

df = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/results_csv/MAN_30prec_allfreq_bp1y_ews.csv")


tau_values = {'1': 0.406, '3': 0.445, '5': 0.292, '7': 0.269, 'regularclimate': 0.334}
df['tau'] = df['frequency'].map(tau_values)

# Mapear as cores para cada frequência
color_mapping = {
    '1': 'orange',
    '3': 'green',
    '5': 'red',
    '7': 'purple',
    'regularclimate': 'blue'
}

# Plotar os dados com a legenda personalizada
plt.figure(figsize=(10, 6))

for freq, group_df in df.groupby('frequency'):
    plt.plot(group_df['timeindex'], group_df['ar1'], label=f"${freq}$ (τ = {tau_values[freq]:.3f})", color=color_mapping[freq])

plt.title('AR1 - Manaus - 30% prec reduction')
plt.xlabel('Time Index')
plt.ylabel('AR1')
plt.legend()

plt.savefig("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/ar1_allfreq_bp1y.png")
plt.show()