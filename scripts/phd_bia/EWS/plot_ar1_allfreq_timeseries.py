import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/results_csv/MAN_30prec_allfreq_timeseries_ews.csv")

# Filtrar para os anos de 1980, 1981 e 1982
df_1980 = df[df['Dates'].str.startswith('1980')]
df_1981 = df[df['Dates'].str.startswith('1981')]
df_1983 = df[df['Dates'].str.startswith('1982')]

freq = df['frequency']
tau = ['1', '2', '3', '4', '5']

fig, axs = plt.subplots(5, 1, figsize=(8, 4 * 5), sharex=True)

for i, f in enumerate(freq.unique()):
    subset_df = df[df['frequency'] == f]
    
        
    axs[i].set_ylabel('AR1')  # Adicionar rótulo ao eixo y
    
    # Adicionar linha tracejada vertical quando a frequência é "regularclimate" e o timeindex é 362
    if f == 'regularclimate':
        tau = 0.037
        axs[i].axvline(x=362, linestyle='--', color='grey')
        axs[i].plot(subset_df['timeindex'], subset_df['ar1'], label=f'Frequency {f}\n\n tau= {tau}', color='black')
        
    elif f == '1':
        tau = 0.434
        axs[i].axvline(x=99, linestyle='--', color='grey')
        axs[i].axvline(x=167, linestyle='--', color='grey')
        axs[i].axvline(x=244, linestyle='--', color='grey')
        axs[i].axvline(x=334, linestyle='--', color='grey')
        
        axs[i].plot(label=f'{tau}')
        axs[i].plot(subset_df['timeindex'], subset_df['ar1'], label=f'Frequency {f}\n\n tau= {tau}', color='black')
        # Adicionar faixas vermelhas para os ranges de 1980, 1982, 1984, ..., 2016
        for year in range(1980, 2017, 2):
            df_year = df[df['Dates'].str.startswith(str(year))]
            axs[i].axvspan(df_year['timeindex'].min(), df_year['timeindex'].max(), color='red', alpha=0.2)
            
    elif f == '3':
        tau = 0.534
        axs[i].axvline(x=154, linestyle='--', color='grey')
        axs[i].axvline(x=222, linestyle='--', color='grey')
        axs[i].axvline(x=374, linestyle='--', color='grey')
        axs[i].plot(subset_df['timeindex'], subset_df['ar1'], label=f'Frequency {f}\n\n tau= {tau}', color='black')
        # Adicionar faixas vermelhas para os ranges de 1980, 1981 e 1982
        for year in range(1980, 2017, 4):
            df_year = df[df['Dates'].str.startswith(str(year))]
            axs[i].axvspan(df_year['timeindex'].min(), df_year['timeindex'].max(), color='red', alpha=0.2)
            
    elif f == '5':
        tau = 0.721
        axs[i].axvline(x=202, linestyle='--', color='grey')
        axs[i].axvline(x=298, linestyle='--', color='grey')
        axs[i].axvline(x=366, linestyle='--', color='grey')
        axs[i].plot(subset_df['timeindex'], subset_df['ar1'], label=f'Frequency {f}\n\n tau= {tau}', color='black')
        for year in range(1980, 2017, 6):
            df_year = df[df['Dates'].str.startswith(str(year))]
            axs[i].axvspan(df_year['timeindex'].min(), df_year['timeindex'].max(), color='red', alpha=0.2)
    elif f == '7':
        tau = 0.774
        axs[i].axvline(x=300, linestyle='--', color='grey')
        axs[i].axvline(x=368, linestyle='--', color='grey')
        axs[i].plot(subset_df['timeindex'], subset_df['ar1'], label=f'Frequency {f}\n\n tau= {tau}', color='black')
        for year in range(1980, 2017, 8):
            df_year = df[df['Dates'].str.startswith(str(year))]
            axs[i].axvspan(df_year['timeindex'].min(), df_year['timeindex'].max(), color='red', alpha=0.2)
    axs[i].legend()
    

# Ajustar os eixos y para variar de 0 a 1
for ax in axs:
    ax.set_ylim(0, 1)

# Adicionar legenda no eixo x apenas no último subplot
axs[-1].set_xlabel('Time index')
axs[-1].get_xaxis().set_label_coords(0.5, -0.15)  # Ajuste da posição do rótulo no eixo x


plt.suptitle('AR(1) Time series - 30%prec reduction - Manaus', y=1.02)
plt.tight_layout()
plt.savefig("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/ar1_allfreq_complete_timeseries.png")

plt.show()

print(df['timeindex'])
