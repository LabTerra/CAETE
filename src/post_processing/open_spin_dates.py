import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

run_name = input('run name: ')
spin = input('which spin?')

path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/outputs/{}/gridcell175-235'.format(run_name)

# Navegue para a pasta
os.chdir(path)

# Carregue os dados
with open("spin{}.pkz".format(spin), 'rb') as fh:
    dt = joblib.load(fh)

# Crie uma lista de datas para o período desejado
data_inicial = datetime(2015, 1, 1)
datas = [data_inicial + timedelta(days=i) for i in range(len(dt['npp']))]

# Configure o formato de data para o eixo x
fig, ax = plt.subplots(figsize=(10, 6))
fig.autofmt_xdate()  # Ajusta automaticamente o formato de data

# # Plote a variável NPP com datas no eixo x
# ax.plot(datas, dt['npp'])
# ax.set_title('NPP')
# ax.xaxis.set_major_locator(mdates.YearLocator())  # Coloca um rótulo para cada ano
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato de data

# # Plote a variável NPP com datas no eixo x
ax.plot(datas, dt['photo'])
ax.set_title('GPP')
ax.xaxis.set_major_locator(mdates.YearLocator())  # Coloca um rótulo para cada ano
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato de data

# # Plote a variável NPP com datas no eixo x
# ax.plot(datas, dt['ar'])
# ax.set_title('ar')
# ax.xaxis.set_major_locator(mdates.YearLocator())  # Coloca um rótulo para cada ano
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato de data

plt.tight_layout()
plt.show()


# import joblib
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# run_name = input('run name: ')
# spin = input('which spin?')

# path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/outputs/{}/gridcell175-235'.format(run_name)

# # Navegue para a pasta
# os.chdir(path)

# # Carregue os dados
# with open("spin{}.pkz".format(spin), 'rb') as fh:
#     dt = joblib.load(fh)

# # Selecione apenas os primeiros 365 dados da variável 'NPP'
# npp_data = dt['npp'][:365]

# # Crie um índice de datas correspondentes aos primeiros 365 dias do ano de 2015
# start_date = '2015-01-01'
# end_date = '2015-12-31'
# date_index = pd.date_range(start=start_date, end=end_date)

# # Plotar gráfico para os primeiros 365 dados de 'NPP' com rótulos mensais
# plt.plot(date_index, npp_data)
# plt.title('NPP - Ano de 2015')
# plt.xlabel('Data')
# plt.ylabel('Valor')
# plt.xticks(pd.date_range(start=start_date, end=end_date, freq='M'), [month.strftime('%B') for month in pd.date_range(start=start_date, end=end_date, freq='M')], rotation=45)
# plt.show()
