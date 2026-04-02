import h5py
import csv
import numpy as np


# # Salvar
# with h5py.File('arquivo.h5', 'w') as f:
#     for key, value in data.items():
#         if isinstance(value, np.ndarray):
#             f.create_dataset(key, data=value)
#         else:
#             f.attrs[key] = value

data = {}


input_h5 = 'outputs/lu0104/gridcell186-239/spin01.h5'

# Carregar
with h5py.File(input_h5, 'r') as f:
    # Para datasets
    for key in f.keys():
        data[key] = f[key][:]
    # Para atributos
    for key in f.attrs:
        data[key] = f.attrs[key]

# print(data)

data_obj = data


print(data_obj.keys())


def to_list(value):
    """Converte qualquer valor para lista, mantendo estrutura"""
    if value is None:
        return [None]
    elif isinstance(value, (list, tuple)):
        return list(value) if len(value) > 0 else [None]
    elif isinstance(value, np.ndarray):
        return value.tolist() if value.size > 0 else [None]
    elif isinstance(value, (int, float, np.integer, np.floating, str)):
        return [value]
    else:
        return [value]
    
for k in data_obj.keys():
    li = to_list(data_obj[k])
    print("---------------")
    print(k)
    print(f"len: {len(li)}")
    print(f"type: {type(li)}")
    print(f"type element: {type(li[0])}")
    if (k == "area"):
        print(f"len element: {len(li[0])}")
    #     print(type(li[0]])

# max_len = (.values() for k in )
# print(max_len[0])

# for key, values in data.items():
#     if len(values) < max_len:
#         # Se for menor, repete o último valor ou preenche com None
#         data[key] = values + [values[-1] if values else None] * (max_len - len(values))


# with open(input_h5.replace(".h5",".csv"), 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=data_obj.keys())
#     writer.writeheader()
#     writer.writerow(data_obj)