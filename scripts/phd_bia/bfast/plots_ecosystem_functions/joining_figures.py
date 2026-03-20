import pandas as pd
from PIL import Image

df = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/image_paths.csv")

# Criação da coluna 'image_path'
base_path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/bfast/plots_ecosystem_functions/'
df['image_path'] = base_path + df['init'] + '_' + df['climate'] + '_' + df['var'] + '.png'

# Tamanho desejado para cada imagem na grade final
image_size = (600, 600)

# Organizar o DataFrame para criar uma imagem por 'var' e 'climate'
grouped_df = df.groupby(['var', 'climate'])

# Número de colunas e linhas na imagem final
num_cols = len(df['climate'].unique())
num_rows = len(df['var'].unique())

# Criação de uma imagem em branco com as dimensões desejadas
final_image = Image.new('RGB', (num_cols * image_size[0], num_rows * image_size[1]))

# Adição das imagens à imagem final
for i, ((var, climate), group) in enumerate(grouped_df):
    img_path = group['image_path'].values[0]  # Pega apenas uma imagem para cada combinação única
    img = Image.open(img_path)
    img = img.resize(image_size)  # Ajuste o tamanho conforme necessário
    final_image.paste(img, ((i % num_cols) * image_size[0], (i // num_cols) * image_size[1]))

# Salvar a imagem final
final_image.save(f'{base_path}/allplots.png')
