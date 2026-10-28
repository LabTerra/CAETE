from PIL import Image
import os

# Especifica o caminho para a pasta que contém os arquivos PNG
folder_path = '/home/bianca/bianca/CAETE-DVM-alloc-allom/scripts/EWS/csv_allvar'

# Lista todos os arquivos na pasta
file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# Cria uma lista de objetos de imagem
images = [Image.open(os.path.join(folder_path, file)) for file in file_list]

# Define o número de colunas e linhas desejado
num_columns = 2
num_rows = 7

# Calcula as dimensões da imagem combinada
width, height = images[0].size
combined_width = num_columns * width
combined_height = num_rows * height

# Cria uma nova imagem grande para combinar todas as outras
combined_image = Image.new('RGB', (combined_width, combined_height))

# Cola cada imagem na imagem combinada, seguindo o layout especificado
for i, image in enumerate(images):
    col = i % num_columns
    row = i // num_columns
    combined_image.paste(image, (col * width, row * height))

# Salva a imagem combinada com uma qualidade mais alta (por exemplo, 90)
combined_image.save(os.path.join(folder_path, 'combined_output.png'))
