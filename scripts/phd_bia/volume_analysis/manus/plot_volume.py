# -*- coding: utf-8 -*-
"""
Script para visualizar e analisar o espaço funcional 3D de comunidades vegetais.

Este script lê dados de atributos funcionais (g1, SLA, densidade da madeira)
 de indivíduos de plantas de um arquivo CSV, gera um gráfico de dispersão 3D
 interativo usando Plotly, e calcula o volume do casco convexo (convex hull)
 para cada comunidade (condição climática) como uma medida do volume funcional ocupado.
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull
import sys

# --- Configurações (AJUSTE CONFORME SEUS DADOS) ---

# Nome do seu arquivo CSV
NOME_ARQUIVO_CSV = 'seus_dados.csv' # !! SUBSTITUA PELO NOME DO SEU ARQUIVO !!

# Nomes exatos das colunas no seu arquivo CSV
COLUNA_G1 = 'g1' # Coluna com os valores de g1
COLUNA_SLA = 'SLA' # Coluna com os valores de SLA
COLUNA_DENSIDADE = 'wood_density' # Coluna com os valores de densidade da madeira
COLUNA_CONDICAO = 'condicao_climatica' # Coluna que identifica a condição (ex: 'Regular', 'Reduzida')

# Valores exatos que identificam as duas condições climáticas na COLUNA_CONDICAO
# !! AJUSTE ESTES VALORES PARA CORRESPONDER AOS SEUS DADOS !!
CONDICAO_1 = 'Regular' 
CONDICAO_2 = 'Precipitacao Reduzida'

# Título do gráfico
TITULO_GRAFICO = 'Espaço Funcional 3D e Volume por Condição Climática'

# Nome do arquivo HTML de saída para o gráfico interativo
NOME_ARQUIVO_HTML = 'espaco_funcional_3d_com_hulls.html'

# --- Funções Auxiliares ---

def calcular_volume_convex_hull(dataframe, col_x, col_y, col_z):
    """Calcula o volume do casco convexo 3D para os pontos no dataframe."""
    try:
        pontos = dataframe[[col_x, col_y, col_z]].values
        # O Convex Hull precisa de pelo menos 4 pontos não coplanares em 3D
        if len(pontos) < 4:
            print(f"Aviso: Menos de 4 pontos para calcular o casco convexo. Volume será 0.")
            return 0.0
        hull = ConvexHull(pontos)
        return hull.volume
    except Exception as e:
        # Pode falhar se os pontos forem coplanares ou colineares
        print(f"Aviso: Não foi possível calcular o volume do casco convexo (pontos podem ser coplanares/colineares). Erro: {e}. Volume será 0.")
        return 0.0

# --- Leitura e Processamento dos Dados ---

try:
    df = pd.read_csv(NOME_ARQUIVO_CSV)
    print(f"Arquivo '{NOME_ARQUIVO_CSV}' lido com sucesso.")
    print(f"Colunas encontradas: {df.columns.tolist()}")

    colunas_necessarias = [COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE, COLUNA_CONDICAO]
    colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
    if colunas_faltantes:
        print(f"Erro: As seguintes colunas não foram encontradas: {colunas_faltantes}")
        sys.exit(1)

    # Verifica se os valores das condições existem
    condicoes_presentes = df[COLUNA_CONDICAO].unique()
    if CONDICAO_1 not in condicoes_presentes:
        print(f"Erro: O valor '{CONDICAO_1}' não foi encontrado na coluna '{COLUNA_CONDICAO}'. Valores presentes: {condicoes_presentes}")
        sys.exit(1)
    if CONDICAO_2 not in condicoes_presentes:
        print(f"Erro: O valor '{CONDICAO_2}' não foi encontrado na coluna '{COLUNA_CONDICAO}'. Valores presentes: {condicoes_presentes}")
        sys.exit(1)

except FileNotFoundError:
    print(f"Erro: Arquivo '{NOME_ARQUIVO_CSV}' não encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"Ocorreu um erro inesperado ao ler o arquivo CSV: {e}")
    sys.exit(1)

# --- Separação dos Dados por Condição ---
df_condicao1 = df[df[COLUNA_CONDICAO] == CONDICAO_1]
df_condicao2 = df[df[COLUNA_CONDICAO] == CONDICAO_2]

print(f"Número de indivíduos em '{CONDICAO_1}': {len(df_condicao1)}")
print(f"Número de indivíduos em '{CONDICAO_2}': {len(df_condicao2)}")

# --- Cálculo dos Volumes Funcionais (Convex Hull) ---

print("\nCalculando volumes funcionais (casco convexo)...\n")

volume_condicao1 = calcular_volume_convex_hull(df_condicao1, COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE)
volume_condicao2 = calcular_volume_convex_hull(df_condicao2, COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE)

print(f"Volume funcional estimado para '{CONDICAO_1}': {volume_condicao1:.4f}")
print(f"Volume funcional estimado para '{CONDICAO_2}': {volume_condicao2:.4f}")

if volume_condicao1 > 0 and volume_condicao2 > 0:
    ratio = volume_condicao2 / volume_condicao1
    print(f"Razão do Volume ({CONDICAO_2} / {CONDICAO_1}): {ratio:.4f}")
    if ratio < 1:
        print(f"Sugestão: O volume funcional sob '{CONDICAO_2}' é {((1-ratio)*100):.2f}% menor que sob '{CONDICAO_1}'.")
    elif ratio > 1:
        print(f"Sugestão: O volume funcional sob '{CONDICAO_2}' é {((ratio-1)*100):.2f}% maior que sob '{CONDICAO_1}'.")
    else:
        print("Sugestão: Os volumes funcionais estimados são iguais.")

# --- Criação da Visualização 3D com Hulls (Opcional) ---

print("\nGerando o gráfico 3D com pontos e cascos convexos...")

fig = go.Figure()

# Adiciona pontos da Condição 1
fig.add_trace(go.Scatter3d(
    x=df_condicao1[COLUNA_G1],
    y=df_condicao1[COLUNA_SLA],
    z=df_condicao1[COLUNA_DENSIDADE],
    mode='markers',
    marker=dict(size=4, color='blue', opacity=0.6),
    name=CONDICAO_1
))

# Adiciona pontos da Condição 2
fig.add_trace(go.Scatter3d(
    x=df_condicao2[COLUNA_G1],
    y=df_condicao2[COLUNA_SLA],
    z=df_condicao2[COLUNA_DENSIDADE],
    mode='markers',
    marker=dict(size=4, color='red', opacity=0.6),
    name=CONDICAO_2
))

# Adiciona o casco convexo da Condição 1 (se calculável)
try:
    pontos_c1 = df_condicao1[[COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE]].values
    if len(pontos_c1) >= 4:
        hull_c1 = ConvexHull(pontos_c1)
        fig.add_trace(go.Mesh3d(
            x=pontos_c1[:, 0],
            y=pontos_c1[:, 1],
            z=pontos_c1[:, 2],
            i=hull_c1.simplices[:, 0],
            j=hull_c1.simplices[:, 1],
            k=hull_c1.simplices[:, 2],
            opacity=0.1,
            color='blue',
            flatshading=True,
            name=f'Hull {CONDICAO_1}'
        ))
except Exception as e:
    print(f"Não foi possível adicionar o casco convexo para '{CONDICAO_1}' ao gráfico. Erro: {e}")

# Adiciona o casco convexo da Condição 2 (se calculável)
try:
    pontos_c2 = df_condicao2[[COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE]].values
    if len(pontos_c2) >= 4:
        hull_c2 = ConvexHull(pontos_c2)
        fig.add_trace(go.Mesh3d(
            x=pontos_c2[:, 0],
            y=pontos_c2[:, 1],
            z=pontos_c2[:, 2],
            i=hull_c2.simplices[:, 0],
            j=hull_c2.simplices[:, 1],
            k=hull_c2.simplices[:, 2],
            opacity=0.1,
            color='red',
            flatshading=True,
            name=f'Hull {CONDICAO_2}'
        ))
except Exception as e:
    print(f"Não foi possível adicionar o casco convexo para '{CONDICAO_2}' ao gráfico. Erro: {e}")

# Atualiza o layout
fig.update_layout(
    title=TITULO_GRAFICO,
    scene=dict(
        xaxis_title=COLUNA_G1,
        yaxis_title=COLUNA_SLA,
        zaxis_title=COLUNA_DENSIDADE
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# --- Salvando o Gráfico Interativo ---
try:
    fig.write_html(NOME_ARQUIVO_HTML)
    print(f"\nGráfico 3D salvo com sucesso como '{NOME_ARQUIVO_HTML}'.")
    print("Abra este arquivo em um navegador web para explorar o gráfico interativamente.")
except Exception as e:
    print(f"Ocorreu um erro ao salvar o gráfico: {e}")
    sys.exit(1)


