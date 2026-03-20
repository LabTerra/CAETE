# -*- coding: utf-8 -*-
"""
Script para visualizar e analisar o espaço funcional 3D de comunidades vegetais.

Este script lê dados de atributos funcionais (g1, SLA, densidade da madeira)
 de indivíduos de plantas de um arquivo CSV, **padroniza os dados (média 0, desvio padrão 1)**,
 gera um gráfico de dispersão 3D interativo usando Plotly com os dados padronizados,
 e calcula o volume do casco convexo (convex hull) para cada comunidade
 (condição climática) usando os dados padronizados como uma medida do volume funcional ocupado.
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler # Importa o StandardScaler
import sys

# --- Configurações (AJUSTE CONFORME SEUS DADOS) ---

# Nome do seu arquivo CSV
NOME_ARQUIVO_CSV = 'seus_dados.csv' # !! SUBSTITUA PELO NOME DO SEU ARQUIVO !!

# Nomes exatos das colunas no seu arquivo CSV
COLUNA_G1 = 'g1' # Coluna com os valores de g1
COLUNA_SLA = 'SLA' # Coluna com os valores de SLA
COLUNA_DENSIDADE = 'wood_density' # Coluna com os valores de densidade da madeira
COLUNA_CONDICAO = 'condicao_climatica' # Coluna que identifica a condição

# Valores exatos que identificam as duas condições climáticas na COLUNA_CONDICAO
# !! AJUSTE ESTES VALORES PARA CORRESPONDER AOS SEUS DADOS !!
CONDICAO_1 = 'Regular'
CONDICAO_2 = 'Precipitacao Reduzida'

# Título do gráfico
TITULO_GRAFICO = 'Espaço Funcional 3D (Dados Padronizados) e Volume por Condição Climática'

# Nome do arquivo HTML de saída para o gráfico interativo
NOME_ARQUIVO_HTML = 'espaco_funcional_3d_padronizado_com_hulls.html'

# --- Funções Auxiliares ---

def calcular_volume_convex_hull(dataframe, col_x, col_y, col_z):
    """Calcula o volume do casco convexo 3D para os pontos no dataframe."""
    try:
        pontos = dataframe[[col_x, col_y, col_z]].values
        if len(pontos) < 4:
            print(f"Aviso: Menos de 4 pontos para calcular o casco convexo. Volume será 0.")
            return 0.0
        hull = ConvexHull(pontos)
        return hull.volume
    except Exception as e:
        print(f"Aviso: Não foi possível calcular o volume do casco convexo (pontos podem ser coplanares/colineares). Erro: {e}. Volume será 0.")
        return 0.0

# --- Leitura e Processamento dos Dados ---

try:
    df = pd.read_csv(NOME_ARQUIVO_CSV)
    print(f"Arquivo '{NOME_ARQUIVO_CSV}' lido com sucesso.")

    colunas_necessarias = [COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE, COLUNA_CONDICAO]
    colunas_faltantes = [col for col in colunas_necessarias if col not in df.columns]
    if colunas_faltantes:
        print(f"Erro: As seguintes colunas não foram encontradas: {colunas_faltantes}")
        sys.exit(1)

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

# --- Padronização dos Dados Numéricos ---

print("\nPadronizando os dados (g1, SLA, wood_density)...\n")
colunas_para_padronizar = [COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE]
scaler = StandardScaler()

# Cria um novo DataFrame com os dados padronizados
df_scaled = df.copy()
df_scaled[colunas_para_padronizar] = scaler.fit_transform(df[colunas_para_padronizar])

print("Dados padronizados. As colunas agora representam Z-scores.")
print("Primeiras 5 linhas dos dados padronizados:")
print(df_scaled.head())

# --- Separação dos Dados Padronizados por Condição ---
# !! IMPORTANTE: Usar o DataFrame padronizado (df_scaled) daqui em diante para visualização e volume !!
df_condicao1_scaled = df_scaled[df_scaled[COLUNA_CONDICAO] == CONDICAO_1]
df_condicao2_scaled = df_scaled[df_scaled[COLUNA_CONDICAO] == CONDICAO_2]

print(f"\nNúmero de indivíduos em '{CONDICAO_1}': {len(df_condicao1_scaled)}")
print(f"Número de indivíduos em '{CONDICAO_2}': {len(df_condicao2_scaled)}")

# --- Cálculo dos Volumes Funcionais (Convex Hull) com Dados Padronizados ---

print("\nCalculando volumes funcionais (casco convexo) usando DADOS PADRONIZADOS...\n")

# Usa as colunas padronizadas para o cálculo
volume_condicao1 = calcular_volume_convex_hull(df_condicao1_scaled, COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE)
volume_condicao2 = calcular_volume_convex_hull(df_condicao2_scaled, COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE)

print(f"Volume funcional padronizado estimado para '{CONDICAO_1}': {volume_condicao1:.4f}")
print(f"Volume funcional padronizado estimado para '{CONDICAO_2}': {volume_condicao2:.4f}")

if volume_condicao1 > 0 and volume_condicao2 > 0:
    ratio = volume_condicao2 / volume_condicao1
    print(f"Razão do Volume Padronizado ({CONDICAO_2} / {CONDICAO_1}): {ratio:.4f}")
    if ratio < 1:
        print(f"Sugestão: O volume funcional padronizado sob '{CONDICAO_2}' é {((1-ratio)*100):.2f}% menor que sob '{CONDICAO_1}'.")
    elif ratio > 1:
        print(f"Sugestão: O volume funcional padronizado sob '{CONDICAO_2}' é {((ratio-1)*100):.2f}% maior que sob '{CONDICAO_1}'.")
    else:
        print("Sugestão: Os volumes funcionais padronizados estimados são iguais.")
else:
    print("Não foi possível calcular a razão dos volumes (um ou ambos são zero).")

# --- Criação da Visualização 3D com Dados Padronizados e Hulls ---

print("\nGerando o gráfico 3D com pontos padronizados e cascos convexos...")

fig = go.Figure()

# Adiciona pontos da Condição 1 (padronizados)
fig.add_trace(go.Scatter3d(
    x=df_condicao1_scaled[COLUNA_G1],
    y=df_condicao1_scaled[COLUNA_SLA],
    z=df_condicao1_scaled[COLUNA_DENSIDADE],
    mode='markers',
    marker=dict(size=4, color='blue', opacity=0.6),
    name=CONDICAO_1
))

# Adiciona pontos da Condição 2 (padronizados)
fig.add_trace(go.Scatter3d(
    x=df_condicao2_scaled[COLUNA_G1],
    y=df_condicao2_scaled[COLUNA_SLA],
    z=df_condicao2_scaled[COLUNA_DENSIDADE],
    mode='markers',
    marker=dict(size=4, color='red', opacity=0.6),
    name=CONDICAO_2
))

# Adiciona o casco convexo da Condição 1 (calculado com dados padronizados)
try:
    pontos_c1 = df_condicao1_scaled[[COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE]].values
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

# Adiciona o casco convexo da Condição 2 (calculado com dados padronizados)
try:
    pontos_c2 = df_condicao2_scaled[[COLUNA_G1, COLUNA_SLA, COLUNA_DENSIDADE]].values
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
        xaxis_title=f'{COLUNA_G1} (Padronizado)',
        yaxis_title=f'{COLUNA_SLA} (Padronizado)',
        zaxis_title=f'{COLUNA_DENSIDADE} (Padronizado)'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

# --- Salvando o Gráfico Interativo ---
try:
    fig.write_html(NOME_ARQUIVO_HTML)
    print(f"\nGráfico 3D padronizado salvo com sucesso como '{NOME_ARQUIVO_HTML}'.")
    print("Abra este arquivo em um navegador web para explorar o gráfico interativamente.")
except Exception as e:
    print(f"Ocorreu um erro ao salvar o gráfico: {e}")
    sys.exit(1)


