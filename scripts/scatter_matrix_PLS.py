# import matplotlib.pyplot as plt
# import pandas as pd


# # Edite o caminho para o arquivo .csv con as PLSs, na lista de colunas, coloque as colunas que deseja plotar
# # Pra saber quais as colunas disponíveis, abra o arquivo .csv no Excel
# # table_woody = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[1600:, ["aleaf","awood","aroot","tleaf", 'twood', "troot"]]
# # table_grass = pd.read_csv("./PLS_MAIN/pls_attrs-20000.csv").loc[:16, ["aleaf","aroot","tleaf","troot"]]

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from math import ceil
import numpy as np
from matplotlib.patches import Patch

# Must match plsgen.toml
GRASS_FRAC = 0.07  # Fraction of grass PLSs in the dataset

# Table with all PLSs and traits
pls_table = pd.read_csv("../src/PLS_MAIN/pls_attrs-5000.csv")


def assert_data_size(dsize):
    """ Assertion of datasets sizes """

    g2w_ratio = GRASS_FRAC
    diffg = ceil(dsize * g2w_ratio)
    diffw = int(dsize - diffg)
    assert diffg + diffw == dsize
    return diffg, diffw

def create_colored_scatter_matrix(data, columns, plant_type_col, title, filename):
    """Create a scatter matrix with colored plant types"""

    # Color mapping
    color_map = {'Grass': "#C709B7", 'Woody': "#05F826"}
    colors = [color_map[pt] for pt in data[plant_type_col]]

    # Create scatter matrix
    axes = pd.plotting.scatter_matrix(data[columns],
                              c=colors,
                              figsize=(8, 6),
                              alpha=0.3)

    # Format tick labels: scientific notation and smaller font
    for ax_row in axes:
        for ax in ax_row:
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
            ax.tick_params(axis='both', labelsize=6)
            ax.xaxis.offsetText.set_fontsize(5)
            ax.yaxis.offsetText.set_fontsize(5)
            # Move x-axis offset text to top right of axes
            ax.xaxis.offsetText.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # Draw to compute offset text values, then annotate
    plt.gcf().canvas.draw()
    for ax_row in axes:
        for ax in ax_row:
            x_offset = ax.xaxis.get_offset_text().get_text()
            y_offset = ax.yaxis.get_offset_text().get_text()
            offset_parts = []
            if x_offset:
                offset_parts.append(f'x: {x_offset}')
            if y_offset:
                offset_parts.append(f'y: {y_offset}')
            if offset_parts:
                ax.annotate('  '.join(offset_parts),
                           xy=(0.95, 0.92), xycoords='axes fraction',
                           ha='right', va='top', fontsize=5)

    # Add legend
    legend_elements = [Patch(facecolor=color_map[pt], label=pt)
                      for pt in color_map.keys() if pt in data[plant_type_col].values]
    plt.figlegend(handles=legend_elements,
                  loc='lower left',
                  bbox_to_anchor=(-0.01, -0.01),
                  ncol=1,
                  fontsize='small',
                  handlelength=1.0,
                  handletextpad=0.4,
                  frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=400)

# Add plant type classification (adjust ranges as needed)
def classify_plant_type(df):
    plant_types = []
    gf, _ = assert_data_size(len(df))
    for i in range(len(df)):
        if i < gf:
            plant_types.append('Grass')
        else:
            plant_types.append('Woody')
    return plant_types


#---------------
# Main execution
# ---------------

pls_table['plant_type'] = classify_plant_type(pls_table)

# Create some plots
create_colored_scatter_matrix(
    pls_table,
    ["aleaf","awood","aroot","tleaf", 'twood', "troot"],
    'plant_type',
    'Carbon Allocation and Turnover Traits',
    'scatter_matrix_Cturn_colored.png'
)

create_colored_scatter_matrix(
    pls_table,
    ["leaf_n2c","awood_n2c","froot_n2c","leaf_p2c","awood_p2c","froot_p2c"],
    'plant_type',
    'Nutrient Concentration Ratios',
    'scatter_matrix_nconc_colored.png'
)

create_colored_scatter_matrix(
    pls_table,
    ["leaf_n2c","leaf_p2c", "aleaf", "tleaf"],
    'plant_type',
    'Leaf CNP and Allocation Traits',
    'scatter_matrix_leafCNP_colored.png'
)

create_colored_scatter_matrix(
    pls_table,
    ["awood_n2c","awood_p2c", "awood", "twood"],
    'plant_type',
    'Nutrient Concentration Ratios',
    'scatter_matrix_wood_CNP.png'
)

create_colored_scatter_matrix(
    pls_table,
    ["froot_n2c","froot_p2c", "aroot", "troot"],
    'plant_type',
    'Nutrient Concentration Ratios',
    'scatter_matrix_root_CNP.png'
)
