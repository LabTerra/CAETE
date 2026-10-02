"""Figuras da rodada ALLOM_NUTRI (celula 186-239) a partir de anual.csv"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False, 'lines.linewidth': 2})
# d (tabela), x (eixo), H0 (ultimo ano do spinup), XLAB e SUFFIX sao definidos
# no fim do arquivo, para cada conjunto de figuras

def base(ax, title, ylab, zero=True):
    ax.set_title(title); ax.set_ylabel(ylab)
    if H0: ax.axvline(H0 + 0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax._zero = zero
    ax.set_xlim(x[0], x[-1])

def mark_hist(ax):
    ax._hist = True

def _mark_hist(ax):
    ax.text(H0 + 2, ax.get_ylim()[1], 'histórico\n1979–2016', va='top', ha='left', fontsize=8, color=MUTED)

def lines(ax, cols, labels, colors=None, end_labels=True):
    colors = colors or C
    for c, l, k in zip(cols, labels, colors):
        ax.plot(x, d[c], color=k, label=l)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=len(cols), fontsize=9, handlelength=1.4)

def finish(fig, name, sup):
    fig.suptitle(sup, x=0.01, ha='left', fontsize=13, fontweight='bold')
    for ax in fig.axes:
        ax.set_xlabel(XLAB)
        if getattr(ax, '_zero', False) and not getattr(ax, '_fixed', False):
            ax.set_ylim(0, max(l.get_ydata().max() for l in ax.get_lines() if len(l.get_ydata()) > 2) * 1.12)
        if getattr(ax, '_hist', False) and H0: _mark_hist(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(name.replace('.png', SUFFIX + '.png'), dpi=140); plt.close(fig)

SUB = 'CAETÊ alométrico + nutrientes · célula 186-239 (Amazônia central) · 3000 PLSs'

def make():
    # 1 - fluxos de carbono
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    base(ax[0, 0], 'Fotossíntese, respiração e NPP', 'kgC m⁻² ano⁻¹')
    lines(ax[0, 0], ['photo', 'ar', 'npp'], ['Fotossíntese (GPP)', 'Respiração autotrófica', 'NPP']); mark_hist(ax[0, 0])
    base(ax[0, 1], 'NPP', 'kgC m⁻² ano⁻¹'); ax[0, 1].plot(x, d['npp'], color=C[2])
    d['cue'] = d['npp'] / d['photo']
    base(ax[1, 0], 'Eficiência de uso do carbono (NPP / GPP)', 'fração'); ax[1, 0].plot(x, d['cue'], color=C[0])
    base(ax[1, 1], 'Índice de área foliar (LAI)', 'm² m⁻²'); ax[1, 1].plot(x, d['lai'], color=C[0])
    finish(fig, 'fig1_fluxos_carbono.png', 'Fluxos de carbono — ' + SUB)

    # 2 - biomassa
    d['lenho'] = d['csap'] + d['cheart']; d['total'] = d['cleaf'] + d['cfroot'] + d['lenho'] + d['csto']
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    base(ax[0, 0], 'Biomassa total', 'kgC m⁻²'); ax[0, 0].plot(x, d['total'], color=C[0]); mark_hist(ax[0, 0])
    base(ax[0, 1], 'Biomassa por tecido', 'kgC m⁻²')
    lines(ax[0, 1], ['cleaf', 'cfroot', 'csap', 'cheart'], ['Folha', 'Raiz fina', 'Sapwood', 'Heartwood'])
    base(ax[1, 0], 'Folha e raiz fina', 'kgC m⁻²'); lines(ax[1, 0], ['cleaf', 'cfroot'], ['Folha', 'Raiz fina'])
    base(ax[1, 1], 'Estoque de carbono não estrutural (csto)', 'gC m⁻²'); ax[1, 1].plot(x, d['csto'] * 1e3, color=C[0])
    finish(fig, 'fig2_biomassa.png', 'Biomassa — ' + SUB)

    # 3 - N e P nos tecidos
    for t, c in (('folha', 'cleaf'), ('raiz', 'cfroot'), ('lenho', 'csap')):
        d[f'N_{t}'] = d[c] * 1e3 * d[f'n2c_{t}']; d[f'P_{t}'] = d[c] * 1e3 * d[f'p2c_{t}']
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    lab = ['Folha', 'Raiz fina', 'Sapwood']
    for j, (nut, f) in enumerate((('N', 1e3), ('P', 1e3))):
        base(ax[0, j], f'Razão {nut}:C média da comunidade (ponderada pela área)', f'mg {nut} por gC')
        for t, l, k in zip(('folha', 'raiz', 'lenho'), lab, C):
            ax[0, j].plot(x, d[f'{nut.lower()}2c_{t}'] * f, color=k, label=l)
        ax[0, j].legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9, handlelength=1.4)
        base(ax[1, j], f'{nut} contido em cada tecido (carbono × razão média)', f'g {nut} m⁻²')
        for t, l, k in zip(('folha', 'raiz', 'lenho'), lab, C):
            ax[1, j].plot(x, d[f'{nut}_{t}'], color=k, label=l)
        ax[1, j].legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9, handlelength=1.4)
    mark_hist(ax[0, 0])
    finish(fig, 'fig3_N_P_tecidos.png', 'N e P nos tecidos — ' + SUB)

    # 4 - comunidade e limitacao
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    base(ax[0, 0], 'PLSs vivos (de 3000)', 'número de PLSs'); ax[0, 0].plot(x, d['pls_vivos'], color=C[0]); mark_hist(ax[0, 0])
    base(ax[0, 1], 'Área ocupada por lenhosas', 'fração da área'); ax[0, 1].plot(x, d['area_lenhosas'], color=C[0]); ax[0, 1].set_ylim(0, 1.02); ax[0, 1]._fixed = True
    base(ax[1, 0], 'Área por tipo de limitação (média anual)', 'fração da área')
    ax[1, 0].stackplot(x, d['area_sem_lim'], d['area_lim_P'], d['area_lim_N'] + d['area_colim'].clip(lower=0),
                       colors=[C[0], C[1], C[2]], labels=['Sem limitação', 'Limitada por P', 'Limitada por N ou colimitada'],
                       edgecolor=SURF, linewidth=0.3)
    ax[1, 0]._fixed = True; ax[1, 0].set_ylim(0, 1); ax[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9, handlelength=1.4)
    base(ax[1, 1], 'Área com cada órgão limitado (média anual)', 'fração da área')
    lines(ax[1, 1], ['area_lim_folha', 'area_lim_raiz', 'area_lim_lenho'], ['Folha', 'Raiz fina', 'Sapwood']); ax[1, 1].set_ylim(0, 1); ax[1, 1]._fixed = True
    finish(fig, 'fig4_comunidade_limitacao.png', 'Sobrevivência e limitação — ' + SUB)

    # 5 - ciclo de nutrientes
    d['nupt'] = d['nupt_mineral'] + d['nupt_organico']
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    base(ax[0, 0], 'N: absorção e retorno pela serapilheira', 'g N m⁻² ano⁻¹')
    lines(ax[0, 0], ['nupt', 'litter_n'], ['Absorção', 'Serapilheira']); mark_hist(ax[0, 0])
    base(ax[0, 1], 'N mineral no solo', 'g N m⁻²'); ax[0, 1].plot(x, d['nmin'], color=C[0])
    base(ax[0, 2], 'Reserva de N das plantas', 'g N m⁻²'); ax[0, 2].plot(x, d['nsto'], color=C[0])
    base(ax[1, 0], 'P: absorção por origem e retorno pela serapilheira', 'g P m⁻² ano⁻¹')
    lines(ax[1, 0], ['pupt_labil', 'pupt_organico', 'pupt_sorvido', 'litter_p'], ['Absorção: lábil', 'Absorção: orgânico', 'Absorção: sorvido', 'Serapilheira'])
    ax[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9, handlelength=1.4)
    base(ax[1, 1], 'P lábil no solo', 'g P m⁻²'); ax[1, 1].plot(x, d['pmin'], color=C[0])
    base(ax[1, 2], 'Reserva de P das plantas', 'g P m⁻²'); ax[1, 2].plot(x, d['psto'], color=C[0])
    finish(fig, 'fig5_ciclo_nutrientes.png', 'Ciclo de N e P — ' + SUB)

    # 6 - respiracao e solo
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    d['r_sto'] = (d['ar'] - d['rm'] - d['rg']).clip(lower=0)
    base(ax[0, 0], 'Respiração autotrófica por componente', 'kgC m⁻² ano⁻¹')
    lines(ax[0, 0], ['rm', 'rg', 'r_sto'], ['Manutenção', 'Crescimento', 'Estoque (inclui excedente do teto)']); mark_hist(ax[0, 0])
    base(ax[0, 1], 'Respiração heterotrófica (solo)', 'gC m⁻² dia⁻¹'); ax[0, 1].plot(x, d['hresp'], color=C[0])
    base(ax[1, 0], 'Serapilheira de carbono', 'gC m⁻² ano⁻¹')
    lines(ax[1, 0], ['litter_l', 'litter_fr', 'cwd'], ['Folha', 'Raiz fina', 'Lenho (cwd)'])
    base(ax[1, 1], 'Carbono no solo', 'kgC m⁻²'); ax[1, 1].plot(x, d['csoil'] / 1e3, color=C[0])
    finish(fig, 'fig6_respiracao_solo.png', 'Respiração e solo — ' + SUB)


ALL = pd.read_csv('anual.csv', index_col=0)

# serie completa: spinup com nutrientes + historico
d = ALL.copy(); x = d.index.values; H0 = int((d['fase'] == 'spinup').sum())
XLAB, SUFFIX = 'ano de simulação com nutrientes', ''
make()

# so a rodada historica, com o ano civil no eixo
d = ALL[ALL['fase'] == 'historico'].copy(); x = 1979 + np.arange(len(d)); d.index = x; H0 = None
XLAB, SUFFIX = 'ano', '_historico'
SUB = SUB + ' · rodada histórica'
make()
print('ok')
