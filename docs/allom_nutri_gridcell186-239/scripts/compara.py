"""Compara duas versoes (v1_inicial e v2_atual) da celula 186-239.
Uso: python compara.py   (de dentro de scripts/; grava os PNG na pasta acima)
Entradas: ../v1_inicial/anual.csv e ../v2_atual/anual.csv (gerados por agrega.py)."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
C1, C2 = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False, 'lines.linewidth': 2})
V1 = pd.read_csv('../v1_inicial/anual.csv', index_col=0)
V2 = pd.read_csv('../v2_atual/anual.csv', index_col=0)
for d in (V1, V2):
    d['lenho'] = d.csap + d.cheart
    d['total'] = d.cleaf + d.cfroot + d.lenho + d.csto
    d['cue'] = d.npp / d.photo
    d['pupt'] = d.pupt_labil + d.pupt_sorvido + d.pupt_organico
    d['sla_ef'] = d.lai / d.cleaf                     # m2 por kgC (LAI / carbono da folha)
    d['ll_ef'] = d.cleaf * 1e3 / d.litter_l           # anos (carbono da folha / serapilheira foliar)
H0 = int((V1.fase == 'spinup').sum())
L1 = 'Versão 1 (publicada)'
L2 = 'Versão 2 (atual)'

# ------------------------------------------------------------ 1. series temporais
panels = [('photo', 'Fotossíntese (GPP)', 'kgC m⁻² ano⁻¹'), ('ar', 'Respiração autotrófica', 'kgC m⁻² ano⁻¹'),
          ('npp', 'NPP', 'kgC m⁻² ano⁻¹'), ('cue', 'NPP / GPP', 'fração'),
          ('lai', 'LAI', 'm² m⁻²'), ('cleaf', 'Folha', 'kgC m⁻²'), ('cfroot', 'Raiz fina', 'kgC m⁻²'), ('csap', 'Sapwood', 'kgC m⁻²'),
          ('cheart', 'Heartwood', 'kgC m⁻²'), ('total', 'Biomassa total', 'kgC m⁻²'), ('pls_vivos', 'PLSs vivos', 'número'),
          ('area_lim_P', 'Área limitada por P', 'fração da área'),
          ('nmin', 'N mineral do solo', 'g N m⁻²'), ('nsto', 'Reserva de N das plantas', 'g N m⁻²'),
          ('evapm', 'Evapotranspiração', 'mm dia⁻¹'), ('csoil', 'Carbono no solo', 'gC m⁻²')]
fig, axes = plt.subplots(4, 4, figsize=(18, 13.5))
for ax, (k, t, u) in zip(axes.flat, panels):
    ax.plot(V1.index, V1[k], color=C1, label=L1)
    ax.plot(V2.index, V2[k], color=C2, label=L2)
    ax.axvline(H0 + 0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_title(t); ax.set_ylabel(u); ax.set_xlim(1, V1.index[-1]); ax.set_xlabel('ano de simulação com nutrientes')
    ax.set_ylim(0, max(V1[k].max(), V2[k].max()) * 1.12)
axes.flat[0].text(H0 + 2, axes.flat[0].get_ylim()[1], 'histórico\n1979–2016', va='top', fontsize=8, color=MUTED)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=2, fontsize=11)
fig.suptitle('Versão 1 × Versão 2 — série completa (385 anos de spinup + histórico 1979–2016) · célula 186-239 · 3000 PLSs',
             x=0.01, ha='left', fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig('../comparacao_v1_v2_series.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------ 2. barras no fim do spinup (ano 385)
y = 385
a, b = V1.loc[y], V2.loc[y]
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
xs = np.arange(2); w = 0.55
def bars(axx, items, title, unit, stack=True):
    bottom = np.zeros(2)
    for lab, k, col in items:
        vals = np.array([a[k], b[k]])
        axx.bar(xs, vals, w, bottom=bottom, color=col, edgecolor=SURF, linewidth=1.5, label=lab)
        for i, v in enumerate(vals):
            if v > 0.07 * max(sum(a[kk] for _, kk, _ in items), sum(b[kk] for _, kk, _ in items)):
                axx.text(xs[i], bottom[i] + v / 2, (f'{v:.0f}' if v >= 100 else f'{v:.2f}'), ha='center', va='center', fontsize=9, color='white')
        bottom += vals
    axx.set_xticks(xs); axx.set_xticklabels([L1, L2]); axx.set_title(title); axx.set_ylabel(unit)
    axx.set_ylim(0, bottom.max() * 1.32); axx.legend(fontsize=8, loc='upper center', ncol=2); axx.grid(axis='x', visible=False)
P = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a4fd1']
bars(ax[0, 0], [('NPP', 'npp', P[2]), ('Respiração autotrófica', 'ar', P[1])], 'Fotossíntese (GPP) = NPP + respiração', 'kgC m⁻² ano⁻¹')
bars(ax[0, 1], [('Folha', 'cleaf', P[2]), ('Raiz fina', 'cfroot', P[3]), ('Sapwood', 'csap', P[0]), ('Heartwood', 'cheart', P[4])], 'Biomassa por tecido', 'kgC m⁻²')
bars(ax[0, 2], [('Serapilheira foliar', 'litter_l', P[2]), ('Serapilheira de raiz', 'litter_fr', P[3]), ('Detrito lenhoso (cwd)', 'cwd', P[4])], 'Serapilheira de carbono', 'gC m⁻² ano⁻¹')
bars(ax[1, 0], [('Absorção de N', 'nupt_mineral', P[0])], 'Absorção de N', 'g N m⁻² ano⁻¹')
bars(ax[1, 1], [('P lábil', 'pupt_labil', P[0]), ('P orgânico', 'pupt_organico', P[1])], 'Absorção de P', 'g P m⁻² ano⁻¹')
bars(ax[1, 2], [('Evapotranspiração', 'evapm', P[0])], 'Evapotranspiração (potencial: %.2f mm dia⁻¹)' % a['emaxm'], 'mm dia⁻¹')
fig.suptitle('Versão 1 × Versão 2 — fim do spinup (ano 385) · célula 186-239', x=0.01, ha='left', fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig('../comparacao_v1_v2_barras.png', dpi=130); plt.close(fig)

# ------------------------------------------------------------ 3. tabela resumo
rows = [('Fotossíntese, GPP (kgC m⁻² ano⁻¹)', 'photo'), ('Respiração autotrófica (kgC m⁻² ano⁻¹)', 'ar'), ('NPP (kgC m⁻² ano⁻¹)', 'npp'),
        ('NPP / GPP', 'cue'), ('LAI (m² m⁻²)', 'lai'), ('SLA efetivo, LAI / folha (m² kgC⁻¹)', 'sla_ef'),
        ('Longevidade foliar efetiva (anos)', 'll_ef'), ('Folha (kgC m⁻²)', 'cleaf'), ('Raiz fina (kgC m⁻²)', 'cfroot'),
        ('Sapwood (kgC m⁻²)', 'csap'), ('Heartwood (kgC m⁻²)', 'cheart'), ('Biomassa total (kgC m⁻²)', 'total'),
        ('PLSs vivos (de 3000)', 'pls_vivos'), ('Serapilheira foliar (gC m⁻² ano⁻¹)', 'litter_l'),
        ('Carbono no solo (gC m⁻²)', 'csoil'), ('Evapotranspiração (mm dia⁻¹)', 'evapm'), ('Escoamento (mm dia⁻¹)', 'runom'),
        ('N mineral do solo (g m⁻²)', 'nmin'), ('Reserva de N das plantas (g m⁻²)', 'nsto'), ('Absorção de N (g m⁻² ano⁻¹)', 'nupt_mineral'),
        ('Absorção de P (g m⁻² ano⁻¹)', 'pupt'), ('Área limitada por P', 'area_lim_P')]
tab = pd.DataFrame({'variavel': [r[0] for r in rows],
                    'v1_385': [V1.loc[385, r[1]] for r in rows], 'v2_385': [V2.loc[385, r[1]] for r in rows],
                    'v1_2016': [V1.loc[423, r[1]] for r in rows], 'v2_2016': [V2.loc[423, r[1]] for r in rows]})
tab['dif_385_pct'] = 100 * (tab.v2_385 / tab.v1_385 - 1)
tab.to_csv('../comparacao_v1_v2_tabela.csv', index=False)
pd.set_option('display.width', 200); pd.set_option('display.float_format', '{:.3g}'.format)
print(tab.to_string(index=False))
