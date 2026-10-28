"""Agrega as saidas diarias (spinNN.pkz) de uma celula em medias/somas anuais.

Uso (de qualquer pasta):
    python agrega.py <pasta da rodada> <pasta de saida> [celula]
    ex.: python agrega.py ../../../outputs/ALLOM_NUTRI_V2 ../v2_atual gridcell186-239

Gera <pasta de saida>/anual.csv, com uma linha por ano de simulacao. Os 35 primeiros
arquivos spinNN.pkz sao o segundo spinup (11 anos cada); os demais, a rodada historica.
Variaveis de estado: media do ano; fluxos de serapilheira e absorcao: soma anual.
"""
import glob, os, sys
import numpy as np, pandas as pd
from joblib import load

run, out = sys.argv[1], sys.argv[2]
cell = sys.argv[3] if len(sys.argv) > 3 else 'gridcell186-239'
tr = pd.read_csv(f'{run}/pls_attrs-3000.csv')
woody = tr['awood'].values > 0
rows = []
for nf, f in enumerate(sorted(glob.glob(f'{run}/{cell}/spin*.pkz'))):
    d = load(f)
    n = len(d['npp'])
    nyr = max(1, round(n / 365.25))
    for idx in np.array_split(np.arange(n), nyr):
        r = {'fase': 'spinup' if nf < 35 else 'historico'}
        for k in ('photo', 'npp', 'ar', 'rm', 'rg', 'lai', 'cleaf', 'cfroot', 'csap', 'cheart', 'csto',
                  'nsto', 'psto', 'nmin', 'pmin', 'hresp', 'wue', 'evapm', 'emaxm', 'runom', 'wsoil', 'rcm'):
            r[k] = d[k][idx].mean()
        for k in ('litter_l', 'litter_fr', 'cwd', 'c_cost'):
            r[k] = d[k][idx].sum() * 365.0 / len(idx)
        for i, k in enumerate(('nupt_mineral', 'nupt_organico')):
            r[k] = d['nupt'][i, idx].sum()
        for i, k in enumerate(('pupt_labil', 'pupt_sorvido', 'pupt_organico')):
            r[k] = d['pupt'][i, idx].sum()
        r['litter_n'] = d['lnc'][:3, idx].sum(); r['litter_p'] = d['lnc'][3:, idx].sum()
        r['csoil'] = d['csoil'][:, idx].sum(axis=0).mean()
        last = idx[-1]
        r['pls_vivos'] = d['ls'][last]
        a = d['area'][:, last]; a = np.where(a > 0, a, 0.0); a = a / a.sum()
        r['area_lenhosas'] = a[woody].sum()
        for t, col in (('folha', 'leaf'), ('raiz', 'froot')):
            r[f'n2c_{t}'] = (a * tr[f'{col}_n2c'].values).sum()
            r[f'p2c_{t}'] = (a * tr[f'{col}_p2c'].values).sum()
        aw = a[woody] / a[woody].sum()
        r['n2c_lenho'] = (aw * tr['awood_n2c'].values[woody]).sum()
        r['p2c_lenho'] = (aw * tr['awood_p2c'].values[woody]).sum()
        # limitacao: media anual da fracao de area, com todos os dias do ano
        A = d['area'][:, idx]; A = np.where(A > 0, A, 0.0); A = A / A.sum(axis=0)
        lim = d['lim_status'][:, :, idx]
        hasn = np.isin(lim, (1, 4)).any(axis=0); hasp = np.isin(lim, (2, 5)).any(axis=0); has6 = (lim == 6).any(axis=0)
        no = (lim == 0).all(axis=0)
        r['area_sem_lim'] = (A * no).sum(axis=0).mean()
        r['area_lim_N'] = (A * (hasn & ~hasp & ~has6)).sum(axis=0).mean()
        r['area_lim_P'] = (A * (hasp & ~hasn & ~has6)).sum(axis=0).mean()
        r['area_colim'] = max(0.0, 1.0 - r['area_sem_lim'] - r['area_lim_N'] - r['area_lim_P'])
        for i, t in enumerate(('folha', 'lenho', 'raiz')):
            r[f'area_lim_{t}'] = (A * (lim[i] != 0)).sum(axis=0).mean()
        rows.append(r)
    print(os.path.basename(f), nyr, flush=True)
df = pd.DataFrame(rows); df.index = np.arange(1, len(df) + 1); df.index.name = 'ano_sim'
os.makedirs(out, exist_ok=True)
df.to_csv(f'{out}/anual.csv')
print(df.groupby('fase').size())
