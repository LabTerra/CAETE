"""Ajusta o g1 de Medlyn et al. (2011) por espécie no banco de Lin et al. (2015).

Base da mudança 5 (intervalo do g1 sorteado em plsgen.py). Gera, em ../tabelas:
    g1_lin2015_por_especie.csv   g1 de cada espécie (pelo menos 3 medidas de campo)
    g1_lin2015_por_grupo.csv     mediana, média e percentis por grupo

Uso (de qualquer pasta):
    python ajusta_g1_lin2015.py [WUEdatabase_merged_Lin_et_al_2015_NCC.csv]

Sem o argumento, o arquivo é baixado do figshare (CC BY 4.0):
    Lin, Y.-S., Medlyn, B.E. & Duursma, R.A. (2015). Optimal stomatal behaviour
    around the world. figshare. https://doi.org/10.6084/m9.figshare.1304289.v2

Ajuste (g0 = 0, como em Lin et al. 2015), por mínimos quadrados:
    gs = 1,6 (1 + g1/√D) A/Ca  ->  gs - x = g1·y,  x = 1,6 A/Ca,  y = x/√D
    g1 = Σ y (gs - x) / Σ y²
Só medidas de campo (Growthcond = field), com gs, A, D e Ca positivos, D > 0,05 kPa
e PAR registrado. Os limites do sorteio são os percentis 5 e 95 das espécies:
C3 1,6 a 8,6 e C4 0,7 a 4,1.
"""
import os
import sys
import tempfile
import urllib.request

import numpy as np
import pandas as pd

URL = "https://ndownloader.figshare.com/files/1886204"
TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tabelas")

if len(sys.argv) > 1:
    banco = sys.argv[1]
else:
    # baixado numa pasta temporária (o banco original não é guardado no repositório)
    banco = os.path.join(tempfile.gettempdir(), "WUEdatabase_merged_Lin_et_al_2015_NCC.csv")
    if not os.path.exists(banco):
        print("baixando", URL)
        urllib.request.urlretrieve(URL, banco)

d = pd.read_csv(banco, encoding="latin1")
d = d[d.Growthcond.str.strip().str.lower() == "field"]
for c in ("Cond", "Photo", "VPD", "CO2S"):
    d = d[d[c] > 0]
d = d[d.PARin.notna() & (d.VPD > 0.05)]
x = 1.6 * d.Photo / d.CO2S
d = d.assign(x=x, y=x / np.sqrt(d.VPD))

linhas = []
for sp, g in d.groupby("Species"):
    if len(g) < 3:
        continue
    linhas.append(dict(sp=sp, n=len(g), g1=(g.y * (g.Cond - g.x)).sum() / (g.y ** 2).sum(),
                       path=g.Pathway.iloc[0], type=g.Type.iloc[0], form=g.Plantform.iloc[0],
                       leaf=g.Leafspan.iloc[0], treg=g.Tregion.iloc[0], wreg=g.Wregion.iloc[0],
                       loc=g.Location.iloc[0]))
sp = pd.DataFrame(linhas)
sp.to_csv(os.path.join(TAB, "g1_lin2015_por_especie.csv"), index=False)

grupos = (("Árvores tropicais", (sp.form == "tree") & (sp.treg == "tropical")),
          ("Paracou (Guiana Francesa)", sp["loc"] == "Paracou_French Guiana"),
          ("Tambopata (Peru)", sp["loc"] == "Tambopata_Peru"),
          ("Todas as C3", sp.path == "C3"),
          ("Angiospermas arbóreas C3", (sp.path == "C3") & (sp.type == "angiosperm") & (sp.form == "tree")),
          ("Gramíneas C3", (sp.form == "grass") & (sp.path == "C3")),
          ("Árvores de savana", sp.form == "savanna"),
          ("Gimnospermas", sp.type == "gymnosperm"),
          ("Todas as C4", sp.path == "C4"))
res = []
for nome, m in grupos:
    v = sp[m].g1
    res.append(dict(grupo=nome, especies=len(v), mediana=v.median(), media=v.mean(),
                    p5=np.percentile(v, 5), p95=np.percentile(v, 95), min=v.min(), max=v.max()))
res = pd.DataFrame(res).round(2)
res.to_csv(os.path.join(TAB, "g1_lin2015_por_grupo.csv"), index=False)
print(res.to_string(index=False))
