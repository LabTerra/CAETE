"""Extrai das saídas do modelo e do forçamento as tabelas usadas nas figuras.

Uso (a partir de src/, com o caete_module compilado):
    python ../docs/eixo_carbono_agua_gridcell186-239/scripts/extrai_tabelas.py

Precisa das rodadas longas da célula 186-239 em ../outputs (não versionadas):
    ALLOM_NUTRI_V2   antes (commit 6705a0b)
    TEST_WATER2      mudanças 1 a 3
    TEST_WATER3      mudanças 1 a 4
    TEST_WATER4      mudanças 1 a 5 (código atual)
A V2 é lida de docs/allom_nutri_gridcell186-239/v2_atual/anual.csv; as demais são
agregadas por docs/allom_nutri_gridcell186-239/scripts/agrega.py.

Gera, em ../tabelas:
    solo_celulas.csv                 parâmetros de solo das células 185-240 (K34) e 186-239
    ciclo_anual_mensal.csv           ciclo anual (meses do calendário), 1979-2016
    radiacao_mensal.csv              energia do Penman antes e agora, 1979-2016
    rodadas_longas_historico.csv     médias do período histórico das rodadas longas
    teste_seca_isolado_diario.csv    teste isolado do balanço de água (seca), série diária
    teste_isolado_balanco_agua.csv   resumo dos testes isolados do balanço de água

As tabelas k34_agua_solo_diaria.csv (rodadas intermediárias do teste K34) e as da
bancada (src/tests/carbono_agua/referencia_*.csv) não são geradas aqui; ver o README.
"""
import datetime as dt
import glob
import os
import subprocess
import sys
import tempfile
import bz2
import pickle

import numpy as np
import pandas as pd
from joblib import load

SRC = os.getcwd()
REPO = os.path.dirname(SRC)
AQUI = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(AQUI, "..", "tabelas")
AGREGA = os.path.join(REPO, "docs", "allom_nutri_gridcell186-239", "scripts", "agrega.py")
sys.path.insert(0, SRC)
from hydro_caete import soil_water, net_radiation  # noqa: E402

CELULA = "gridcell186-239"
N_SPINUP = 35  # arquivos spinNN.pkz do segundo spinup; os demais são o histórico
D0_FORC = dt.date(1901, 1, 1)  # início do forçamento (input_data_*.pbz2)
D0_MOD = dt.date(1860, 1, 1)   # referência dos índices de tempo do modelo (sind)

# ---------------- solo
solo = {}
for nome, (x, y) in (("185-240", (185, 240)), ("186-239", (186, 239))):
    v = {n: float(np.load(os.path.join(REPO, f"input/soil/{n}.npy"))[x, y])
         for n in ("ws", "sws", "fc", "sfc", "wp", "swp")}
    v["saturacao_mm"] = 300 * v["ws"] + 700 * v["sws"]          # camadas de 30 e 70 cm
    v["capacidade_campo_mm"] = 300 * v["fc"] + 700 * v["sfc"]
    v["ponto_murcha_mm"] = 300 * v["wp"] + 700 * v["swp"]
    solo[nome] = v
pd.DataFrame(solo).T.rename_axis("celula").to_csv(os.path.join(TAB, "solo_celulas.csv"), float_format="%.6g")

# ---------------- forçamento da célula 186-239
with bz2.BZ2File(os.path.join(REPO, "input/central/input_data_186-239.pbz2")) as fh:
    inp = pickle.load(fh)
pr = np.asarray(inp["pr"]) * 86400.0             # mm dia-1
rsds = np.asarray(inp["rsds"])                    # W m-2
tas = np.asarray(inp["tas"]) - 273.15             # oC
hurs = np.asarray(inp["hurs"]) / 100.0            # 0-1
datas = [D0_FORC + dt.timedelta(int(i)) for i in range(len(rsds))]
mes = np.array([d.month for d in datas])
ano = np.array([d.year for d in datas])
doy = np.array([d.timetuple().tm_yday for d in datas])
off = (D0_FORC - D0_MOD).days


# ---------------- ciclo anual (meses do calendário), período histórico
def ciclo(rodada):
    arqs = sorted(glob.glob(os.path.join(REPO, f"outputs/{rodada}/{CELULA}/spin*.pkz")))[N_SPINUP:]
    W, E, G, P, M = [], [], [], [], []
    for f in arqs:
        d = load(f)
        s0, n = int(d["sind"]), len(d["evapm"])
        W.append(np.asarray(d["wsoil"]) + np.asarray(d["swsoil"]))
        E.append(np.asarray(d["evapm"]))
        G.append(np.asarray(d["photo"]))
        P.append(pr[s0 - off:s0 - off + n])
        M.append([(D0_MOD + dt.timedelta(s0 + i)).month for i in range(n)])
    W, E, G, P, M = map(np.concatenate, (W, E, G, P, M))
    mm = lambda X: np.array([X[M == k].mean() for k in range(1, 13)])
    return dict(P=mm(P), W=mm(W), E=mm(E), G=mm(G))


antes, atual = ciclo("ALLOM_NUTRI_V2"), ciclo("TEST_WATER4")
pd.DataFrame({"mes": range(1, 13), "chuva": antes["P"],
              "agua_V2": antes["W"], "agua_atual": atual["W"],
              "ET_V2": antes["E"], "ET_atual": atual["E"],
              "GPP_V2": antes["G"], "GPP_atual": atual["G"]}).round(3).to_csv(
    os.path.join(TAB, "ciclo_anual_mensal.csv"), index=False)

# ---------------- radiação: energia do Penman antes e agora (1979-2016)
lat = 89.75 - 0.5 * 186  # latitude do centro da célula (mesma conta de caete.py)
rn = net_radiation(rsds, tas, hurs, lat, doy)
energia_antes = 2.895 * tas + 52.326             # available_energy(temp), funcs.f90
shuttleworth = 0.858 * rsds - 35.0               # Shuttleworth (1984), Manaus
h = ano >= 1979
linhas = [dict(mes=str(k), rsds=rsds[h & (mes == k)].mean(), energia_antes=energia_antes[h & (mes == k)].mean(),
               Rn_FAO56=rn[h & (mes == k)].mean(), Rn_Shuttleworth=shuttleworth[h & (mes == k)].mean())
          for k in range(1, 13)]
linhas.append(dict(mes="ano", rsds=rsds[h].mean(), energia_antes=energia_antes[h].mean(),
                   Rn_FAO56=rn[h].mean(), Rn_Shuttleworth=shuttleworth[h].mean()))
pd.DataFrame(linhas).round(2).to_csv(os.path.join(TAB, "radiacao_mensal.csv"), index=False)

# ---------------- rodadas longas: médias do período histórico
colunas = ["photo", "npp", "ar", "lai", "cleaf", "cfroot", "csap", "cheart", "bio", "evapm", "emaxm",
           "runom", "rcm", "pls_vivos", "area_lim_P", "nsto", "nupt_mineral"]
anuais = {"V2": os.path.join(REPO, "docs/allom_nutri_gridcell186-239/v2_atual/anual.csv")}
tmp = tempfile.mkdtemp()
for nome, rodada in (("TW2", "TEST_WATER2"), ("TW3", "TEST_WATER3"), ("TW4", "TEST_WATER4")):
    saida = os.path.join(tmp, nome)
    subprocess.run([sys.executable, AGREGA, os.path.join(REPO, "outputs", rodada), saida, CELULA],
                   check=True, stdout=subprocess.DEVNULL)
    anuais[nome] = os.path.join(saida, "anual.csv")
medias = {}
for nome, f in anuais.items():
    a = pd.read_csv(f, index_col=0)
    a["bio"] = a.cleaf + a.cfroot + a.csap + a.cheart
    medias[nome] = a[a.fase == "historico"][colunas].mean()
pd.DataFrame(medias).to_csv(os.path.join(TAB, "rodadas_longas_historico.csv"), float_format="%.4g")


# ---------------- testes isolados do balanço de água (solo do K34)
def novo_solo():
    v = solo["185-240"]
    s = soil_water(v["ws"], v["sws"], v["fc"], v["sfc"], v["wp"], v["swp"])
    s.w1, s.w2 = s.w1_max * 0.9, s.w2_max * 0.9
    return s


def seca(rng):
    """Chuva só em 500 de cada 730 dias (média 1,8 mm dia-1) e demanda de 3 a 6 mm dia-1."""
    while True:
        for i in range(730):
            p = rng.gamma(0.3, 6) if i < 500 else 0.0
            yield p, rng.uniform(3, 6)


# (a) solo úmido: o método novo deve dar o mesmo resultado do original
r = np.random.default_rng(1)
a1, b1 = novo_solo(), novo_solo()
dmax = 0.0
for i in range(3000):
    p, e = r.gamma(0.6, 12), r.uniform(0, 4)
    q1 = a1._update_pool(p, e)
    q2, et = b1._update_pool_allom(p, e)
    dmax = max(dmax, abs(q1 - q2), abs(a1.w1 - b1.w1), abs(a1.w2 - b1.w2), abs(et - e))

# (b) seca: método novo (conservação de massa) e antigo (camadas negativas corrigidas para zero)
s_novo, s_ant = novo_solo(), novo_solo()
wpm = s_novo.p1_vol * s_novo.wp1 + s_novo.p2_vol * s_novo.wp2
g_novo, g_ant = seca(np.random.default_rng(1)), seca(np.random.default_rng(1))
erro, neg, criada = 0.0, 0, 0.0
serie = []
for i in range(3000):
    p, e = next(g_novo)
    w0 = s_novo.w1 + s_novo.w2
    ro, et = s_novo._update_pool_allom(p, e)
    erro = max(erro, abs((s_novo.w1 + s_novo.w2) - (w0 + p - et - ro)))
    neg += (s_novo.w1 < 0) or (s_novo.w2 < 0)
    p, e = next(g_ant)
    s_ant._update_pool(p, e)
    criada += max(0.0, -s_ant.w1) + max(0.0, -s_ant.w2)
    s_ant.w1, s_ant.w2 = max(0.0, s_ant.w1), max(0.0, s_ant.w2)
    serie.append((i, s_ant.w1 + s_ant.w2, s_novo.w1 + s_novo.w2, criada))
serie = pd.DataFrame(serie, columns=["dia", "agua_metodo_antigo", "agua_metodo_novo", "agua_criada_acumulada"])
serie.round(3).to_csv(os.path.join(TAB, "teste_seca_isolado_diario.csv"), index=False)
pd.DataFrame([
    dict(teste="solo úmido: diferença máxima para o original (mm)", valor=dmax),
    dict(teste="seca: erro máximo de conservação de massa (mm)", valor=erro),
    dict(teste="seca: dias com camada negativa", valor=neg),
    dict(teste="seca: água mínima no solo, novo (mm)", valor=serie.agua_metodo_novo.min()),
    dict(teste="ponto de murcha das duas camadas (mm)", valor=wpm),
    dict(teste="seca: água mínima no solo, original (mm)", valor=serie.agua_metodo_antigo.min()),
    dict(teste="seca: água criada pela correção para zero, original (mm em 3000 dias)", valor=criada),
]).to_csv(os.path.join(TAB, "teste_isolado_balanco_agua.csv"), index=False)
print("tabelas gravadas em", os.path.abspath(TAB))
