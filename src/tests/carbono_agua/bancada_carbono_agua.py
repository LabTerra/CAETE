"""Bancada de teste do eixo carbono-agua (versao com alometria e nutrientes).

Chama a rotina `productivity.prod_allom` (a mesma usada por budget_allom.f90) e o
Penman de `water`, com entradas controladas, numa grade fatorial:

    agua do solo x CO2 x g1 x VPD

Forcantes medias e estado da vegetacao da celula 186-239 (rodada
ALLOM_NUTRI_V2). Cada linha da tabela e uma combinacao; ao final, um resumo
compara o comportamento do modelo com o esperado pela fisiologia.

Uso (a partir de src/, com o caete_module compilado):
    PYTHONPATH=. python tests/carbono_agua/bancada_carbono_agua.py [saida.csv] [--src DIR]

--src DIR usa o caete_module compilado em DIR (para comparar versoes).

Resultados guardados nesta pasta:
    referencia_antes.csv  codigo do commit 6705a0b (antes do eixo carbono-agua)
    referencia_atual.csv  codigo atual (resistencia do dossel sol/sombra,
                          radiacao liquida, ci de Medlyn)
"""
import sys
import os
import numpy as np
import pandas as pd

args = sys.argv[1:]
if "--src" in args:
    i = args.index("--src")
    sys.path.insert(0, os.path.abspath(args[i + 1]))
    del args[i:i + 2]
saida = args[0] if args else None

import caete_module as m  # noqa: E402

ph_mod = m.photo
wt = m.water
pp = m.photo_par
# rotina da versao com alometria (prod_allom); em compilacoes antigas, prod
prod = getattr(m.productivity, "prod_allom", m.productivity.prod)

AQUI = os.path.dirname(os.path.abspath(__file__))
SOLO = os.path.join(AQUI, "..", "..", "..", "input", "soil")

# ---------------------------------------------------------------------------
# Celula 186-239: solo (duas camadas, 300 e 700 mm, como hydro_caete.py)
# ---------------------------------------------------------------------------
X, Y = 186, 239
ler = lambda n: float(np.load(os.path.join(SOLO, n + ".npy"))[X, Y])
WMAX = 300 * ler("ws") + 700 * ler("sws")   # saturacao (mm)
WFC = 300 * ler("fc") + 700 * ler("sfc")    # capacidade de campo (mm)
WWP = 300 * ler("wp") + 700 * ler("swp")    # ponto de murcha (mm)

# Forcantes medias da celula e estado da vegetacao (ALLOM_NUTRI_V2, ano 385)
TEMP, P0, RSDS = 27.4, 1002.0, 190.0        # oC, hPa, W m-2
IPAR = RSDS * 0.5 / 2.18e5                  # mol fotons m-2 s-1 (como caete.py)
LAT = 89.75 - 0.5 * X                       # graus (X = linha da grade, como caete.py)

# Energia disponivel no Penman: compilacoes com o argumento rnet em
# daily_budget_allom usam a radiacao liquida (FAO-56, media anual com o rsds
# medio); as antigas, a funcao de temperatura available_energy.
USA_RNET = "rnet" in m.budget_allom.daily_budget_allom.__doc__
if USA_RNET:
    sys.path.insert(0, os.path.join(AQUI, "..", ".."))
    from hydro_caete import net_radiation
LAI = 6.8
CWOOD, CROOT = 25.4, 1.16                   # kgC m-2 (sapwood + heartwood; raiz fina)
ALTURA = 25.0                               # m

# PLS lenhoso C3 representativo: medianas dos PLS lenhosos de ALLOM_NUTRI_V2
DT = np.array([
    4.0,       # 1  g1 (substituido em cada combinacao)
    0.4577,    # 2  resopfrac
    4.184,     # 3  tleaf (nao usado; longevidade vem do SLA)
    49.22,     # 4  twood
    4.201,     # 5  troot
    0.300,     # 6  aleaf
    0.3113,    # 7  awood
    0.300,     # 8  aroot
    0.0,       # 9  c4
    0.02838,   # 10 leaf_n2c
    0.007222,  # 11 awood_n2c
    0.03151,   # 12 froot_n2c
    0.004505,  # 13 leaf_p2c
    0.000784,  # 14 awood_p2c
    0.002689,  # 15 froot_p2c
    0.4972,    # 16 amp
    0.0,       # 17 pdia
    9.446,     # 18 sla_random (m2 kg-1 de materia seca)
    0.5989,    # 19 wd_random
    15.09,     # 20 restime_sap
], dtype=np.float64)

SLA = DT[17] * 1.0e-3 / 0.47                # m2 gC-1 (sla_trait)
CLEAF = LAI / (SLA * 1.0e3)                 # kgC m-2

# ---------------------------------------------------------------------------
# Grade fatorial
# ---------------------------------------------------------------------------
agua = {  # nome -> agua do solo (mm)
    "cap_campo": WFC,
    "meio_disp": 0.5 * (WFC + WWP),
    "murcha": WWP,
    "wa_0.20": 0.20 * WMAX,
    "wa_0.10": 0.10 * WMAX,
    "wa_0.05": 0.05 * WMAX,
}
co2s = [338.0, 700.0]
g1s = [2.0, 4.0, 10.0, 18.0]
vpds = [0.5, 0.77, 1.5, 2.5]                # kPa (0.77 = media da celula)


def rh_para_vpd(vpd_alvo):
    """Umidade relativa que produz o VPD desejado na temperatura TEMP."""
    lo, hi = 0.01, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ph_mod.vapor_p_defcit(TEMP, mid) > vpd_alvo:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ci usado pela compilacao testada: com photosynthesis_rate_allom (etapa 2),
# ci/ca = g1 / (g1 + sqrt(D)) (Medlyn et al. 2011); antes, a formula empirica.
CI_MEDLYN = hasattr(m.photo, "photosynthesis_rate_allom")


def ci_ca_usado(co2, rh, g1, vpd):
    if CI_MEDLYN:
        d = min(max(vpd, 0.001), 4.0)
        return g1 / (g1 + np.sqrt(d))
    return ci_ca_formula(co2, rh)


def ci_ca_formula(co2, rh):
    """ci/ca da formula empirica (funcs.f90, photosynthesis_rate)."""
    es = ph_mod.tetens(TEMP)  # hPa
    rmax = 0.622 * (es / (P0 - es))
    r = max(0.0, rmax * (1.0 - rh))
    return pp.p19 * max(0.0, 1.0 - r / pp.p20)


linhas = []
linc = np.array([IPAR], dtype=np.float64)  # uma camada, sem sombreamento por vizinhos
for nome_agua, w in agua.items():
    for co2 in co2s:
        for g1 in g1s:
            for vpd_alvo in vpds:
                rh = rh_para_vpd(vpd_alvo)
                dt = DT.copy()
                dt[0] = g1
                if USA_RNET:
                    ae = float(np.mean(net_radiation(RSDS, TEMP, rh, LAT, np.arange(1, 366))))
                else:
                    ae = wt.available_energy(TEMP)
                emax = wt.evpot2(P0, TEMP, rh, ae)
                (ph, ar, nppa, laia, f5, vpd, rm, rg, rc, wue, cdef, vm,
                 sla, e) = prod(
                    dt, co2, TEMP, TEMP, P0, w, IPAR, rh, emax,
                    CLEAF, CWOOD, CROOT, 0.0, 0.0, 0.0, ALTURA,
                    linc, ALTURA, WMAX)
                et = wt.penman(P0, TEMP, rh, ae, rc)
                # grandezas foliares derivadas das saidas de prod:
                # e (mm s-1) -> E (mol H2O m-2 s-1); A = wue * E
                E = e / 18.0e-3
                A = wue * E
                # condutancia foliar ao vapor: E = gs (VPD / P); em s m-1 via RT/P
                gs_w = E / (vpd / (P0 / 10.0))
                rc_folha = (101325.0 / (8.314 * (TEMP + 273.15))) / gs_w if gs_w > 0 else np.nan
                # ci/ca que fecha a difusao: A = gs_c (ca - ci), gs_c = gs_w / 1.6
                cica_dif = 1.0 - 1.6 * wue * (vpd / (P0 / 10.0)) / (co2 * 1e-6) if wue > 0 else np.nan
                linhas.append(dict(
                    agua=nome_agua, w_mm=w,
                    f_disp=(w - WWP) / (WFC - WWP),
                    co2=co2, g1=g1, vpd=vpd, rh=rh,
                    gpp_kgC_ano=ph,
                    npp_kgC_ano=nppa,
                    f5=f5,
                    A_folha_umol=A * 1e6,
                    rc_folha_s_m=rc_folha,
                    rc_dossel_s_m=rc,
                    et_penman_mm_d=et,
                    et_potencial_mm_d=emax,
                    energia_W_m2=ae,
                    transp_folha_mm_d=e * 86400.0,
                    cica_difusao=cica_dif,
                    cica_usado=ci_ca_usado(co2, rh, g1, vpd),
                    cica_medlyn=g1 / (g1 + np.sqrt(vpd)),
                ))

df = pd.DataFrame(linhas)
if saida:
    df.to_csv(saida, index=False, float_format="%.6g")

# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------
pd.set_option("display.width", 200)
fmt = lambda v: "%.3f" % v
print("Celula %d-%d | saturacao %.0f mm | cap. campo %.0f mm | murcha %.0f mm"
      % (X, Y, WMAX, WFC, WWP))
print("T %.1f oC | rsds %.0f W m-2 | LAI %.1f | raiz fina %.2f kgC m-2"
      % (TEMP, RSDS, LAI, CROOT))
print("Energia no Penman: %s, %.0f W m-2 (cap. de campo, VPD 0.77)\n"
      % ("radiacao liquida FAO-56" if USA_RNET else "funcao da temperatura",
         df[(df.vpd.round(2) == 0.77)].energia_W_m2.iloc[0]))

base = df[(df.agua == "cap_campo") & (df.co2 == 338) & (df.vpd.round(2) == 0.77)]
print("1) Papel do g1 (solo na cap. de campo, CO2 338, VPD 0.77)")
print(base[["g1", "gpp_kgC_ano", "A_folha_umol", "rc_folha_s_m", "rc_dossel_s_m", "et_penman_mm_d",
            "cica_usado", "cica_difusao", "cica_medlyn"]].to_string(index=False, float_format=fmt))

print("\n2) Seca (g1 4, VPD 0.77): GPP relativa a cap. de campo")
for co2 in co2s:
    s = df[(df.co2 == co2) & (df.g1 == 4) & (df.vpd.round(2) == 0.77)].copy()
    s["gpp_rel"] = s.gpp_kgC_ano / s.gpp_kgC_ano.iloc[0]
    print("   CO2 %.0f" % co2)
    print(s[["agua", "f_disp", "f5", "gpp_kgC_ano", "gpp_rel", "et_penman_mm_d"]]
          .to_string(index=False, float_format=fmt))

print("\n3) VPD (cap. de campo, CO2 338, g1 4)")
s = df[(df.agua == "cap_campo") & (df.co2 == 338) & (df.g1 == 4)]
print(s[["vpd", "gpp_kgC_ano", "rc_folha_s_m", "et_penman_mm_d", "et_potencial_mm_d",
         "cica_usado", "cica_medlyn"]].to_string(index=False, float_format=fmt))

# Verificacoes contra o esperado pela fisiologia
def g(agua_, co2, g1=4.0, vpd=0.77, col="gpp_kgC_ano"):
    r = df[(df.agua == agua_) & (df.co2 == co2) & (df.g1 == g1) & (df.vpd.round(2) == vpd)]
    return float(r[col].iloc[0])

print("\n4) Verificacoes (esperado pela fisiologia -> modelo)")
v = []
r_g1 = g("cap_campo", 338, 18) / g("cap_campo", 338, 2)
v.append(("g1 afeta a fotossintese (GPP g1=18 / g1=2 > 1.05)", "%.3f" % r_g1, r_g1 > 1.05))
dif = np.nanmax(np.abs(base.cica_usado - base.cica_difusao))
v.append(("ci/ca usado = ci/ca da difusao (dif. max < 0.02)", "%.3f" % dif, dif < 0.02))
r_et = g("cap_campo", 700, col="et_penman_mm_d") / g("cap_campo", 338, col="et_penman_mm_d")
v.append(("CO2 elevado reduz ET (ET700/ET338 < 0.95)", "%.3f" % r_et, r_et < 0.95))
r_gpp = g("cap_campo", 700) / g("cap_campo", 338)
v.append(("CO2 elevado aumenta GPP (1.1 a 1.6)", "%.3f" % r_gpp, 1.1 < r_gpp < 1.6))
f5_meio = g("meio_disp", 338, col="f5")
v.append(("Estresse com metade da agua disponivel (f5 < 0.95)", "%.3f" % f5_meio, f5_meio < 0.95))
f5_wp = g("murcha", 338, col="f5")
v.append(("Estresse forte no ponto de murcha (f5 < 0.2)", "%.3f" % f5_wp, f5_wp < 0.2))
q338 = g("meio_disp", 338) / g("cap_campo", 338)
q700 = g("meio_disp", 700) / g("cap_campo", 700)
v.append(("CO2 elevado atenua a seca (GPP rel. meio_disp: 700 > 338)",
          "%.3f vs %.3f" % (q700, q338), q700 > q338 + 0.005))
r_vpd = g("cap_campo", 338, vpd=2.5) / g("cap_campo", 338, vpd=0.5)
v.append(("VPD alto reduz a fotossintese (GPP 2.5/0.5 kPa < 0.95)", "%.3f" % r_vpd, r_vpd < 0.95))
et_ok = bool((df.et_penman_mm_d <= df.et_potencial_mm_d + 1e-9).all())
v.append(("ET real <= ET potencial em todas as combinacoes", str(et_ok), et_ok))
for desc, val, ok in v:
    print("   [%s] %-62s %s" % ("ok " if ok else "NAO", desc, val))
