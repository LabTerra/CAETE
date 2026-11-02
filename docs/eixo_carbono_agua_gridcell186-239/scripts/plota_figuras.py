"""Figuras do eixo carbono-água (versão com alometria e ciclo de nutrientes).

Lê só as tabelas de ../tabelas e as referências da bancada em
src/tests/carbono_agua/ (não precisa do modelo compilado nem das saídas).

Uso (de qualquer pasta):
    python plota_figuras.py            # todas as figuras
    python plota_figuras.py 1 9        # só as figuras 1 e 9

As figuras são gravadas em ../figuras. Ver o README para o conteúdo de cada uma.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(AQUI, "..", "tabelas")
FIG = os.path.join(AQUI, "..", "figuras")
BANCADA = os.path.join(AQUI, "..", "..", "..", "src", "tests", "carbono_agua")

ESTILO = {"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
          "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"}
C_ANT, C_NOV, C_REF, C_OBS = "#9a9a9a", "#1f6fb4", "#d9822b", "#2e7d32"
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
CAIXA = dict(boxstyle="round,pad=0.4", facecolor="#f4f8fc", edgecolor=C_NOV, lw=0.8)


def tabela(nome, **kw):
    return pd.read_csv(os.path.join(TAB, nome), **kw)


def bancada():
    """Bancada (tests/carbono_agua): antes (commit 6705a0b) e código atual."""
    return (pd.read_csv(os.path.join(BANCADA, "referencia_antes.csv")),
            pd.read_csv(os.path.join(BANCADA, "referencia_atual.csv")))


def linha_base(d, co2=338, vpd=0.77):
    """Condições médias da célula 186-239: solo úmido, CO2 338 ppm, VPD 0,77 kPa; g1 varia."""
    return d[(d.agua == "cap_campo") & (d.co2 == co2) & (d.vpd.round(2) == vpd)].sort_values("g1")


def salva(fig, nome):
    fig.savefig(os.path.join(FIG, nome))
    plt.close(fig)
    print(nome)


def virgula(v, n=2):
    return f"{v:.{n}f}".replace(".", ",")


# ---------------------------------------------------------------------------------
def fig1():
    """Mudança 1: resistência do dossel e evapotranspiração em função do g1 (bancada)."""
    ant, nov = bancada()
    a, n = linha_base(ant), linha_base(nov)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(a.g1, a.rc_dossel_s_m, "o-", c=C_ANT, label="antes: rc folha × LAI")
    ax[0].plot(n.g1, n.rc_dossel_s_m, "o-", c=C_NOV, label="atual: sol/sombra (2 folhas grandes)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("g1 (kPa$^{0,5}$)")
    ax[0].set_ylabel("resistência do dossel (s m$^{-1}$)")
    ax[0].axhspan(50, 135, color="#e8f0e0", zorder=0)
    ax[0].text(10.6, 150, "medido ao meio-dia, Manaus:\n50–135 (Malhi et al. 2002)", fontsize=7, color="#4a7a2a")
    ax[0].legend(fontsize=7, frameon=False, loc="upper right")
    ax[1].plot(a.g1, a.et_penman_mm_d, "o-", c=C_ANT, label="antes")
    ax[1].plot(n.g1, n.et_penman_mm_d, "o-", c=C_NOV, label="atual")
    ax[1].axhspan(3.0, 3.75, color="#e8f0e0", zorder=0)
    ax[1].text(7.5, 2.6, "Amazônia: 3,0–3,75 (Fisher et al. 2009)\nManaus: 3,08 (Malhi et al. 2002)",
               fontsize=7, color="#4a7a2a")
    ax[1].set_xlabel("g1 (kPa$^{0,5}$)")
    ax[1].set_ylabel("evapotranspiração (mm dia$^{-1}$)")
    ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Bancada, célula 186-239 (LAI 6,8; VPD 0,77 kPa; CO$_2$ 338 ppm; solo úmido)", fontsize=10)
    salva(fig, "fig1_resistencia_dossel_ET.png")


# ---------------------------------------------------------------------------------
def fig2():
    """Mudança 2: água no solo no teste K34 e teste isolado de seca."""
    solo = tabela("solo_celulas.csv", index_col=0, dtype={"celula": str}).loc["185-240"]
    wfc, wwp = solo.capacidade_campo_mm, solo.ponto_murcha_mm
    k = tabela("k34_agua_solo_diaria.csv", parse_dates=["data"])
    k = k[k.data >= "1987-01-01"]
    datas = k.data.values
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})

    a = ax[0]
    a.axhspan(wwp, wfc, color="#eef5e8", zorder=0)
    a.axhspan(0, wwp, color="#fbeaea", zorder=0)
    a.text(datas[15], (wwp + wfc) / 2, "água disponível\npara as plantas", fontsize=8, color="#4a7a2a", va="center")
    a.text(datas[15], wwp / 2, "abaixo do ponto de murcha:\nas raízes não extraem", fontsize=8, color="#a33",
           va="center")
    a.axhline(wfc, ls="--", c="b", lw=0.8)
    a.axhline(wwp, ls="--", c="r", lw=0.8)
    a.plot(datas, k.antes, c=C_ANT, lw=1.4, label="antes (rc × LAI, sem limite)")
    a.plot(datas, k.mudanca1, c=C_REF, lw=1.4, label="só mudança 1 (ET realista, sem limite)")
    a.plot(datas, k.mudancas1e2, c=C_NOV, lw=1.6, label="mudanças 1 + 2 (ET realista, com limite)")
    a.set_xlim(datas[0], datas[-1] + np.timedelta64(1, "D"))
    a.set_ylim(0, 520)
    a.set_ylabel("água no solo, 0–100 cm (mm)")
    a.xaxis.set_major_locator(mdates.YearLocator())
    a.xaxis.set_major_formatter(plt.NullFormatter())
    a.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7, bymonthday=1))
    a.xaxis.set_minor_formatter(mdates.DateFormatter("%Y"))
    a.tick_params(axis="x", which="minor", length=0)
    a.legend(fontsize=8, frameon=False, loc="lower right")
    a.set_title("(a) Teste K34 (Manaus): água no solo, 1987–1989", fontsize=10, loc="left")
    a.text(datas[-1], wfc + 6, "capacidade de campo", fontsize=7, color="b", ha="right")
    a.text(datas[-1], wwp + 6, "ponto de murcha", fontsize=7, color="r", ha="right")

    s = tabela("teste_seca_isolado_diario.csv").iloc[:1100]
    n = len(s)
    b = ax[1]
    b.axhspan(wwp, wfc, color="#eef5e8", zorder=0)
    b.axhspan(0, wwp, color="#fbeaea", zorder=0)
    b.axhline(wwp, ls="--", c="r", lw=0.8)
    b.axhline(wfc, ls="--", c="b", lw=0.8)
    for j in range(2):
        if 500 + 730 * j < n:
            b.axvspan(500 + 730 * j, min(730 * (j + 1), n), color="#f4f0e0", zorder=0.5)
    b.text(505, 495, "sem chuva", fontsize=7, color="#8a7a3a")
    b.plot(s.dia, s.agua_metodo_antigo, c=C_ANT, lw=1.4, label="método antigo")
    b.plot(s.dia, s.agua_metodo_novo, c=C_NOV, lw=1.6, label="método novo")
    b.set_xlim(0, n)
    b.set_ylim(0, 520)
    b.set_xlabel("dias")
    b.set_ylabel("água no solo (mm)")
    b2 = b.twinx()
    b2.plot(s.dia, s.agua_criada_acumulada, c="k", ls=":", lw=1.2, label="água criada pelo método antigo (acumulada)")
    b2.set_ylabel("água criada (mm, acumulada)")
    b2.spines["top"].set_visible(False)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="center left")
    b.set_title("(b) Teste isolado: seca (demanda 3–6 mm dia$^{-1}$; chuva média 1,8 mm dia$^{-1}$)",
                fontsize=10, loc="left")
    salva(fig, "fig2_agua_solo_limite_extracao.png")


# ---------------------------------------------------------------------------------
def fig3():
    """Mudanças 1 a 5 na rodada longa: ciclo anual da água, da evapotranspiração e da GPP."""
    solo = tabela("solo_celulas.csv", index_col=0, dtype={"celula": str}).loc["186-239"]
    wfc, wwp = solo.capacidade_campo_mm, solo.ponto_murcha_mm
    m = tabela("ciclo_anual_mensal.csv")
    x = np.arange(12)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(x, m.agua_V2, "o-", c=C_ANT, label="V2 (antes)")
    ax[0].plot(x, m.agua_atual, "o-", c=C_NOV, label="atual (TEST_WATER4)")
    ax[0].axhline(wfc, ls="--", c="b", lw=1)
    ax[0].axhline(wwp, ls="--", c="r", lw=1)
    ax[0].text(6.5, wfc + 4, "capacidade de campo", color="b", fontsize=7)
    ax[0].text(6.5, wwp + 4, "ponto de murcha", color="r", fontsize=7)
    ax[0].set_ylabel("água no solo (mm)")
    ax[0].set_ylim(170, 390)
    ax[0].legend(fontsize=8, frameon=False, loc="center left")
    ax[1].bar(x, m.chuva, color="#cfe3f3", label="chuva")
    ax[1].plot(x, m.ET_V2, "o-", c=C_ANT, label="ET V2")
    ax[1].plot(x, m.ET_atual, "o-", c=C_NOV, label="ET atual")
    ax[1].axhline(3.08, ls=":", c="#4a7a2a", lw=1)
    ax[1].set_ylim(0, 12.5)
    ax[1].text(-0.4, 10.0, "pontilhado: ET medida em Manaus, 3,08 (Malhi et al. 2002)", fontsize=6, color="#2a5a1a")
    ax[1].set_ylabel("mm dia$^{-1}$")
    ax[1].legend(fontsize=8, frameon=False, loc="upper right", ncol=3, columnspacing=0.8)
    ax[2].plot(x, m.GPP_V2, "o-", c=C_ANT, label="V2")
    ax[2].plot(x, m.GPP_atual, "o-", c=C_NOV, label="atual")
    ax[2].set_ylabel("GPP (kgC m$^{-2}$ ano$^{-1}$)")
    ax[2].legend(fontsize=8, frameon=False)
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(MESES, fontsize=8)
    fig.suptitle("Ciclo anual médio, célula 186-239, período histórico (1979-2016)", fontsize=10)
    salva(fig, "fig3_ciclo_anual_agua_ET_GPP.png")


# ---------------------------------------------------------------------------------
def fig4():
    """Mudança 3: balanço de radiação e energia disponível ao longo do ano."""
    r = tabela("radiacao_mensal.csv", dtype={"mes": str}).set_index("mes")
    ano, mm = r.loc["ano"], r.drop("ano")
    S, RN, AE = ano.rsds, ano.Rn_FAO56, ano.energia_antes
    refl = 0.13 * S
    lw = (1 - 0.13) * S - RN
    C_SOL, C_LW = "#f2b134", "#e07b67"
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.25]})

    a = ax[0]
    casos = [("Medido, floresta\nde Manaus\n(Malhi et al. 2002)", 11.0, 15.0, 74.0),
             ("Atual: FAO-56\n(célula 186-239)", 100 * refl / S, 100 * lw / S, 100 * RN / S)]
    for i, (nome, rf, l, e) in enumerate(casos):
        y = len(casos) - i
        a.barh(y, e, color=C_NOV, height=0.55)
        a.barh(y, l, left=e, color=C_LW, height=0.55)
        a.barh(y, rf, left=e + l, color="#d6d6d6", height=0.55)
        a.text(e / 2, y, f"{e:.0f}%", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        a.text(e + l / 2, y, f"{l:.0f}%", ha="center", va="center", fontsize=8)
        a.text(e + l + rf / 2, y, f"{rf:.0f}%", ha="center", va="center", fontsize=8)
    a.barh(0, 100 * AE / S, color="none", edgecolor=C_ANT, hatch="///", height=0.55)
    a.text(100 * AE / S / 2, 0, f"{100 * AE / S:.0f}% em média,\nmas não depende do sol", ha="center",
           va="center", fontsize=7, color="#444", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    a.set_yticks([2, 1, 0])
    a.set_yticklabels([casos[0][0], casos[1][0], "Antes:\n2,895·T + 52,3"], fontsize=8)
    a.set_xlim(0, 100)
    a.set_xlabel("% da radiação solar que chega à floresta")
    a.barh(-10, 0, color=C_NOV, label="energia disponível (radiação líquida)")
    a.barh(-10, 0, color=C_LW, label="perdida como calor (onda longa)")
    a.barh(-10, 0, color="#d6d6d6", label="refletida (albedo)")
    a.set_ylim(-0.6, 2.6)
    a.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    a.set_title("(a) Para onde vai a radiação solar (média anual)", fontsize=10, loc="left")

    b = ax[1]
    x = np.arange(12)
    b.plot(x, mm.rsds, "o--", c=C_SOL, lw=1.2, ms=4, label="radiação solar do forçamento")
    b.plot(x, mm.energia_antes, "o-", c=C_ANT, lw=2, label="antes: energia = 2,895·T + 52,3")
    b.plot(x, mm.Rn_FAO56, "o-", c=C_NOV, lw=2, label="atual: radiação líquida (FAO-56)")
    b.annotate(f"antes: quase constante\n({mm.energia_antes.min():.0f} a {mm.energia_antes.max():.0f} W m$^{{-2}}$)",
               xy=(4, mm.energia_antes.iloc[4]), xytext=(1.0, 108), fontsize=8, color="#555",
               arrowprops=dict(arrowstyle="->", color="#999"))
    b.annotate(f"atual: acompanha o sol\n({mm.Rn_FAO56.min():.0f} a {mm.Rn_FAO56.max():.0f} W m$^{{-2}}$)",
               xy=(9, mm.Rn_FAO56.iloc[9]), xytext=(6.6, 105), fontsize=8, color=C_NOV,
               arrowprops=dict(arrowstyle="->", color=C_NOV))
    b.axvspan(6.5, 10.5, color="#fbf3dc", zorder=0)
    b.text(8.5, 228, "estação seca\n(mais sol)", ha="center", fontsize=7, color="#8a7a3a")
    b.set_xticks(x)
    b.set_xticklabels(MESES, fontsize=8)
    b.set_ylabel("W m$^{-2}$")
    b.set_ylim(95, 240)
    b.legend(fontsize=7, frameon=False, loc="upper left")
    b.set_title("(b) Energia disponível ao longo do ano (célula 186-239, 1979-2016)", fontsize=10, loc="left")
    salva(fig, "fig4_radiacao_liquida.png")


# ---------------------------------------------------------------------------------
def fig5():
    """Mudança 4: efeitos do g1 e do VPD no ci, na fotossíntese e na transpiração (bancada)."""
    ant, nov = bancada()
    a0, n0 = linha_base(ant), linha_base(nov)
    vp = lambda d: d[(d.agua == "cap_campo") & (d.co2 == 338) & (d.g1 == 4)].sort_values("vpd")
    av, nv = vp(ant), vp(nov)
    C_ET = "#2a9d8f"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    a = ax[0]
    g = np.linspace(1, 18, 200)
    a.axvspan(1.6, 8.6, color="#eef3f9", zorder=0)
    a.text(5.1, 0.60, "g1 sorteado hoje\n(C3: 1,6 a 8,6)", ha="center", fontsize=7, color="#557")
    a.plot(g, g / (g + np.sqrt(0.77)), c="k", lw=1, label="exigido pela difusão (Medlyn)")
    a.plot(a0.g1, a0.cica_usado, "o-", c=C_ANT, lw=2, label="antes: ci/ca usado (fixo)")
    a.plot(n0.g1, n0.cica_usado, "o", c=C_NOV, ms=8, label="atual: ci/ca usado")
    a.set_xlabel("g1 (kPa$^{0,5}$)")
    a.set_ylabel("ci / ca  (CO$_2$ dentro ÷ fora da folha)")
    a.set_ylim(0.55, 1.0)
    a.legend(fontsize=7, frameon=False, loc="lower right")
    a.text(9.5, 0.72, "g1 alto → estômatos mais\nabertos → ci maior", fontsize=8, bbox=CAIXA)
    a.annotate("antes: o mesmo ci para\nqualquer g1 (incoerente)", xy=(15, 0.855), xytext=(11.3, 0.785),
               fontsize=7, color="#555", arrowprops=dict(arrowstyle="->", color="#999"))
    a.set_title("(a) CO$_2$ dentro da folha × g1", fontsize=10, loc="left")

    b = ax[1]
    b.axvspan(1.6, 8.6, color="#eef3f9", zorder=0)
    b.plot(a0.g1, a0.gpp_kgC_ano, "o-", c=C_ANT, lw=2, label="GPP antes")
    b.plot(n0.g1, n0.gpp_kgC_ano, "o-", c=C_NOV, lw=2, label="GPP atual")
    b.set_xlabel("g1 (kPa$^{0,5}$)")
    b.set_ylabel("GPP (kgC m$^{-2}$ ano$^{-1}$)", color=C_NOV)
    b2 = b.twinx()
    b2.spines["top"].set_visible(False)
    b2.plot(n0.g1, n0.et_penman_mm_d, "s--", c=C_ET, lw=1.6, label="evapotranspiração atual")
    b2.set_ylabel("evapotranspiração (mm dia$^{-1}$)", color=C_ET)
    b2.set_ylim(2.5, 5.0)
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="lower right")
    b.set_ylim(3.4, 4.8)
    b.text(2.0, 4.62, "g1 alto → mais fotossíntese\ne mais transpiração\n(troca carbono × água)", fontsize=8,
           bbox=CAIXA)
    b.annotate("antes: GPP igual\npara qualquer g1", xy=(14, 4.308), xytext=(11.5, 3.75), fontsize=7,
               color="#555", arrowprops=dict(arrowstyle="->", color="#999"))
    b.set_title("(b) Fotossíntese e transpiração × g1", fontsize=10, loc="left")

    c = ax[2]
    c.plot(nv.vpd, nv.cica_usado, "o-", c=C_NOV, lw=2, label="ci/ca atual")
    c.plot(av.vpd, av.cica_usado, "o--", c=C_ANT, lw=1.4, label="ci/ca antes (fórmula de umidade)")
    c.set_xlabel("VPD, secura do ar (kPa)")
    c.set_ylabel("ci / ca")
    c.set_ylim(0.66, 0.9)
    c2 = c.twinx()
    c2.spines["top"].set_visible(False)
    c2.plot(nv.vpd, nv.gpp_kgC_ano, "s-", c="#e07b67", lw=1.6, label="GPP atual")
    c2.set_ylabel("GPP (kgC m$^{-2}$ ano$^{-1}$)", color="#e07b67")
    c2.set_ylim(3.6, 4.4)
    h1, l1 = c.get_legend_handles_labels()
    h2, l2 = c2.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="lower left")
    c.text(1.15, 0.86, "ar seco (VPD alto) → estômatos\nfecham → ci menor →\nmenos fotossíntese", fontsize=8,
           bbox=CAIXA)
    c.set_title("(c) Efeito da secura do ar (g1 = 4)", fontsize=10, loc="left")

    fig.suptitle("Bancada, célula 186-239: um PLS em condições médias (CO$_2$ 338 ppm, solo úmido, LAI 6,8); "
                 "só g1 ou VPD variam", fontsize=9, y=1.02)
    fig.tight_layout()
    salva(fig, "fig5_ci_ca_GPP_g1.png")


# ---------------------------------------------------------------------------------
def fig6():
    """Mudança 4: esquema do fluxo ci -> A -> gs -> transpiração, antes e agora."""
    C_ERR, TXT = "#c0392b", "#2b2b2b"
    plt.rcParams.update({"font.size": 9.5})
    fig, ax = plt.subplots(figsize=(14, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.07, 1)
    ax.axis("off")
    W, H = 0.155, 0.13
    XS = [0.09, 0.29, 0.49, 0.69, 0.89]

    def caixa(x, y, titulo, corpo, cor, fundo="white"):
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.006,rounding_size=0.015",
                                    facecolor=fundo, edgecolor=cor, lw=1.4, zorder=3))
        ax.text(x, y + 0.03, titulo, ha="center", va="center", fontsize=10, fontweight="bold", color=cor, zorder=4)
        ax.text(x, y - 0.022, corpo, ha="center", va="center", fontsize=8.3, color=TXT, zorder=4, linespacing=1.3)

    def seta(x0, y0, x1, y1, cor, ls="-", rad=0.0, lw=1.6):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=cor, lw=lw,
                                     linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))

    def linha(y, cor, nomes, fundo_ci, g1_no_ci):
        for i in range(4):
            seta(XS[i] + W / 2, y, XS[i + 1] - W / 2, y, cor)
        for x, (t, c), f in zip(XS, nomes, ["white", fundo_ci, "white", "white", "white"]):
            caixa(x, y, t, c, cor if f == "white" else (C_ERR if cor == C_ANT else C_NOV), fundo=f)
        ax.text(XS[3], y + 0.165, "g1 do PLS", ha="center", fontsize=9, fontweight="bold", color=TXT)
        seta(XS[3], y + 0.15, XS[3], y + H / 2 + 0.005, TXT, lw=1.3)
        if g1_no_ci:
            seta(XS[3] - 0.035, y + 0.155, XS[1] + 0.03, y + H / 2 + 0.005, TXT, lw=1.3, rad=0.12)

    yA = 0.75
    ax.text(0.005, 0.975, "ANTES", fontsize=13, fontweight="bold", color=C_ANT, va="top")
    ax.text(0.085, 0.972, "o ci era fixado pelo clima; o g1 só entrava no gs", fontsize=10, color=TXT, va="top")
    linha(yA, C_ANT, [("Clima", "temperatura e\numidade do ar"),
                      ("ci", "ci/ca ≈ 0,87\nfórmula fixa,\nsem g1"),
                      ("Fotossíntese A", "Farquhar,\ncalculada com\nesse ci"),
                      ("gs (Medlyn)", "1,6 (1 + g1/√D) A/ca"),
                      ("Transpiração", "Penman, com\nrc = 1/gs")], "#fdf0ee", False)
    seta(XS[3] - 0.01, yA - H / 2 - 0.005, XS[1] + 0.01, yA - H / 2 - 0.005, C_ERR, ls="--", rad=-0.32)
    ax.text(XS[2], yA - 0.205, "difusão:  A = gs (ca − ci) / 1,6   ✗  não confere",
            ha="center", fontsize=9.5, fontweight="bold", color=C_ERR)
    ax.text(XS[2], yA - 0.243, "o gs exige um ci; a fotossíntese usou outro. Ex. (D = 0,77 kPa): g1 = 10 → "
            "A com ci/ca = 0,87, mas os estômatos dariam 0,92", ha="center", fontsize=8.5, color=TXT)

    yB = 0.19
    ax.plot([0.01, 0.99], [0.465, 0.465], c="#dddddd", lw=1)
    ax.text(0.005, 0.44, "AGORA", fontsize=13, fontweight="bold", color=C_NOV, va="top")
    ax.text(0.09, 0.437, "o ci sai do g1 e da secura do ar, coerente com o gs (Medlyn et al. 2011, Eq. 13)",
            fontsize=10, color=TXT, va="top")
    linha(yB, C_NOV, [("Ar", "secura do ar\n(déficit de pressão\nde vapor, D)"),
                      ("ci", "ci/ca = g1 / (g1 + √D)"),
                      ("Fotossíntese A", "Farquhar,\ncalculada com\nesse ci"),
                      ("gs (Medlyn)", "1,6 (1 + g1/√D) A/ca"),
                      ("Transpiração", "Penman, com\nrc = 1/gs")], "#eef4fb", True)
    seta(XS[3] - 0.01, yB - H / 2 - 0.005, XS[1] + 0.01, yB - H / 2 - 0.005, C_OBS, ls="--", rad=-0.32)
    ax.text(XS[2], yB - 0.205, "difusão:  A = gs (ca − ci) / 1,6   ✓  confere",
            ha="center", fontsize=9.5, fontweight="bold", color=C_OBS)
    ax.text(XS[2], yB - 0.243, "g1 alto → estômatos mais abertos → ci maior → mais fotossíntese e mais "
            "transpiração.   Ar seco (D alto) → estômatos fecham → ci menor → menos fotossíntese",
            ha="center", fontsize=8.5, color=TXT)
    fig.savefig(os.path.join(FIG, "fig6_esquema_ci_gs.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(ESTILO)
    print("fig6_esquema_ci_gs.png")


# ---------------------------------------------------------------------------------
def fig7():
    """Mudança 5: intervalo do g1 (dados de Lin et al. 2015 e sorteio do plsgen.py)."""
    C_C4 = "#2a9d8f"
    caixa_c = dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor=C_ANT, lw=0.8)
    sp = tabela("g1_lin2015_por_especie.csv")
    trop = sp[(sp.form == "tree") & (sp.treg == "tropical")].g1
    c3, c4 = sp[sp.path == "C3"].g1, sp[sp.path == "C4"].g1
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.7))
    bins = np.arange(0, 18.5, 0.5)

    a = ax[0]
    a.axvspan(2, 18, color=C_ANT, alpha=0.12, zorder=0)
    a.axvspan(10, 18, color=C_ANT, alpha=0.12, zorder=0)
    a.hist(c3, bins=bins, color="#b7cde6", label=f"todas as C3 ({len(c3)} espécies)")
    a.hist(trop, bins=bins, color=C_NOV, label=f"árvores tropicais ({len(trop)})")
    a.hist(c4, bins=bins, histtype="step", color=C_C4, lw=1.6, label=f"C4 ({len(c4)})")
    a.axvline(np.median(trop), c=C_NOV, ls="--", lw=1)
    a.text(np.median(trop) + 0.2, 24.5, f"mediana das árvores\ntropicais: {virgula(np.median(trop), 1)}",
           fontsize=7, color=C_NOV)
    a.text(13.7, 20, "sorteio antigo:\n2 a 18", ha="center", fontsize=8, color="#666")
    a.text(13.7, 6.5, "de 10 a 18:\nquase nenhuma\nespécie", ha="center", fontsize=8, bbox=caixa_c)
    a.set_xlabel("g1 (kPa$^{0,5}$)")
    a.set_ylabel("número de espécies")
    a.set_xlim(0, 18)
    a.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.72))
    a.set_title("(a) O que as plantas medidas mostram\n(Lin et al. 2015, g1 por espécie)", fontsize=10, loc="left")

    b = ax[1]
    g = np.linspace(0.3, 18.5, 2000)
    dens_old = np.where((g >= 2) & (g <= 18), 1 / 16, 0)
    logu = lambda g, lo, hi: np.where((g >= lo) & (g <= hi), 1 / (g * np.log(hi / lo)), 0)
    b.fill_between(g, dens_old, color=C_ANT, alpha=0.35, step="mid", label="antes: uniforme 2 a 18 (C3 e C4)")
    b.plot(g, logu(g, 1.6, 8.6), c=C_NOV, lw=2.2, label="atual C3: log-uniforme 1,6 a 8,6")
    b.plot(g, logu(g, 0.7, 4.1), c=C_C4, lw=2.2, label="atual C4: log-uniforme 0,7 a 4,1")
    b.axvline(10, c=C_ANT, ls=":", lw=1.2)
    b.text(10.2, 0.55, "média antiga: 10", fontsize=8, color="#666")
    b.axvline(np.sqrt(1.6 * 8.6), c=C_NOV, ls=":", lw=1.2)
    b.text(np.sqrt(1.6 * 8.6) + 0.2, 0.66, "mediana atual C3: 3,7", fontsize=8, color=C_NOV)
    b.text(9.3, 0.30, "log-uniforme: cada faixa em que o\ng1 dobra (1,6–3,2; 3,2–6,4; ...)\ntem a mesma "
           "chance → mais PLSs\ncom g1 baixo, como nas espécies\nmedidas (painel a)", fontsize=8, bbox=CAIXA)
    b.set_xlabel("g1 (kPa$^{0,5}$)")
    b.set_ylabel("densidade de probabilidade")
    b.set_xlim(0, 18)
    b.set_ylim(0, 0.88)
    b.legend(fontsize=7, frameon=False, loc="upper right")
    b.set_title("(b) Como o plsgen.py sorteia o g1\n(limites = percentis 5 e 95 das espécies)", fontsize=10,
                loc="left")
    fig.tight_layout()
    salva(fig, "fig7_g1_lin2015.png")


# ---------------------------------------------------------------------------------
def fig8():
    """Rodadas longas: GPP, evapotranspiração, escoamento e reserva de N, por conjunto de mudanças."""
    T = tabela("rodadas_longas_historico.csv", index_col=0)
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    rot = ["V2\n(antes)", "TW2\n(1-3)", "TW3\n(1-4)", "TW4\n(1-5)"]
    cor = [C_ANT, "#9cc3e4", "#5d9bd1", C_NOV]
    for a, v, t in zip(ax, ["photo", "evapm", "runom", "nsto"],
                       ["GPP (kgC m$^{-2}$ ano$^{-1}$)", "evapotranspiração (mm dia$^{-1}$)",
                        "escoamento (mm dia$^{-1}$)", "reserva de N (gN m$^{-2}$)"]):
        a.bar(range(4), T.loc[v, ["V2", "TW2", "TW3", "TW4"]].values, color=cor)
        a.set_xticks(range(4))
        a.set_xticklabels(rot, fontsize=7)
        a.set_title(t, fontsize=9)
    fig.suptitle("Rodadas longas, célula 186-239, média do período histórico "
                 "(os números indicam as mudanças incluídas)", fontsize=9, y=1.06)
    salva(fig, "fig8_rodadas_longas.png")


# ---------------------------------------------------------------------------------
def fig9():
    """Síntese: modelo ÷ medido antes e agora, e ciclo anual da água."""
    T = tabela("rodadas_longas_historico.csv", index_col=0)
    M = tabela("ciclo_anual_mensal.csv")
    O = tabela("observacoes.csv", index_col=0)
    solo = tabela("solo_celulas.csv", index_col=0, dtype={"celula": str}).loc["186-239"]
    plt.rcParams.update({"font.size": 9.5})
    C_P = "#dbe7f3"
    P = M.chuva.mean()
    v, a = T["V2"], T["TW4"]
    # (nome, antes, agora, linha de observacoes.csv, casas decimais)
    linhas = [("ÁGUA", None),
              ("Evapotranspiração (mm dia⁻¹)", v.evapm, a.evapm, "evapotranspiracao", 2),
              ("Fração da chuva que volta ao ar (%)", 100 * v.evapm / P, 100 * a.evapm / P, "fracao_chuva_ET", 0),
              ("Resistência do dossel¹ (s m⁻¹)", v.rcm, a.rcm, "resistencia_dossel", 0),
              ("CARBONO", None),
              ("GPP (kgC m⁻² ano⁻¹)", v.photo, a.photo, "gpp", 2),
              ("Respiração autotrófica (Ra) (kgC m⁻² ano⁻¹)", v.ar, a.ar, "ra", 2),
              ("NPP (kgC m⁻² ano⁻¹)", v.npp, a.npp, "npp", 2),
              ("Eficiência no uso do carbono (NPP/GPP)", v.npp / v.photo, a.npp / a.photo, "cue", 2)]

    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.62)
    ax = fig.add_subplot(gs[0])
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.axvline(1, c=C_OBS, lw=1.2, zorder=1)
    y, yt, yl = 0, [], []
    for ln in linhas:
        y -= 1
        if ln[1] is None:
            ax.text(0.26, y + 0.15, ln[0], fontsize=9, fontweight="bold", color="#555", va="center")
            continue
        nome, x0, x1, chave, n = ln
        o = O.loc[chave]
        obs = o.medido
        if pd.notna(o.faixa_min):
            ax.fill_betweenx([y - 0.28, y + 0.28], o.faixa_min / obs, o.faixa_max / obs, color="#e3f0e3", lw=0, zorder=0)
            txt_obs = (f"{virgula(o.faixa_min, n)}–{virgula(o.faixa_max, n)}" if chave == "resistencia_dossel"
                       else virgula(obs, n))
        else:
            txt_obs = virgula(obs, n)
        ax.annotate("", xy=(x1 / obs, y), xytext=(x0 / obs, y),
                    arrowprops=dict(arrowstyle="-|>", color="#7a9cc0", lw=1.4, shrinkA=5, shrinkB=6, mutation_scale=12))
        ax.plot(x0 / obs, y, "o", c=C_ANT, ms=9, zorder=3)
        ax.plot(x1 / obs, y, "o", c=C_NOV, ms=10, zorder=4)
        ax.text(1.03, y, f"{virgula(x0, n)} → {virgula(x1, n)}", transform=tr, va="center", fontsize=8.5, color="#333")
        ax.text(1.24, y, txt_obs, transform=tr, va="center", fontsize=8.5, color=C_OBS)
        yt.append(y)
        yl.append(nome)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 20)
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 10, 20])
    ax.set_xticklabels(["0,25", "0,5", "1", "2", "4", "10", "20"])
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.set_ylim(y - 0.7, -0.4)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("modelo ÷ medido   (1 = igual ao medido; escala log)")
    ax.text(1.03, -0.45, "antes → agora", transform=tr, fontsize=8.5, fontweight="bold", color="#333")
    ax.text(1.24, -0.45, "medido", transform=tr, fontsize=8.5, fontweight="bold", color=C_OBS)
    ax.plot([], [], "o", c=C_ANT, ms=8, label="antes (V2)")
    ax.plot([], [], "o", c=C_NOV, ms=8, label="agora")
    ax.fill_between([], [], color="#e3f0e3", label="faixa medida")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=3, bbox_to_anchor=(0.0, -0.2))
    ax.set_title("(a) Antes e agora, comparados com o medido (Amazônia central)", loc="left", fontsize=10.5)

    sub = gs[1].subgridspec(2, 1, hspace=0.12)
    x = np.arange(12)
    b = fig.add_subplot(sub[0])
    b.bar(x, M.chuva, color=C_P, width=0.75, label="chuva")
    b.plot(x, M.ET_V2, "o-", c=C_ANT, lw=2, ms=4, label="evapotranspiração antes")
    b.plot(x, M.ET_atual, "o-", c=C_NOV, lw=2.2, ms=4, label="evapotranspiração agora")
    b.set_ylim(0, 10.5)
    b.set_ylabel("mm dia⁻¹")
    b.set_xticks(x)
    b.set_xticklabels([])
    b.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
    b.set_title("(b) Ciclo anual: chuva e evapotranspiração (em cima)\n     e água no solo (embaixo)", loc="left",
                fontsize=10.5)
    wfc, wwp = solo.capacidade_campo_mm, solo.ponto_murcha_mm
    c = fig.add_subplot(sub[1])
    c.axhspan(wwp, wfc, color="#f1f6ec", zorder=0)
    c.axhline(wfc, c=C_NOV, ls=":", lw=1)
    c.text(11.45, wfc + 4, "capacidade de campo", ha="right", fontsize=7.5, color=C_NOV)
    c.axhline(wwp, c="#b03a3a", ls=":", lw=1)
    c.text(11.45, wwp + 4, "ponto de murcha", ha="right", fontsize=7.5, color="#b03a3a")
    c.plot(x, M.agua_V2, "o-", c=C_ANT, lw=2, ms=4, label="antes")
    c.plot(x, M.agua_atual, "o-", c=C_NOV, lw=2.2, ms=4, label="agora")
    c.set_ylim(170, 385)
    c.set_ylabel("água no solo,\n0–100 cm (mm)")
    c.set_xticks(x)
    c.set_xticklabels(list("JFMAMJJASOND"))
    c.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2, bbox_to_anchor=(0.0, 0.12))

    fig.text(0.01, -0.08,
             "Célula 186-239, média 1979–2016; antes = ALLOM_NUTRI_V2, agora = TEST_WATER4. Medidos (tabelas/observacoes.csv): "
             "evapotranspiração, fração da chuva e resistência do dossel, Malhi et al. 2002 (faixa da evapotranspiração: "
             "Fisher et al. 2009);\nGPP, Tomasella et al. 2020; Ra, NPP e eficiência no uso do carbono, Malhi et al. 2009. "
             "¹ Picos do ciclo diário medido (~meio-dia); o modelo dá a média de 24 h ponderada pela ocupação.",
             fontsize=7.5, color="#666", va="top")
    fig.savefig(os.path.join(FIG, "fig9_sintese_eixo_carbono_agua.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(ESTILO)
    print("fig9_sintese_eixo_carbono_agua.png")


# ---------------------------------------------------------------------------------
def fig10():
    """Pendente: o fator de estresse hídrico f5 (não alterado) em função da água do solo (bancada)."""
    ant, _ = bancada()
    solo = tabela("solo_celulas.csv", index_col=0, dtype={"celula": str}).loc["186-239"]
    wmax, wfc, wwp = solo.saturacao_mm, solo.capacidade_campo_mm, solo.ponto_murcha_mm
    s = ant[(ant.co2 == 338) & (ant.g1 == 4) & (ant.vpd.round(2) == 0.77)].sort_values("w_mm")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(s.w_mm / wmax, s.f5, "o-", c=C_ANT)
    ax.axvline(wwp / wmax, ls="--", c="r", lw=1)
    ax.text(wwp / wmax + 0.01, 0.3, "ponto de\nmurcha", color="r", fontsize=8)
    ax.axvline(wfc / wmax, ls="--", c="b", lw=1)
    ax.text(wfc / wmax - 0.17, 0.3, "capacidade\nde campo", color="b", fontsize=8)
    ax.set_xlabel("água do solo / saturação")
    ax.set_ylabel("f5 (estresse hídrico)")
    ax.set_ylim(0, 1.05)
    ax.set_title("f5 atual (não alterado): só age abaixo do ponto de murcha", fontsize=9)
    salva(fig, "fig10_f5_agua_solo.png")


FIGURAS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8, 9: fig9, 10: fig10}

if __name__ == "__main__":
    plt.rcParams.update(ESTILO)
    os.makedirs(FIG, exist_ok=True)
    escolha = [int(a) for a in sys.argv[1:]] or list(FIGURAS)
    for k in escolha:
        FIGURAS[k]()
