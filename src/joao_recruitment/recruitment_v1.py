"""
recruitment.py - protótipo do módulo de recrutamento de PLS (chuva de plântulas)

Versão para o esquema alométrico (run_caete_allom), trabalhando com UM pool de C
(folha). A injeção é feita nos arrays INFLADOS (tamanho npls), logo depois do
loop de inflar e ANTES de model_allom.daily_budget_allom(...).

Por que nesse ponto: o Fortran decide quem está vivo com pft_area_frac, usando os
pools de ENTRADA (viva = folha OU raiz >= cmin). Uma PLS recrutada com carbono
>= cmin já é vista como viva no mesmo dia, e o
    vp_lsid = np.where(ocpavg > 0.0)[0]
que vem depois da chamada já a inclui. Não é preciso mexer em vp_lsid.

Fluxo diário esperado em run_caete_allom:
    1. inflar            cleaf_allom[vp_lsid] = vp_cleaf_allom   (já existe)
    2. recrutar          recruitment_step(cleaf_allom, ...)      (este módulo)
    3. Fortran           daily_budget_allom(...)
    4. novo vp_lsid      np.where(ocpavg > 0)[0]                 (já existe)
    5. desinflar         vp_cleaf_allom = dly_cleaf[vp_lsid]     (já existe)
"""
import numpy as np

CMIN = 1.0e-6   # kgC/m2 - limiar de morte (global_par.cmin); seedling_c deve ser >= CMIN


def should_recruit(day_counter, freq_days=30):
    """True se o evento de recrutamento deve ocorrer neste dia.

    day_counter : dias acumulados desde o último evento
    freq_days   : intervalo entre eventos (30 = mensal, 120 = a cada 4 meses, 365 = anual)
    """
    return day_counter >= freq_days


def select_recruits(seedbank, vp_lsid, recruit_frac=0.5, rng=None):
    """Sorteia, entre as PLS MORTAS do banco de plântulas, quais serão recrutadas.

    seedbank     : índices das PLS elegíveis (sobreviventes do spin-up)
    vp_lsid      : índices das PLS vivas agora
    recruit_frac : fração (0-1) das PLS mortas do banco recrutada no evento
    rng          : np.random.Generator (passe uma semente para reprodutibilidade)

    Retorna array ordenado de índices REAIS das PLS (0..npls-1).
    """
    rng = np.random.default_rng() if rng is None else rng
    dead_pls = np.setdiff1d(seedbank, vp_lsid)           # no banco e fora de vp_lsid
    n_recruit = int(round(recruit_frac * dead_pls.size))
    if n_recruit < 1:
        return np.array([], dtype=int)
    return np.sort(rng.choice(dead_pls, size=n_recruit, replace=False))


def inject_seedlings(cleaf_full, recruits, seedling_c):
    """Injeta carbono de plântula no array INFLADO (tamanho npls), IN-PLACE.

    Como os recrutas são PLS mortas, suas posições estão em zero no array inflado,
    então somar equivale a atribuir. Se no futuro PLS já vivas também puderem
    receber propágulos, o += continua correto (acréscimo à biomassa existente).
    """
    cleaf_full[recruits] += seedling_c
    return cleaf_full


def recruitment_step(cleaf_full, vp_lsid, seedbank, day_counter,
                     freq_days=30, recruit_frac=0.5, seedling_c=0.01, rng=None):
    """Chamar uma vez por dia, entre o inflar e a chamada ao Fortran.

    Modifica cleaf_full in-place quando há evento. Retorna (day_counter, recruits).
    """
    day_counter += 1
    recruits = np.array([], dtype=int)
    if should_recruit(day_counter, freq_days):
        recruits = select_recruits(seedbank, vp_lsid, recruit_frac, rng)
        inject_seedlings(cleaf_full, recruits, seedling_c)
        day_counter = 0
    return day_counter, recruits


# --------------------------------------------------------------------------
# Teste com um "Fortran de mentira": verifica o alinhamento vp_lsid <-> vp_cleaf
# --------------------------------------------------------------------------
def _mock_daily_budget(cleaf_full):
    """Imita o contrato do Fortran: viva = carbono de ENTRADA >= CMIN;
    ocpavg = fração de biomassa; dly_cleaf = carbono de fim de dia (cresce 5%)."""
    alive = cleaf_full >= CMIN
    total = cleaf_full[alive].sum()
    ocpavg = np.zeros_like(cleaf_full)
    if total > 0.0:
        ocpavg[alive] = cleaf_full[alive] / total
    dly_cleaf = np.where(alive, cleaf_full * 1.05, 0.0)
    return ocpavg, dly_cleaf


def _demo():
    rng = np.random.default_rng(42)
    NPLS = 100
    seedbank = np.sort(rng.choice(NPLS, 50, replace=False))
    vp_lsid = np.sort(rng.choice(seedbank, 12, replace=False))
    vp_cleaf = rng.uniform(0.1, 10, vp_lsid.size)
    seedling_c = 0.01

    # ---- dia SEM evento (contador longe de 30): nada muda ----
    cleaf = np.zeros(NPLS)
    cleaf[vp_lsid] = vp_cleaf
    antes = cleaf.copy()
    counter, rec = recruitment_step(cleaf, vp_lsid, seedbank, 5, freq_days=30, rng=rng)
    assert counter == 6 and rec.size == 0 and np.array_equal(cleaf, antes)

    # ---- dia COM evento ----
    cleaf = np.zeros(NPLS)                      # 1) inflar
    cleaf[vp_lsid] = vp_cleaf
    counter, recruits = recruitment_step(       # 2) recrutar
        cleaf, vp_lsid, seedbank, 29, freq_days=30, recruit_frac=0.5,
        seedling_c=seedling_c, rng=rng)
    ocpavg, dly_cleaf = _mock_daily_budget(cleaf)   # 3) "Fortran"
    new_lsid = np.where(ocpavg > 0.0)[0]            # 4) novo vp_lsid
    new_cleaf = dly_cleaf[new_lsid]                 # 5) desinflar

    # ---- verificações ----
    assert counter == 0
    assert np.all(np.isin(recruits, seedbank)), "recruta fora do banco"
    assert not np.any(np.isin(recruits, vp_lsid)), "recruta já estava viva"
    assert np.array_equal(new_lsid, np.union1d(vp_lsid, recruits)), "vp_lsid != vivas + recrutas"
    assert np.all(np.diff(new_lsid) > 0), "vp_lsid fora de ordem"
    assert new_lsid.size == new_cleaf.size
    for k, p in enumerate(new_lsid):                # alinhamento posição <-> PLS
        if p in recruits:
            esperado = seedling_c * 1.05
        else:
            esperado = vp_cleaf[np.searchsorted(vp_lsid, p)] * 1.05
        assert np.isclose(new_cleaf[k], esperado), f"desalinhado na PLS {p}"

    print("banco de plântulas  :", seedbank.size, "PLS")
    print("vivas antes         :", vp_lsid)
    print("recrutadas          :", recruits)
    print("vp_lsid depois      :", new_lsid)
    print("vp_cleaf depois     :", np.round(new_cleaf, 3))
    print("OK: alinhamento vp_lsid <-> vp_cleaf preservado")


if __name__ == "__main__":
    _demo()
