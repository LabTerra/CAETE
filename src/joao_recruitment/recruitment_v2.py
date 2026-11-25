"""
recruitment.py - protótipo do módulo de recrutamento de PLS (chuva de plântulas)

Versão para o esquema alométrico (run_caete_allom), com os SEIS pools de C:
cleaf, croot, csap, cheart, cwood, csto.
nsto, psto e uptk_costs NÃO são tocados: nas PLS mortas eles já valem 0 no array
inflado (o run_caete_allom os cria com np.zeros e só preenche as posições de
vp_lsid), que também é o estado inicial de todas as PLS em init_caete_dyn.

A injeção é feita nos arrays INFLADOS (tamanho npls), logo depois do loop de
inflar e ANTES de model_allom.daily_budget_allom(...).

Por que nesse ponto: o Fortran decide quem está vivo com pft_area_frac (é uma
subrotina de funcs.f90, que é chamada pela subrotina daily_budget_allom, pertencente
a budget_allom.f90), usando os pools de ENTRADA (viva = folha OU raiz >= cmin). 
Uma PLS recrutada com carbono >= cmin já é vista como viva no mesmo dia, e o
    vp_lsid = np.where(ocpavg > 0.0)[0]
que vem depois da chamada já a inclui. Não é preciso mexer em vp_lsid.

Os novos argumentos que possivelmente deverão ser guardados como atributos (self) do objeto
(class grd) são: seedbank, day_counter, rng (recomendado)

Fluxo diário esperado em run_caete_allom:
    1. inflar            cleaf_allom[...] = vp_cleaf_allom ...   (já existe)
    2. recrutar          recruitment_step(pools, ...)            (este módulo)
    3. Fortran           daily_budget_allom(...)
    4. novo vp_lsid      np.where(ocpavg > 0)[0]                 (já existe)
    5. desinflar         vp_cleaf_allom = dly_cleaf[vp_lsid] ... (já existe)

Uso dentro de run_caete_allom (os arrays inflados são modificados no lugar):
    pools = dict(cleaf=cleaf_allom, croot=croot_allom, csap=csap_allom,
                 cheart=cheart_allom, cwood=cwood_allom, csto=csto_allom)
    self.recruit_day_counter, recruits = recruitment_step(
        pools, self.vp_lsid, self.seedbank_lsid, self.recruit_day_counter,
        self.pls_table, freq_days=30, recruit_frac=0.10, rng=self.rng)
"""
import numpy as np

CMIN = 1.0e-6   # kgC/m2 - limiar de morte (global_par.cmin) -> aplicado para folhas E raízes

# Carbono inicial (kgC/m2) das PLS recrutadas: mesmos valores de init_caete_dyn.
# cwood NÃO entra aqui: é sempre csap + cheart.
SEEDLING_C = {       #definidos com base nos valores em init_caete_dyn
    "cleaf": 1.0,
    "croot": 0.8,
    "csto": 0.1,
    "csap": 0.1,     # só lenhosas (pls_table[6] > 0); gramíneas recebem 0
    "cheart": 0.0,   # plântula não tem cerne
}
POOL_NAMES = ("cleaf", "croot", "csap", "cheart", "cwood", "csto")


def should_recruit(day_counter, freq_days=30):
    """True se o evento de recrutamento deve ocorrer neste dia.

    day_counter : dias acumulados desde o último evento
    freq_days   : intervalo entre eventos (30 = mensal, 120 = a cada 4 meses, 365 = anual)

    Obs: ao invés de escolher um numero fixo de dias, o recrutamento pode ocorrer em todo início de 
    mês (dia 1)
    """
    return day_counter >= freq_days


def select_recruits(seedbank, vp_lsid, recruit_frac=0.5, rng=None):
    """Sorteia, entre as PLS MORTAS do banco de plântulas, quais serão recrutadas.

    seedbank     : índices das PLS elegíveis (sobreviventes do mains spin-up 2)
    vp_lsid      : índices das PLS vivas agora (i.e., em qualquer passo de tempo)
    recruit_frac : fração (0-1) das PLS mortas do banco recrutada no evento
    rng          : np.random.Generator (passe uma semente para reprodutibilidade) -> da mesma forma que no R

    Retorna array ordenado de índices REAIS das PLS a serem recrutadas (0..npls-1).

    Obs: talvez seja melhor tentar recrutar todos os PLS em cada evento (deixar a sazonalidade atuar).
    Para tanto, recruit_frac=1
    """
    rng = np.random.default_rng() if rng is None else rng #o próprio python cria uma semente ou eu preciso definir uma
    dead_pls = np.setdiff1d(seedbank, vp_lsid)            #i.e., PLS no banco e fora de vp_lsid
    n_recruit = int(round(recruit_frac * dead_pls.size))  #número de PLS a serem recrutadas
    if n_recruit < 1:
        return np.array([], dtype=int)                    #uma array vazia
    return np.sort(rng.choice(dead_pls, size=n_recruit, replace=False))


def inject_seedlings(pools, recruits, pls_table, seedling_c=None):
    """Injeta carbono de plântula nos arrays INFLADOS (tamanho npls), IN-PLACE.

    pools      : dict com os arrays inflados {"cleaf", "croot", "csap", "cheart",
                 "cwood", "csto"}; os arrays do dict são os mesmos objetos usados
                 no run_caete_allom, então a modificação chega ao Fortran -> são
                 as arrays que entram em daily_budget_allom
    recruits   : índices REAIS das PLS recrutadas (saída de select_recruits)
    pls_table  : matriz de traits (ntraits, npls); a linha 6 diz se a PLS é lenhosa (i.e., valores > 0)
    seedling_c : dict com o C inicial (padrão: SEEDLING_C) -> mas eu posso passar outro dicionário com outros valores de C

    Os recrutas são PLS mortas, então suas posições estão em zero no array inflado
    e somar equivale a atribuir. O += continua correto se PLS já vivas também receberem propágulos.
    """
    sc = SEEDLING_C if seedling_c is None else seedling_c
    assert sc["cleaf"] >= CMIN or sc["croot"] >= CMIN, "plântula abaixo de cmin morreria"  #verificando se os valores iniciais de C escolhidos para folhas e raízes são maiores do que a quantidade mínima para sobrevivência
                                                                                           #se estiverem abaixo, a execução é encerrada
    if recruits.size == 0:
        return pools

    # máscara de lenhosas (máscara booleana -> True/False), ALINHADA a `recruits` (tamanho = recruits.size). E não vp_lsid.size, como ocorre durante a repopulação do spin-up
    woody = pls_table[6, recruits] > 0.0

    pools["cleaf"][recruits] += sc["cleaf"]
    pools["croot"][recruits] += sc["croot"]
    pools["csto"][recruits] += sc["csto"]
    pools["csap"][recruits] += np.where(woody, sc["csap"], 0.0)     #adiciona 0.1 para PLS lenhosas e 0.0 para não lenhosas
    pools["cheart"][recruits] += np.where(woody, sc["cheart"], 0.0) #adiciona 0 para PLS lenhosas e não-lenhosas ("plântulas" ainda não têm heaart wood -> como acontece em init_caete_dyn)
    # madeira total acima do solo = alburno + cerne (mesma regra de init_caete_dyn)
    pools["cwood"][recruits] = pools["csap"][recruits] + pools["cheart"][recruits]
    return pools


def recruitment_step(pools, vp_lsid, seedbank, day_counter, pls_table,
                     freq_days=30, recruit_frac=0.5, seedling_c=None, rng=None):
    """Chamar uma vez por dia, entre o inflar e a chamada ao Fortran.
    É a execução do processo de recrutamento propriamente dito

    Modifica os arrays de `pools` in-place quando há evento.
    Retorna (day_counter, recruits).

    Obs: o principal problema dessa forma de injeção de C nos pools dos PLS recrutados é que ela fere 
    o princiípio de conservação de massa do modelo (cada célula de grade funciona como um sistema fechado).
    Sendo assim, todo C que eu adicionar para os PLS recrutados, a mesma quantidade deve sair dos PLS que já 
    estavam lá estabelecidos -> para isso, mexer na em Fortran no budget_allom.f90 (e potencialmente outros 
    módulos)
    """
    day_counter += 1                            #o contador aumenta em 1 toda a vez que a função é chamada
    recruits = np.array([], dtype=int)
    if should_recruit(day_counter, freq_days):  #as duas outras funções são executadas se for o momento de ter recrutamento, senão, recruits volta como um array  vazio e o contador continua a crescer partindo de onde estava
        recruits = select_recruits(seedbank, vp_lsid, recruit_frac, rng)
        inject_seedlings(pools, recruits, pls_table, seedling_c)
        day_counter = 0
    return day_counter, recruits


# --------------------------------------------------------------------------
# Teste com um "Fortran de mentira": verifica o alinhamento vp_lsid <-> pools
# --------------------------------------------------------------------------
def _inflate(vp, vp_lsid, npls):
    """Compacto -> inflado (tamanho npls, zeros nas PLS mortas).
    Cria os arrays inflados, incialmente zerados em todas as posiçoes, e 
    depois atribui os valores de C para as PLS já vivas, através de vp_lsid
    Esses arrays entram em daily_budget_allom

    vp: dicionário com os nomes dos pools COMPACTOS e arrays inflados com valores de C para cada PLS
    """
    pools = {}
    for name, arr in vp.items():
        full = np.zeros(npls)
        full[vp_lsid] = arr
        pools[name] = full
    return pools


def _mock_daily_budget(pools):
    """Imita a subrotina do Fortran daily_budget_allom: viva = folha OU raiz de ENTRADA >= CMIN;
    ocpavg = fração de biomassa (folha + raiz + madeira); dly = biomassa dee fim de dia,
    considerando o mesmo acréscimo de biomassa para todas as PLS (+5%).
     """
    alive = (pools["cleaf"] >= CMIN) | (pools["croot"] >= CMIN)                       #cria uma máscara booleana de tamanho npls, ja que sua criacao é baseada nos arrays inflados -> pools. A | é a versão de OU do numpy
    biomass = np.where(alive, pools["cleaf"] + pools["croot"] + pools["cwood"], 0.0)  #array de tamanho npls e com valores de biomassa total para os PLS vivos como sendo a soma de todos os pools. PLS mortos recebem 0
    total = biomass.sum()                                                             #soma todos os elementos da array biomass
    ocpavg = biomass / total if total > 0.0 else np.zeros_like(biomass)               #cálculo dos coeficientes de ocupação dos PLS, caso a biomassa na célula de grade seja > 0
    dly = {name: np.where(alive, arr * 1.05, 0.0) for name, arr in pools.items()}     #name é o nome do pool e array é a sequencia de valores
    return ocpavg, dly

'''
Para ficar mais fácil de eu entender, a criação do dicionário dly poderia ser feita da seguinte maneira
dly = {}
for name, arr in pools.items():
    dly[name] = np.where(alive, arr * 1.05, 0.0)
'''


def _demo():
    rng = np.random.default_rng(42)  #setando uma semente para reprodutibilidade
    NPLS = 100 
    pls_table = np.zeros((20, NPLS)) #criando uma tabela de mentira, com 20 traits e o número de PLS escolhidos
    is_w = rng.random(NPLS) < 0.6    #gera 100 números em [0, 1) e compara com 0.6, resultando numa máscara booleana (um pouco menos que 60% das PLS são lenhosas)
    pls_table[6, :] = np.where(is_w, rng.uniform(0.5, 2.0, NPLS), 0.0)   # põe um valor positivo aleatorio entre 0.5 e 2.0 para lenhosas e 0 = gramínea

    seedbank = np.sort(rng.choice(NPLS, 50, replace=False))     #supondo que 50 PLS sobreviveram ao spiup
    vp_lsid = np.sort(rng.choice(seedbank, 12, replace=False))  #supondo que há 12 PLS vivas num passo de tempo qualquer
    n = vp_lsid.size
    woody_live = pls_table[6, vp_lsid] > 0.0                    #máscara booleana para identificar PLS vivos que são lenhosos

    # estado compacto das PLS vivas (consistente: gramíneas sem madeira)
    vp = {                                                              #É um dicionário com um array de tamanho 12 por pool, correspondente às 12 PLS vivas, na ordem de vp_lsid
        "cleaf": rng.uniform(0.1, 10, n),
        "croot": rng.uniform(0.1, 8, n),
        "csto": rng.uniform(0.05, 1, n),
        "csap": np.where(woody_live, rng.uniform(0.1, 5, n), 0.0),
        "cheart": np.where(woody_live, rng.uniform(0.0, 20, n), 0.0),
    }
    vp["cwood"] = vp["csap"] + vp["cheart"]                             #adiciona mais um pool ao dicionário já criado
    # reservas de N e P (g/m2): o recrutamento NÃO as toca (recrutas ficam em 0)
    vp["nsto"] = rng.uniform(0.1, 3.0, n)                               #nsto e psto tão com valores positivos de propósito só pra testar se o recrutamento vai alterar seus valores
    vp["psto"] = rng.uniform(0.05, 1.0, n)

    # ---- dia SEM evento: nenhum pool muda ----
    pools = _inflate(vp, vp_lsid, NPLS)                                      #cria, para cada pool, um array de 100 zeros e preenche as posições de vp_lsid
    antes = {k: v.copy() for k, v in pools.items()}
    counter, rec = recruitment_step(pools, vp_lsid, seedbank, 5, pls_table,  #Com o contador em 5, a função o incrementa para 6, should_recruit(6, 30) dá False e nada é injetado.
                                    freq_days=30, rng=rng)
    assert counter == 6 and rec.size == 0                                    #confere se o contador aumento em mais um e se o array retornado é vazio
    assert all(np.array_equal(pools[k], antes[k]) for k in pools)            #confere se os pools inflados permanecem iguais, sem acréscimo de recrutas

    # ---- dia COM evento ----
    pools = _inflate(vp, vp_lsid, NPLS)                      # 1) inflar (inflados novamente, já que isso é feito todo dia no código original)
    counter, recruits = recruitment_step(                    # 2) recrutar (com o contador em 29, o incremento leva a 30 e o recrutamento ocorre -> should recruit dá True)
        pools, vp_lsid, seedbank, 29, pls_table,             # select_recruits calcula as 38 PLS mortas do banco (np.setdiff1d), sorteia round(0.5 × 38) = 19 sem repetição e ordena.                                              
        freq_days=30, recruit_frac=0.5, rng=rng)             ##inject_seedlings soma os valores de SEEDLING_C nas posições dos recrutas, e o contador volta a 0.
    ocpavg, dly = _mock_daily_budget(pools)                  # 3) "Fortran": calcula quem está vivo (folha ou raiz ≥ CMIN), a ocupação ocpavg e os pools de fim de dia dly (+5% nas vivas, zero nas mortas). Os recrutas entram como vivos, porque cleaf = 1.0 ≥ CMIN
    new_lsid = np.where(ocpavg > 0.0)[0]                     # 4) novo vp_lsid (no código real, vp_lsid nunca muda de nome: ele é sobrescrito)
    new = {k: dly[k][new_lsid] for k in POOL_NAMES}          # 5) desinflar (novo dicionário com arrays desinflados com somente as posições vivas (o “coletar”), para os seis pools de C. nsto e psto são desinflados mais adiante, dentro de suas verificações.) 

    # ---- VERIFICAÇÕES ----
    assert counter == 0                                                                              #o contador foi zerado após o evento de recrutamento
    assert np.all(np.isin(recruits, seedbank)), "recruta fora do banco"                              #recrutas pertencem ao banco
    assert not np.any(np.isin(recruits, vp_lsid)), "recruta já estava viva"                          #nenhum recruta já está vivo
    assert np.array_equal(new_lsid, np.union1d(vp_lsid, recruits)), "vp_lsid != vivas + recrutas"    #o conjunto das PLS vivas após o recrutamento é a união das "vivas antigas" e as recrutas
    assert np.all(np.diff(new_lsid) > 0), "vp_lsid fora de ordem"                                    #os índices de new_lsid estão em ordem crescente
    assert all(new[k].size == new_lsid.size for k in POOL_NAMES)                                     #todos os arrays desinflados têm o mesmo tamanho que new_lsid

    sc = SEEDLING_C
    for k_pos, p in enumerate(new_lsid):                     # alinhamento posição <-> PLS
        if p in recruits:                                    #se p é recruta, os valores de C esperados ao fim do dia vem de SEEDLING_C com sapwood só se for lenhosa e, depois, x 1.05
            w = pls_table[6, p] > 0.0                        #se p já estava viva, o esperado é o valor original x 1.05
            esperado = {
                "cleaf": sc["cleaf"], "croot": sc["croot"], "csto": sc["csto"],
                "csap": sc["csap"] if w else 0.0,
                "cheart": sc["cheart"] if w else 0.0,
            }
            esperado["cwood"] = esperado["csap"] + esperado["cheart"]
            esperado = {k: v * 1.05 for k, v in esperado.items()}
        else:
            pos = np.searchsorted(vp_lsid, p)
            esperado = {k: vp[k][pos] * 1.05 for k in POOL_NAMES}
        for k in POOL_NAMES:
            assert np.isclose(new[k][k_pos], esperado[k]), f"desalinhado: PLS {p}, pool {k}"

    # invariante: cwood = csap + cheart em todas as PLS vivas
    assert np.allclose(new["cwood"], new["csap"] + new["cheart"])
    # nsto/psto: recrutas em 0; PLS antigas continuam alinhadas
    for name in ("nsto", "psto"):
        compact = dly[name][new_lsid]
        for k_pos, p in enumerate(new_lsid):
            if p in recruits:
                assert compact[k_pos] == 0.0, f"{name} de recruta deveria ser 0"
            else:
                pos = np.searchsorted(vp_lsid, p)
                assert np.isclose(compact[k_pos], vp[name][pos] * 1.05), f"{name} desalinhado: PLS {p}"
    # gramíneas recrutadas não têm madeira
    grass_rec = recruits[pls_table[6, recruits] <= 0.0]
    for p in grass_rec:
        i = np.searchsorted(new_lsid, p)
        assert new["csap"][i] == 0.0 and new["cheart"][i] == 0.0 and new["cwood"][i] == 0.0

    n_w = int((pls_table[6, recruits] > 0.0).sum())
    print(f"banco de plântulas : {seedbank.size} PLS | vivas antes: {vp_lsid.size}")
    print(f"recrutadas         : {recruits.size} ({n_w} lenhosas, {recruits.size - n_w} gramíneas)")
    print(f"vp_lsid depois     : {new_lsid.size} PLS")
    i = np.searchsorted(new_lsid, recruits[:2])
    for p, k_pos in zip(recruits[:2], i):
        linha = ", ".join(f"{k}={new[k][k_pos]:.3f}" for k in POOL_NAMES)
        print(f"  PLS {p:>2} (recruta, lenhosa={pls_table[6, p] > 0}): {linha}")
    print("OK: alinhamento vp_lsid <-> 6 pools preservado; cwood = csap + cheart; nsto/psto = 0 nos recrutas")


if __name__ == "__main__":
    _demo()
