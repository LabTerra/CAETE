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

----------------------------------------------------------------------------
NOVO NESTA VERSÃO: conservação de carbono na célula de grade
----------------------------------------------------------------------------
Todo o C injetado nos recrutas (C_in) é RETIRADO das PLS pré-existentes (as vivas
em vp_lsid antes do recrutamento), de forma que o C total da vegetação não muda
no evento de recrutamento (o recrutamento só redistribui C entre as PLS).

1) Quanto cada PLS pré-existente doa (r_i) depende do seu coeficiente de
   ocupação (ocp_i, do dia anterior -> self.vp_ocp_allom) e do modo escolhido
   (argumento removal_mode):

     "ocp"      ->  r_i = C_in * ocp_i       / soma(ocp_j)
                    PLS dominantes doam mais.
     "inv_ocp"  ->  r_i = C_in * (1 - ocp_i) / soma(1 - ocp_j)
                    PLS menos dominantes doam mais.

   As somas são feitas sobre as PLS doadoras ELEGÍVEIS (ver item 3). A
   normalização garante soma(r_i) = C_in nos dois modos (sem ela, no modo
   "inv_ocp" a soma dos pesos seria N - 1 e não 1).

2) A retirada é PROPORCIONAL nos 5 compartimentos (cleaf, croot, csto, csap,
   cheart): cada pool da PLS i perde a mesma FRAÇÃO
        f_i = r_i / (cleaf_i + croot_i + csto_i + csap_i + cheart_i)
   isto é, pool_novo = pool * (1 - f_i). Assim a composição relativa da PLS se
   mantém e nenhum pool fica negativo. cwood é recalculado como csap + cheart
   (não é um compartimento à parte, senão a madeira seria descontada 2 vezes).

3) Verificação de sobrevivência: se, após a retirada, cleaf e/ou croot de uma
   PLS ficarem abaixo de CMIN (= 1.0e-6), essa PLS NÃO doa nada e a quantidade
   total C_in é redistribuída (pesos recalculados) entre as demais doadoras. O
   processo é repetido até nenhuma doadora violar o limite. Obs: a verificação só
   vale para o pool que estava >= CMIN antes da retirada (uma PLS viva só pelas
   folhas, com croot já abaixo de CMIN, não é excluída por causa da raiz).

4) Se nenhuma PLS pré-existente puder doar (ou não houver PLS vivas), o evento é
   CANCELADO: ninguém é recrutado e nenhum pool é alterado (o balanço continua
   fechado).

Os novos argumentos que possivelmente deverão ser guardados como atributos (self) do objeto
(class grd) são: seedbank, day_counter, rng (recomendado), removal_mode

Fluxo diário esperado em run_caete_allom:
    1. inflar            cleaf_allom[...] = vp_cleaf_allom ...   (já existe)
    2. recrutar          recruitment_step(pools, ...)            (este módulo)
                           2a. sortear recrutas
                           2b. calcular C_in e planejar a retirada (com verificação de CMIN)
                           2c. retirar C das pré-existentes
                           2d. injetar C nos recrutas
    3. Fortran           daily_budget_allom(...)
    4. novo vp_lsid      np.where(ocpavg > 0)[0]                 (já existe)
    5. desinflar         vp_cleaf_allom = dly_cleaf[vp_lsid] ... (já existe)

Uso dentro de run_caete_allom (os arrays inflados são modificados no lugar):
    pools = dict(cleaf=cleaf_allom, croot=croot_allom, csap=csap_allom,
                 cheart=cheart_allom, cwood=cwood_allom, csto=csto_allom)
    ocp_full = np.zeros(npls)                       # ocupação do dia anterior, inflada
    ocp_full[self.vp_lsid] = self.vp_ocp_allom
    self.recruit_day_counter, recruits, info = recruitment_step(
        pools, self.vp_lsid, self.seedbank_lsid, self.recruit_day_counter,
        self.pls_table, ocp_full, freq_days=30, recruit_frac=0.10,
        removal_mode="ocp", rng=self.rng)
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

# Compartimentos dos quais o C é retirado (cwood fica de fora: é csap + cheart)
DONOR_POOLS = ("cleaf", "croot", "csto", "csap", "cheart")

REMOVAL_MODES = ("ocp", "inv_ocp")


def should_recruit(day_counter, freq_days=30):
    """True se o evento de recrutamento deve ocorrer neste dia.

    day_counter : dias acumulados desde o último evento
    freq_days   : intervalo entre eventos (30 = mensal, 120 = a cada 4 meses, 365 = anual)

    Obs: ao invés de escolher um numero fixo de dias, o recrutamento pode ocorrer em todo início de
    mês (dia 1)
    """
    return day_counter >= freq_days


def select_recruits(seedbank, vp_lsid, recruit_frac=1, rng=None):
    """Sorteia, entre as PLS MORTAS do banco de plântulas, quais serão recrutadas.

    seedbank     : índices das PLS elegíveis (sobreviventes do mains spin-up 2)
    vp_lsid      : índices das PLS vivas agora (i.e., em qualquer passo de tempo)
    recruit_frac : fração (0-1) das PLS mortas do banco recrutada no evento
    rng          : np.random.Generator (passe uma semente para reprodutibilidade) -> da mesma forma que no R

    Retorna array ordenado de índices REAIS das PLS a serem recrutadas (0..npls-1).
    """
    rng = np.random.default_rng() if rng is None else rng #o próprio python cria uma semente ou eu preciso definir uma
    dead_pls = np.setdiff1d(seedbank, vp_lsid)            #i.e., PLS no banco e fora de vp_lsid
    n_recruit = int(round(recruit_frac * dead_pls.size))  #número de PLS a serem recrutadas
    if n_recruit < 1:
        return np.array([], dtype=int)                    #uma array vazia
    return np.sort(rng.choice(dead_pls, size=n_recruit, replace=False))


def seedling_carbon(recruits, pls_table, seedling_c=None):
    """C total (kgC/m2) que será injetado nos recrutas (C_in).

    recruits   : array retornado de select_recruits, com índices das PLS a serem recrutadas
    pls_table  : tabela de atributos das PLS, para verificação de PLS lenhosas
    seedling_c : dicionário com valores de C a serem injetados em cada compartimento da PLS

    Soma, para cada recruta, cleaf + croot + csto + (csap + cheart se lenhosa).
    É exatamente o que inject_seedlings vai adicionar aos pools.
    """
    sc = SEEDLING_C if seedling_c is None else seedling_c
    if recruits.size == 0:
        return 0.0
    woody = pls_table[6, recruits] > 0.0  #construção de máscara booleana para PLS lenhosas                           
    per_recruit = (sc["cleaf"] + sc["croot"] + sc["csto"]
                   + np.where(woody, sc["csap"] + sc["cheart"], 0.0))
    return float(per_recruit.sum())       #retorna o valor total de C de todos os recrutas


def withdrawal_weights(ocp, donors, removal_mode="ocp"):
    """Pesos normalizados (somam 1) de cada doadora na retirada de C.

    ocp          : array INFLADO (tamanho npls) com os coeficientes de ocupação
                   do dia anterior (zeros nas PLS mortas)
    donors       : índices REAIS das PLS doadoras elegíveis
    removal_mode : "ocp"     -> peso_i = ocp_i       / soma(ocp_j)    -> remoção de C diretamente proporcional à ocupação 
                    (deve ser normalizado para incluir somente os coeficientes de ocupação das PLS elegíveis e que, portanto, de fato se tornam doadoras)
                   "inv_ocp" -> peso_i = (1 - ocp_i) / soma(1 - ocp_j)-> remoção de C inversamente proporcional à ocupação

    Caso a soma dos pesos brutos seja 0 (ex.: modo "inv_ocp" com uma única doadora
    que tem ocp = 1, ou ocp ainda não calculado), os pesos são iguais entre as doadoras.
    """
    if removal_mode not in REMOVAL_MODES:
        raise ValueError(f"removal_mode deve ser um de {REMOVAL_MODES}, recebido: {removal_mode!r}")
    o = ocp[donors]                               #retorna somente o coeficientee de ocupação das PLS doadoras
    raw = o if removal_mode == "ocp" else 1.0 - o #retorna o próprio coeficiente de ocupação ou o seu inverso, dependendo do modo escolhido
    raw = np.clip(raw, 0.0, None)                 #proteção contra ocp levemente > 1 por arredondamento
    total = raw.sum()                             #soma dos coeficientes de ocupação das PLS doadoras
    if total <= 0.0:                              #caso especial que pode ocorrer em removal_mode é "inv_ocp", quando há só uma doadora
        return np.full(donors.size, 1.0 / donors.size)
    return raw / total                            #retorna array com coeficientes de ocupação já normalizados pela soma dos coeficientes dos PLS que podem de fato ser doadores


def plan_withdrawal(pools, donors, ocp, c_in, removal_mode="ocp", cmin=CMIN):
    """Calcula qual e quanto cada PLS pré-existente doa, sem alterar os pools.

    Implementa a verificação de sobrevivência: doadoras cujas folhas e/ou raízes
    ficariam abaixo de cmin são excluídas e C_in é redistribuído entre as demais,
    repetindo até não haver violações (o laço termina porque a cada volta pelo
    menos uma doadora sai do conjunto).

    Retorna um dict com
        donors   : índices das PLS que efetivamente doam
        frac     : fração f_i retirada de cada pool de cada doadora (alinhada a donors)
        removed  : C retirado de cada doadora, r_i (alinhado a donors)
        excluded : índices das PLS excluídas pela verificação de cmin
    ou None se não há doadora possível (evento deve ser cancelado).
    """
    eligible = np.asarray(donors, dtype=int).copy() #começa com todas as PLS pré-existentes (vp_lsid). O .copy() garante que, ao remover PLS de eligible, o vp_lsid original não seja afetado. 
    excluded = []
    if c_in <= 0.0:                                 #se não há C a retirar, a função devolve um plano vazio
        return dict(donors=np.array([], dtype=int), frac=np.array([]),
                    removed=np.array([]), excluded=np.array([], dtype=int))
    #o laço para testar quais PLS podem ser de fato doadoras
    while eligible.size > 0:
        w = withdrawal_weights(ocp, eligible, removal_mode)
        r = c_in * w                                                   #array com C que cada doara deve pagar durante a doação
        ctot = sum(pools[k][eligible] for k in DONOR_POOLS)            #array com C total (5 compartimentos - cleaf, croot, csto, csap, cheart) que cada doadora tem no passo de tempo atual
        with np.errstate(divide="ignore", invalid="ignore"):           #se uma PLS tiver ctot = 0, a divisão daria erro ou aviso. np.inf marca essa PLS como incapaz de pagar, e ela é excluída
            f = np.where(ctot > 0.0, r / ctot, np.inf)                 #array com fração do C que cada PLS vai perder (carbono total que a PLS deve pagar/C total atual da PLS)

        leaf, root = pools["cleaf"][eligible], pools["croot"][eligible]
        leaf_bad = (leaf >= cmin) & (leaf * (1.0 - f) < cmin)          #folha cruzaria o limiar de cmin
        root_bad = (root >= cmin) & (root * (1.0 - f) < cmin)          #raiz cruzaria o limiar de cmin
        bad = (f >= 1.0) | leaf_bad | root_bad                         #f >= 1 -> doaria todo (ou mais que todo) o seu C

        if not bad.any():
            return dict(donors=eligible, frac=f, removed=r,
                        excluded=np.array(sorted(excluded), dtype=int))
        excluded.extend(eligible[bad].tolist())                        #essas PLS não doam nada...
        eligible = eligible[~bad]                                      #...e C_in é redistribuído entre as restantes
    return None


def withdraw_carbon(pools, plan):
    """Aplica a retirada planejada nos arrays INFLADOS, IN-PLACE.

    Cada um dos 5 compartimentos da doadora i é multiplicado por (1 - f_i): a mesma
    proporção sai de todos os pools. cwood é recalculado como csap + cheart.
    """
    d, f = plan["donors"], plan["frac"]
    if d.size == 0:
        return pools
    for k in DONOR_POOLS:
        pools[k][d] *= (1.0 - f)
    pools["cwood"][d] = pools["csap"][d] + pools["cheart"][d]
    return pools


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
    pools["csap"][recruits] += np.where(woody, sc["csap"], 0.0)                              #adiciona 0.1 para PLS lenhosas e 0.0 para não lenhosas
    pools["cheart"][recruits] += np.where(woody, sc["cheart"], 0.0)                          #adiciona 0 para PLS lenhosas e não-lenhosas ("plântulas" ainda não têm heaart wood -> como acontece em init_caete_dyn)
    # madeira total acima do solo = alburno + cerne (mesma regra de init_caete_dyn)
    pools["cwood"][recruits] = pools["csap"][recruits] + pools["cheart"][recruits]
    return pools


def recruitment_step(pools, vp_lsid, seedbank, day_counter, pls_table, ocp,
                     freq_days=30, recruit_frac=0.5, seedling_c=None,
                     removal_mode="ocp", rng=None):
    """Chamar uma vez por dia, entre o inflar e a chamada ao Fortran.
    É a execução do processo de recrutamento propriamente dito

    ocp          : array INFLADO (tamanho npls) com a ocupação do dia anterior
                   (ocp[vp_lsid] = self.vp_ocp_allom; zeros nas mortas)
    removal_mode : "ocp" ou "inv_ocp" (ver docstring do módulo)

    Modifica os arrays de `pools` in-place quando há evento: primeiro retira C das
    PLS pré-existentes, depois injeta exatamente a mesma quantidade nos recrutas.

    Retorna (day_counter, recruits, info), em que info é um dict com
        c_in      : C injetado nos recrutas (kgC/m2)
        c_out     : C retirado das pré-existentes (kgC/m2) -> igual a c_in
        donors    : PLS que doaram C
        excluded  : PLS excluídas da doação pela verificação de CMIN
        cancelled : True se o evento foi cancelado por falta de doadoras
    """
    day_counter += 1                            #o contador aumenta em 1 toda a vez que a função é chamada
    recruits = np.array([], dtype=int)
    info = dict(c_in=0.0, c_out=0.0, donors=np.array([], dtype=int),
                excluded=np.array([], dtype=int), cancelled=False)

    if not should_recruit(day_counter, freq_days):  #senão, recruits volta como um array vazio e o contador continua a crescer partindo de onde estava
        return day_counter, recruits, info

    day_counter = 0
    recruits = select_recruits(seedbank, vp_lsid, recruit_frac, rng)
    if recruits.size == 0:
        return day_counter, recruits, info

    # (2b) quanto C entra e de quem ele sai
    c_in = seedling_carbon(recruits, pls_table, seedling_c)
    plan = plan_withdrawal(pools, vp_lsid, ocp, c_in, removal_mode)
    if plan is None:                            #nenhuma PLS pré-existente pode doar sem morrer -> cancela o evento
        info.update(c_in=0.0, cancelled=True)
        return day_counter, np.array([], dtype=int), info

    # (2c) retirada e (2d) injeção
    withdraw_carbon(pools, plan)
    inject_seedlings(pools, recruits, pls_table, seedling_c)

    info.update(c_in=c_in, c_out=float(plan["removed"].sum()),
                donors=plan["donors"], excluded=plan["excluded"])
    return day_counter, recruits, info


# --------------------------------------------------------------------------
# Teste com um "Fortran de mentira": verifica o alinhamento vp_lsid <-> pools
# e o balanço de C do recrutamento
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


def _veg_carbon(pools):
    """C total da vegetação na célula (soma dos 5 compartimentos em todas as PLS)."""
    return sum(pools[k].sum() for k in DONOR_POOLS)


def _check_balance(before, after, recruits, info, ocp, removal_mode, pls_table):
    """Verificações da retirada de C (balanço, proporcionalidade, pesos e CMIN)."""
    # 1) balanço: o C total da vegetação não muda no evento
    assert np.isclose(_veg_carbon(before), _veg_carbon(after)), "balanço de C não fechou"
    assert np.isclose(info["c_in"], info["c_out"]), "C_in != C_out"
    assert np.isclose(info["c_in"], seedling_carbon(recruits, pls_table)), "C_in != C das plântulas"

    donors, excluded = info["donors"], info["excluded"]
    # 2) proporcionalidade: em cada doadora, todos os pools (> 0) perdem a mesma fração
    for p in donors:
        ratios = [after[k][p] / before[k][p] for k in DONOR_POOLS if before[k][p] > 0.0]
        assert np.allclose(ratios, ratios[0]), f"retirada não proporcional na PLS {p}"
    # 3) pesos: o C retirado segue ocp (ou 1 - ocp), normalizado entre as doadoras efetivas
    removed = np.array([sum(before[k][p] - after[k][p] for k in DONOR_POOLS) for p in donors])
    expected = info["c_in"] * withdrawal_weights(ocp, donors, removal_mode)
    assert np.allclose(removed, expected), "C retirado não segue o coeficiente de ocupação"
    # 4) CMIN: nenhuma doadora teve folha/raiz levada para baixo de CMIN
    for p in donors:
        for k in ("cleaf", "croot"):
            assert not (before[k][p] >= CMIN and after[k][p] < CMIN), f"PLS {p}: {k} < CMIN"
    # 5) as excluídas não perderam nada; cwood = csap + cheart
    for p in excluded:
        assert all(after[k][p] == before[k][p] for k in POOL_NAMES), f"PLS excluída {p} foi alterada"
    assert np.allclose(after["cwood"], after["csap"] + after["cheart"])


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

    # ocupação do "dia anterior" (no modelo real: self.vp_ocp_allom, inflado)
    ocp, _ = _mock_daily_budget(_inflate(vp, vp_lsid, NPLS))
    assert np.isclose(ocp[vp_lsid].sum(), 1.0)

    # ---- dia SEM evento: nenhum pool muda ----
    pools = _inflate(vp, vp_lsid, NPLS)                                      #cria, para cada pool, um array de 100 zeros e preenche as posições de vp_lsid
    antes = {k: v.copy() for k, v in pools.items()}
    counter, rec, info = recruitment_step(pools, vp_lsid, seedbank, 5, pls_table, ocp,  #Com o contador em 5, a função o incrementa para 6, should_recruit(6, 30) dá False e nada é injetado.
                                          freq_days=30, rng=rng)
    assert counter == 6 and rec.size == 0                                    #confere se o contador aumento em mais um e se o array retornado é vazio
    assert all(np.array_equal(pools[k], antes[k]) for k in pools)            #confere se os pools inflados permanecem iguais, sem acréscimo de recrutas
    assert info["c_in"] == 0.0 and not info["cancelled"]

    # ---- dia COM evento, nos dois modos de retirada ----
    for mode in REMOVAL_MODES:
        pools = _inflate(vp, vp_lsid, NPLS)                      # 1) inflar (inflados novamente, já que isso é feito todo dia no código original)
        antes = {k: v.copy() for k, v in pools.items()}
        counter, recruits, info = recruitment_step(              # 2) recrutar: sorteia, retira C das pré-existentes e injeta nos recrutas
            pools, vp_lsid, seedbank, 29, pls_table, ocp,
            freq_days=30, recruit_frac=0.5, removal_mode=mode,
            rng=np.random.default_rng(7))                        #  mesma semente nos dois modos -> mesmos recrutas
        depois = {k: v.copy() for k, v in pools.items()}         #  estado que entra no "Fortran"
        _check_balance(antes, depois, recruits, info, ocp, mode, pls_table)

        ocpavg, dly = _mock_daily_budget(pools)                  # 3) "Fortran"
        new_lsid = np.where(ocpavg > 0.0)[0]                     # 4) novo vp_lsid
        new = {k: dly[k][new_lsid] for k in POOL_NAMES}          # 5) desinflar

        # ---- VERIFICAÇÕES DE ALINHAMENTO ----
        assert counter == 0                                                                              #o contador foi zerado após o evento de recrutamento
        assert np.all(np.isin(recruits, seedbank)), "recruta fora do banco"                              #recrutas pertencem ao banco
        assert not np.any(np.isin(recruits, vp_lsid)), "recruta já estava viva"                          #nenhum recruta já está vivo
        assert np.array_equal(new_lsid, np.union1d(vp_lsid, recruits)), "vp_lsid != vivas + recrutas"    #nenhuma pré-existente morreu por causa da retirada
        assert np.all(np.diff(new_lsid) > 0), "vp_lsid fora de ordem"                                    #os índices de new_lsid estão em ordem crescente
        assert all(new[k].size == new_lsid.size for k in POOL_NAMES)                                     #todos os arrays desinflados têm o mesmo tamanho que new_lsid

        sc = SEEDLING_C
        for k_pos, p in enumerate(new_lsid):                     # alinhamento posição <-> PLS
            if p in recruits:                                    #se p é recruta, os valores de C esperados ao fim do dia vem de SEEDLING_C com sapwood só se for lenhosa e, depois, x 1.05
                w = pls_table[6, p] > 0.0
                esperado = {
                    "cleaf": sc["cleaf"], "croot": sc["croot"], "csto": sc["csto"],
                    "csap": sc["csap"] if w else 0.0,
                    "cheart": sc["cheart"] if w else 0.0,
                }
                esperado["cwood"] = esperado["csap"] + esperado["cheart"]
                esperado = {k: v * 1.05 for k, v in esperado.items()}
            else:                                                #se p já estava viva, o esperado é o valor APÓS A RETIRADA x 1.05
                esperado = {k: depois[k][p] * 1.05 for k in POOL_NAMES}
            for k in POOL_NAMES:
                assert np.isclose(new[k][k_pos], esperado[k]), f"desalinhado: PLS {p}, pool {k}"

        # invariante: cwood = csap + cheart em todas as PLS vivas
        assert np.allclose(new["cwood"], new["csap"] + new["cheart"])
        # nsto/psto: recrutas em 0; PLS antigas continuam alinhadas e intocadas pela retirada
        for name in ("nsto", "psto"):
            compact = dly[name][new_lsid]
            for k_pos, p in enumerate(new_lsid):
                if p in recruits:
                    assert compact[k_pos] == 0.0, f"{name} de recruta deveria ser 0"
                else:
                    pos = np.searchsorted(vp_lsid, p)
                    assert np.isclose(compact[k_pos], vp[name][pos] * 1.05), f"{name} desalinhado: PLS {p}"
        # gramíneas recrutadas não têm madeira
        for p in recruits[pls_table[6, recruits] <= 0.0]:
            i = np.searchsorted(new_lsid, p)
            assert new["csap"][i] == 0.0 and new["cheart"][i] == 0.0 and new["cwood"][i] == 0.0

        n_w = int((pls_table[6, recruits] > 0.0).sum())
        print(f"[modo {mode!r}]")
        print(f"  banco de plântulas : {seedbank.size} PLS | vivas antes: {vp_lsid.size}")
        print(f"  recrutadas         : {recruits.size} ({n_w} lenhosas, {recruits.size - n_w} gramíneas)")
        print(f"  C injetado/retirado: {info['c_in']:.4f} / {info['c_out']:.4f} kgC/m2 "
              f"(C da vegetação: {_veg_carbon(antes):.4f} -> {_veg_carbon(depois):.4f})")
        top = vp_lsid[np.argsort(ocp[vp_lsid])[::-1][:3]]
        for p in top:
            perda = sum(antes[k][p] - depois[k][p] for k in DONOR_POOLS)
            print(f"    PLS {p:>2}: ocp={ocp[p]:.3f} -> doou {perda:.4f} kgC/m2")
        print(f"  vp_lsid depois     : {new_lsid.size} PLS")

    _demo_cmin_redistribution()
    _demo_cancel()
    print("OK: balanço de C fechado nos dois modos; retirada proporcional nos 5 compartimentos; "
          "verificação de CMIN com redistribuição; alinhamento vp_lsid <-> 6 pools preservado")


def _demo_cmin_redistribution():
    """Caso montado: a PLS 1 é a mais dominante, mas tem folhas no limite de CMIN.
    Sem a verificação ela ficaria com cleaf < CMIN; com a verificação ela é
    excluída e as PLS 0 e 2 doam todo o C_in."""
    npls = 5
    pls_table = np.zeros((20, npls)); pls_table[6, :] = 1.0          # todas lenhosas
    pools = {k: np.zeros(npls) for k in POOL_NAMES}
    vp_lsid = np.array([0, 1, 2])
    pools["cleaf"][vp_lsid] = [2.0, 1.00001e-6, 3.0]                 # PLS 1: folha logo acima de CMIN
    pools["croot"][vp_lsid] = [1.5, 6.0, 2.0]
    pools["csto"][vp_lsid] = [0.3, 0.5, 0.2]
    pools["csap"][vp_lsid] = [1.0, 4.0, 1.0]
    pools["cheart"][vp_lsid] = [2.0, 10.0, 3.0]
    pools["cwood"] = pools["csap"] + pools["cheart"]
    ocp, _ = _mock_daily_budget(pools)
    seedbank = np.array([0, 1, 2, 3, 4])

    for mode in REMOVAL_MODES:
        p = {k: v.copy() for k, v in pools.items()}
        antes = {k: v.copy() for k, v in p.items()}
        _, recruits, info = recruitment_step(p, vp_lsid, seedbank, 29, pls_table, ocp,
                                             recruit_frac=1.0, removal_mode=mode,
                                             rng=np.random.default_rng(0))
        assert np.array_equal(recruits, [3, 4])
        assert np.array_equal(info["excluded"], [1]), "PLS 1 deveria ser excluída da doação"
        assert np.array_equal(info["donors"], [0, 2])
        _check_balance(antes, p, recruits, info, ocp, mode, pls_table)
        print(f"[CMIN, modo {mode!r}] PLS 1 excluída; C_in = {info['c_in']:.2f} redistribuído "
              f"entre as PLS {info['donors'].tolist()}")


def _demo_cancel():
    """Caso montado: as pré-existentes não têm C suficiente para pagar as plântulas
    -> o evento é cancelado e nada muda."""
    npls = 4
    pls_table = np.zeros((20, npls))                                 # todas gramíneas
    pools = {k: np.zeros(npls) for k in POOL_NAMES}
    vp_lsid = np.array([0])
    pools["cleaf"][0], pools["croot"][0], pools["csto"][0] = 0.5, 0.4, 0.05
    ocp, _ = _mock_daily_budget(pools)
    antes = {k: v.copy() for k, v in pools.items()}
    counter, recruits, info = recruitment_step(pools, vp_lsid, np.arange(npls), 29, pls_table, ocp,
                                               recruit_frac=1.0, rng=np.random.default_rng(0))
    assert info["cancelled"] and recruits.size == 0 and counter == 0
    assert all(np.array_equal(pools[k], antes[k]) for k in pools)
    print("[cancelamento] C insuficiente nas pré-existentes -> evento cancelado, pools intactos")


if __name__ == "__main__":
    _demo()
