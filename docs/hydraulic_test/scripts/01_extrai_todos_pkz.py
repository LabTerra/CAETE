#!/usr/bin/env python3
"""Exporta TODAS as variáveis dos spin*.pkz históricos do CAETÊ para CSV.

Pasta esperada: CAETE/docs/hydraulic_test/scripts/01_extrai_todos_pkz.py
Entrada padrão: CAETE/outputs/allomnutri_hydraulics_3k_PKLS/gridcell186-239/
Saída padrão:   CAETE/docs/hydraulic_test/tabelas/

Executar de qualquer pasta:
    /home/gabrielasophia/anaconda3/bin/python docs/hydraulic_test/scripts/01_extrai_todos_pkz.py

Opcional: --entrada CAMINHO --saida CAMINHO --inicio 1979 --fim 2016

Arquivos de saída:
- diario.csv: variáveis de 1 dimensão e de 2 dimensões pequenas, com uma linha por dia.
- area.csv: uma linha por dia, uma coluna por índice de PLS (3000 colunas).
- lim_status_1.csv, lim_status_2.csv, lim_status_3.csv: matrizes por PLS.
- u_strat_1.csv, u_strat_2.csv: matrizes por PLS.
- metadados.csv: variáveis escalares de cada spin.
- catalogo_variaveis.csv: nome, formato e CSV correspondente.

Os índices pls_0000 ... pls_2999 representam a ordem dos eixos nos arquivos.
Nenhum dado é agregado ou transformado em média.
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from joblib import load

BASE = Path(__file__).resolve().parents[3]
DEFAULT_ENTRADA = BASE / "outputs" / "allomnutri_hydraulics_3k_PKLS" / "gridcell186-239"
DEFAULT_SAIDA = BASE / "docs" / "hydraulic_test" / "tabelas"
DATA_ORIGEM = date(1860, 1, 1)
PADRAO = re.compile(r"spin(\d+)\.pkz$")


def abrir_csv(caminho: Path, cabecalho: list[str], handles: dict):
    chave = caminho.name
    if chave not in handles:
        arq = caminho.open("w", newline="", encoding="utf-8")
        writer = csv.writer(arq)
        writer.writerow(cabecalho)
        handles[chave] = (arq, writer)
    return handles[chave][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=DEFAULT_ENTRADA)
    parser.add_argument("--saida", type=Path, default=DEFAULT_SAIDA)
    parser.add_argument("--inicio", type=int, default=1979)
    parser.add_argument("--fim", type=int, default=2016)
    args = parser.parse_args()
    if args.inicio > args.fim:
        parser.error("--inicio deve ser menor ou igual a --fim")
    if not args.entrada.is_dir():
        parser.error(f"Pasta de entrada não encontrada: {args.entrada}")

    arquivos = []
    for caminho in args.entrada.iterdir():
        match = PADRAO.fullmatch(caminho.name)
        if match:
            numero = int(match.group(1))
            if numero >= 36:  # pula spinup (spin01 a spin35)
                arquivos.append((numero, caminho))
    arquivos.sort()
    if not arquivos:
        parser.error("Nenhum arquivo spin36.pkz ou posterior encontrado")

    args.saida.mkdir(parents=True, exist_ok=True)
    handles = {}
    catalogo = {}
    datas_vistas = set()
    cont = 0
    try:
        for _, caminho in arquivos:
            d = load(caminho)
            sind = int(d["sind"])
            n = int(np.asarray(d["psisoil"]).size)
            datas = [DATA_ORIGEM + timedelta(days=sind + i) for i in range(n)]
            if not datas or datas[-1].year < args.inicio or datas[0].year > args.fim:
                continue
            selecao = [i for i, dia in enumerate(datas) if args.inicio <= dia.year <= args.fim]
            if not selecao:
                continue
            selecionadas = [datas[i] for i in selecao]
            repetidas = set(selecionadas) & datas_vistas
            if repetidas:
                raise RuntimeError(f"Datas duplicadas em {caminho.name}: {min(repetidas)}")
            datas_vistas.update(selecionadas)

            # Todas as séries: arrays que terminam no eixo temporal (dias).
            # Para arrays pequenos, cada componente recebe uma coluna em diario.csv.
            # Para matrizes de 3000 PLSs, cria CSVs separados para evitar uma tabela gigante.
            colunas = {}
            por_pls = {}
            metadados = {"arquivo": caminho.name, "data_inicial": datas[0].isoformat(),
                         "data_final": datas[-1].isoformat(), "dias": n}
            for chave, valor in d.items():
                a = np.asarray(valor)
                if a.ndim == 0:
                    metadados[chave] = a.item()
                    catalogo[chave] = ("escalar", str(a.shape), "metadados.csv")
                    continue
                if a.shape[-1] != n:
                    raise ValueError(f"{chave} em {caminho.name}: último eixo não é diário: {a.shape}")
                if a.ndim == 1:
                    colunas[chave] = a[selecao]
                    catalogo[chave] = ("diaria", str(a.shape), "diario.csv")
                elif a.ndim == 2 and a.shape[0] <= 20:
                    for j in range(a.shape[0]):
                        colunas[f"{chave}_{j+1}"] = a[j, selecao]
                    catalogo[chave] = ("componentes_diarios", str(a.shape), "diario.csv")
                elif a.ndim == 2:
                    por_pls[chave] = a[:, selecao]
                    catalogo[chave] = ("por_pls", str(a.shape), f"{chave}.csv")
                elif a.ndim == 3 and a.shape[1] > 20:
                    for j in range(a.shape[0]):
                        nome = f"{chave}_{j+1}"
                        por_pls[nome] = a[j][:, selecao]
                    catalogo[chave] = ("componentes_por_pls", str(a.shape),
                                       ", ".join(f"{chave}_{j+1}.csv" for j in range(a.shape[0])))
                else:
                    raise ValueError(f"Formato não contemplado: {chave} {a.shape}")

            ordem = list(colunas)
            gravador = abrir_csv(args.saida / "diario.csv", ["data", "ano", "spin"] + ordem, handles)
            for j, dia in enumerate(selecionadas):
                gravador.writerow([dia.isoformat(), dia.year, caminho.name] +
                                  [colunas[nome][j] for nome in ordem])

            metas = abrir_csv(args.saida / "metadados.csv", list(metadados), handles)
            metas.writerow(list(metadados.values()))

            for nome, mat in por_pls.items():
                n_pls = mat.shape[0]
                cols = [f"pls_{i:04d}" for i in range(n_pls)]
                gravador = abrir_csv(args.saida / f"{nome}.csv", ["data"] + cols, handles)
                for j, dia in enumerate(selecionadas):
                    gravador.writerow([dia.isoformat()] + mat[:, j].tolist())

            cont += 1
            print(f"Lido: {caminho.name} | {selecionadas[0]} a {selecionadas[-1]}", flush=True)
    finally:
        for arq, _ in handles.values():
            arq.close()

    if cont == 0:
        raise RuntimeError("Nenhum ano encontrado no intervalo escolhido")
    with (args.saida / "catalogo_variaveis.csv").open("w", newline="", encoding="utf-8") as arq:
        writer = csv.writer(arq)
        writer.writerow(["variavel", "tipo", "formato_no_ultimo_spin", "arquivo_csv"])
        for chave, info in sorted(catalogo.items()):
            writer.writerow([chave, *info])
    print(f"\nConcluído: {cont} arquivos. CSVs em: {args.saida.resolve()}")


if __name__ == "__main__":
    main()
