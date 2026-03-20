# ! Copyright 2017- LabTerra

# !     This program is free software: you can redistribute it and/or modify
# !     it under the terms of the GNU General Public License as published by
# !     the Free Software Foundation, either version 3 of the License, or
# !     (at your option) any later version.)

# !     This program is distributed in the hope that it will be useful,
# !     but WITHOUT ANY WARRANTY; without even the implied warranty of
# !     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# !     GNU General Public License for more details.

# !     You should have received a copy of the GNU General Public License
# !     along with this program.  If not, see <http://www.gnu.org/licenses/>.

# ! AUTHOR: JP Darela e Luana Becker da Luz

# Build caete_module & debug programs
PYEXEC = python
F2PY = $(PYEXEC) -m numpy.f2py
MODNAME = caete_module
INTERFACES = $(MODNAME).pyf
FC = gfortran

# Sources for compilation in run mode (F2PY)
sources = global.f90 funcs.f90 soil_dec.f90 cc.f90 allocation.f90 allocation2.f90 productivity.f90 budget.f90 budget_allom.f90

# FLAGS DE REPRODUTIBILIDADE NUMÉRICA
# -fno-unsafe-math-optimizations: Desativa otimizações que podem violar padrões IEEE
# -frounding-math: Respeita modo de arredondamento
# -fsignaling-nans: Habilita tratamento de NaNs sinalizadores
# -ffloat-store: Força armazenamento em memória (não registradores)
# -mfpmath=sse: Usa SSE para matemática de ponto flutuante
# -msse2: Instruções SSE2
# -fno-fast-math: Desativa todas otimizações "fast-math"
# -fprotect-parens: Respeita parênteses na ordem das operações
# -std=legacy: Mantém compatibilidade com código legado
PREC = -fno-unsafe-math-optimizations -frounding-math -fsignaling-nans \
       -ffloat-store -mfpmath=sse -msse2 -fno-fast-math -fprotect-parens \
       -std=legacy
# PREC = -ffloat-store -mfpmath=sse -msse2 -fno-fast-math -std=legacy

DBG = -g -fcheck=all -ffpe-trap=invalid,zero,overflow -fbacktrace -fbounds-check -Wall -Wextra -Wconversion -pedantic

# Creates interface
interface: $(sources)
	$(PYEXEC) ask_npls.py
	$(F2PY) -h $(INTERFACES) $(sources) -m $(MODNAME) --overwrite-signature --quiet

# Compile
so: interface
	$(F2PY) -c $(INTERFACES) $(sources) --f90flags="$(PREC)" -lgomp
# 	$(F2PY) -c $(INTERFACES) $(sources) --f90flags="$(PREC) $(DBG) -fopenmp" -lgomp
# 	$(F2PY) -c $(INTERFACES) $(sources) --f90flags="$(PREC) -fopenmp" -lgomp

# Clean directory
clean:
	rm -rf test *.mod *.s *.o pls_ex.txt __pycache__ run_debug
	rm -rf NP1.npy NP2.npy NP3.npy
	rm -rf wallo.npy gallo.npy 
	clear
