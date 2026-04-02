import os
import joblib
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional


class CaeteOutputRun:
    """
    Classe para representar uma pasta de execução (run) do CAETE.
    Gerencia a estrutura de arquivos e pastas de uma simulação.
    """
    
    def __init__(self, run_path: str):
        """
        Inicializa um objeto CaeteOutputRun a partir do caminho da pasta.
        
        Args:
            run_path: Caminho para a pasta da run
        """
        self.run_path = Path(run_path)
        self.run_name = self.run_path.name
        
        # Inicializa atributos
        self.gridcells: List[str] = []
        self.gridcell_spins: Dict[str, int] = {}
        self.gridcell_vars: Dict[str, List[str]] = {}
        self.gridcell_years: Dict[str, tuple] = {}
        self.gridcell_special_samples: Dict[str, Dict[str, List[Any]]] = {}
        
        # Verifica arquivos obrigatórios
        self.caete_h5 = self.run_path / "CAETE.h5"
        self.state_start = self.run_path / "CAETE_STATE_START.pkz"
        self.state_end = self.run_path / "CAETE_STATE_END.pkz"
        self.nc_outputs_dir = self.run_path / "nc_outputs"
        
        # Processa a estrutura
        # Encontra todas as pastas gridcell
        for item in sorted(self.run_path.iterdir()):
            if item.is_dir() and item.name.startswith("gridcell"):
                gridcell_name = item.name
                self.gridcells.append(gridcell_name)
                
                spin_files = sorted(item.glob("spin*.pkz"))
                self.gridcell_spins[gridcell_name] = len(spin_files)
                
                variable_names = set()
                year_min: Optional[int] = None
                year_max: Optional[int] = None

                if spin_files:
                    first_spin = spin_files[0]
                    try:
                        data = joblib.load(first_spin)
                        if isinstance(data, dict):
                            variable_names.update(data.keys())
                            self.gridcell_special_samples[gridcell_name] = {
                                key: self._sample_values(data[key], max_len=10)
                                for key in ['time_unit', 'sind', 'eind']
                                if key in data
                            }
                        else:
                            self.gridcell_special_samples[gridcell_name] = {}
                    except Exception:
                        self.gridcell_special_samples[gridcell_name] = {}

                self.gridcell_vars[gridcell_name] = sorted(variable_names)
                self.gridcell_years[gridcell_name] = (year_min, year_max)

    @staticmethod
    def _sample_values(value: Any, max_len: int = 10) -> List[Any]:
        if isinstance(value, np.ndarray):
            return value.flatten()[:max_len].tolist()
        if isinstance(value, pd.Series):
            return value.iloc[:max_len].tolist()
        if isinstance(value, (list, tuple)):
            return list(value)[:max_len]
        try:
            iterator = iter(value)
        except TypeError:
            return [value]
        return [item for _, item in zip(range(max_len), iterator)]

    def get_start_state(self) -> Optional[pd.DataFrame]:
        """
        Lê o arquivo CAETE_STATE_START.pkz e retorna como DataFrame.
        
        Returns:
            DataFrame com os dados do estado inicial ou None se não encontrar
        """
        if not self.state_start.exists():
            return None
        
        
        print(f"Vai ler file {self.state_start}")


        try:
            with open(self.state_start, 'rb') as fh:
                data = joblib.load(fh)
                print(f"Abriu {self.state_start}")
                return data
            # if isinstance(data, pd.DataFrame):
            #     return data
            # if isinstance(data, dict):
            #     return pd.DataFrame(data)
            # return pd.DataFrame([data]) if data is not None else None
        except Exception as e:
            print(f"  ✗ Erro ao ler {self.state_start.name}: {e}")
            return None
    
    def get_end_state(self) -> Optional[pd.DataFrame]:
        """
        Lê o arquivo CAETE_STATE_END.pkz e retorna como DataFrame.
        
        Returns:
            DataFrame com os dados do estado final ou None se não encontrar
        """
        if not self.state_end.exists():
            return None
        
        try:
            data = joblib.load(self.state_end)
            if isinstance(data, pd.DataFrame):
                return data
            if isinstance(data, dict):
                return pd.DataFrame(data)
            return pd.DataFrame([data]) if data is not None else None
        except Exception as e:
            print(f"  ✗ Erro ao ler {self.state_end.name}: {e}")
            return None
    
    def summary_nc(self) -> Dict[str, Any]:
        """
        Sumariza os arquivos netCDF na pasta nc_outputs.
        
        Returns:
            Dicionário com informações sobre os arquivos .nc
        """
        nc_files = list(self.nc_outputs_dir.glob("*.nc4")) if self.nc_outputs_dir.exists() else []
        
        nc_info = {
            'count': len(nc_files),
            'files': []
        }
        
        for nc_file in nc_files:
            file_size = nc_file.stat().st_size
            nc_info['files'].append({
                'name': nc_file.name,
                'size_bytes': file_size,
                'size_mb': file_size / (1024 * 1024)
            })
        
        return nc_info
    
    def print_summary(self):
        """Exibe um sumário formatado da run."""
        print(f"\n{'='*60}")
        print(f"  RUN: {self.run_name}")
        print(f"{'='*60}")
        print(f"  📁 Path: {self.run_path}")
        
        # Gridcells
        print(f"\n  📊 GRIDCELLS:")
        print(f"     Total: {len(self.gridcells)}")
        for gridcell in self.gridcells:
            spins = self.gridcell_spins.get(gridcell, 0)
            print(f"     • {gridcell}: {spins} spin files")
            
            # Mostra variáveis se disponíveis
            vars_list = self.gridcell_vars.get(gridcell, [])
            if vars_list:
                print(f"       📈 Variáveis: {', '.join(vars_list)}")
                sample_values = self.gridcell_special_samples.get(gridcell, {})
                for key in ['time_unit', 'sind', 'eind']:
                    if key in sample_values:
                        print(f"       🔢 {key}: {sample_values[key]}")
            
            # Mostra range de anos se disponível
            years_range = self.gridcell_years.get(gridcell, (None, None))
            if years_range[0] is not None and years_range[1] is not None:
                print(f"       📅 Anos: {years_range[0]} - {years_range[1]}")
        
        # Arquivos de estado
        print(f"\n  💾 STATE FILES:")
        start_icon = "✓" if self.state_start.exists() else "✗"
        end_icon = "✓" if self.state_end.exists() else "✗"
        print(f"     {start_icon} CAETE_STATE_START.pkz")
        print(f"     {end_icon} CAETE_STATE_END.pkz")

        # Outputs
        print(f"\n  🌐 OUTPUTS:")
        # Arquivo CAETE.h5
        h5_icon = "✓" if self.caete_h5.exists() else "✗"
        print(f"     {h5_icon} CAETE.h5")
        # Retrieve nc outputs
        nc_summary = self.summary_nc()
        if nc_summary['count'] > 0:
            print(f"     Total: {nc_summary['count']} arquivos")
            print(f"     Detalhes:")
            for nc_file in nc_summary['files'][:5]:  # Mostra apenas os primeiros 5
                print(f"       • {nc_file['name']}: {nc_file['size_mb']:.2f} MB")
            if nc_summary['count'] > 5:
                print(f"       ... e mais {nc_summary['count'] - 5} arquivos")
        else:
            print(f"     ✗ Nenhum arquivo .nc encontrado")
        
        print(f"\n{'='*60}")
    
    def summary(self) -> Dict[str, Any]:
        """Retorna um dicionário com o sumário (para compatibilidade)."""
        nc_summary = self.summary_nc()

        return {
            'run_name': self.run_name,
            'run_path': str(self.run_path),
            'gridcells': {
                'count': len(self.gridcells),
                'names': self.gridcells,
                'spin_counts': self.gridcell_spins,
                'variables': self.gridcell_vars,
                'year_ranges': self.gridcell_years
            },
            'state_files': {
                'has_start': self.state_start.exists(),
                'has_end': self.state_end.exists()
            },
            'nc_outputs': nc_summary,
            'caete_h5_exists': self.caete_h5.exists()
        }
    
    def __repr__(self):
        return f"CaeteOutputRun(run_name='{self.run_name}', gridcells={len(self.gridcells)})"


class CaeteOutputs:
    """
    Classe principal para gerenciar múltiplas execuções do CAETE.
    A pasta principal deve conter apenas subpastas (runs).
    """

    def __init__(self, path: str):
        """
        Inicializa o objeto CaeteOutputs verificando a estrutura da pasta.
        
        Args:
            path: Caminho para a pasta principal dos outputs
        
        Raises:
            ValueError: Se a pasta contiver arquivos soltos ou estrutura inválida
        """
        self.path = Path(path)
        
        if not self.path.exists():
            raise ValueError(f"❌ O caminho {path} não existe")
        
        if not self.path.is_dir():
            raise ValueError(f"❌ O caminho {path} não é um diretório")
        
        # Lista de nomes das runs (subpastas)
        self.run_names: List[str] = []
        # Dicionário de objetos CaeteOutputRun
        self.runs: Dict[str, CaeteOutputRun] = {}
        
        # Processa a estrutura
        for item in self.path.iterdir():
            if item.is_dir():
                self.run_names.append(item.name)
                self.runs[item.name] = CaeteOutputRun(str(item))
            else:
                raise ValueError(
                    f"❌ A pasta principal {self.path} não pode conter arquivos soltos. "
                    f"Encontre apenas subpastas de runs. Arquivo solto: {item.name}"
                )

        if not self.run_names:
            raise ValueError(f"❌ Nenhuma subpasta (run) encontrada em {self.path}")

    def print_summary(self):
        """Exibe um sumário formatado de todas as runs."""
        print(f"\n{'='*60}")
        print(f"  CAETE OUTPUTS - VISÃO GERAL")
        print(f"{'='*60}")
        print(f"\n  📁 Diretório principal: {self.path}")
        print(f"  🚀 Total de runs: {len(self.run_names)}")
        
        print(f"\n  📋 LISTA DE RUNS:")
        print(f"  {'-'*56}")
        
        for i, run_name in enumerate(self.run_names, 1):
            run_obj = self.runs[run_name]
            nc_count = run_obj.summary_nc()['count']
            state_icon = "✓" if (run_obj.state_start.exists() and run_obj.state_end.exists()) else "✗"
            
            print(f"  {i:2d}. {run_name}")
            print(f"      📊 Gridcells: {len(run_obj.gridcells)} | 🔄 Spins: {sum(run_obj.gridcell_spins.values())}")
            print(f"      🌐 NC files: {nc_count} | 💾 State files: {state_icon}")
            print(f"      {'-'*52}")
        
        print(f"\n{'='*60}\n")

    def summary(self) -> Dict[str, Any]:
        """Retorna um sumário simples com os nomes das runs disponíveis."""
        return {
            'path': str(self.path),
            'run_names': self.run_names
        }
    
    def detailed_summary(self) -> Dict[str, Any]:
        """Retorna um dicionário com sumários detalhados (para compatibilidade)."""
        detailed = {
            'main_path': str(self.path),
            'total_runs': len(self.run_names),
            'runs': {}
        }
        
        for run_name, run_obj in self.runs.items():
            detailed['runs'][run_name] = run_obj.summary()
        
        return detailed
    
    def __repr__(self):
        return f"CaeteOutputs(path='{self.path}', runs={self.run_names})"
    
    def __len__(self):
        return len(self.run_names)
    
    def __iter__(self):
        return iter(self.runs.values())
    
    def __getitem__(self, key):
        return self.runs[key]


# Exemplo de uso:
if __name__ == "__main__":
    # Exemplo de como usar as classes
    try:
        # Para usar, forneça o caminho da pasta principal
        caete_outputs = CaeteOutputs("outputs")
        
        # Sumário básico formatado
        caete_outputs.print_summary()

        # Sumário detalhado de uma run específica
        for run_name in caete_outputs.run_names:
            caete_outputs.runs[run_name].print_summary()
            
            # Acessar e ler estados de uma run
            run_obj = caete_outputs.runs.get(run_name)
            print(type(run_obj))
            if run_obj:
                start_state = run_obj.get_start_state()
                if start_state is not None:
                    print(f"\n  📊 Estado inicial carregado: {start_state.shape[0]} linhas x {start_state.shape[1]} colunas")
                    print(f"  Colunas: {', '.join(start_state.columns[:5])}{'...' if len(start_state.columns) > 5 else ''}")
                
                end_state = run_obj.get_end_state()
                if end_state is not None:
                    print(f"\n  📊 Estado final carregado: {end_state.shape[0]} linhas x {end_state.shape[1]} colunas")
    
    except ValueError as e:
        print(f"\n❌ Erro: {e}")