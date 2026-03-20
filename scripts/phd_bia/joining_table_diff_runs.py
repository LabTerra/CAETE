import pandas as pd

run_name1 = "175235_regprec_3000_1"#input('Give the run name 1 : ')
run1 = pd.read_csv(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name1}/gridcell175-235/csv/final_merged_sorted_data.csv")

run_name2 = "175235_regprec_3000_1"#input('Give the run name 2 : ')
run2 = pd.read_csv(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name2}/gridcell175-235/csv/final_merged_sorted_data.csv")

run_name3 = "175235_regprec_3000_3"#input('Give the run name 2 : ')
run3 = pd.read_csv(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/{run_name3}/gridcell175-235/csv/final_merged_sorted_data.csv")

# Join tables according to common PID
joined_tables = pd.merge(run1, run2, run3, on=['PID', 'YEAR'], how='inner', suffixes=('_tabela1', '_tabela2','_tabela3'))

pid_unicos = joined_tables['PID'].nunique()
print(pid_unicos)   