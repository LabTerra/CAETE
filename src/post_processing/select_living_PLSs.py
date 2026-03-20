import os
import pandas as pd
import re


run_name = input("What is the run name? ")
grd_name = "186-239"
path_csv = f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/30perc_reduction/{run_name}/gridcell{grd_name}/csv"
start_year = int("1979")
end_year = int("2016")
end_year = end_year + 1

# Get a list of all files with .csv extension in the directory(there is one csv for each alive PLS)
list_files = [file for file in os.listdir(path_csv) if file.endswith(".csv")]
   

# # Extract the final group of numbers from the file names(get the PLS id)
s = [re.search(r"_([0-9]+)\.csv", file).group(1) for file in list_files if re.search(r"_([0-9]+)\.csv", file)]

# # Create an empty DataFrame to store the merged and sorted data
final_merged_df = pd.DataFrame()


# # Extract the final group of numbers from the file names and group the files (group each PLS considering all spins)
for file in list_files:
    match = re.search(r"EV_([0-9]+)\.csv", file)
    if match:
        group_number = match.group(1)
        file_path = os.path.join(path_csv, file)
        df = pd.read_csv(file_path)
#       Add a column to store the group number for reference
        df['GroupNumber'] = group_number
        # Merge the current group into the final DataFrame
        final_merged_df = pd.concat([final_merged_df, df], ignore_index=True)
       

# # Create a DataFrame with all combinations of years and group numbers
all_years = range(start_year, end_year)

# #Select the PLS ID for the alives
all_group_numbers = final_merged_df['GroupNumber'].unique()
# print(all_group_numbers)

# Creating a pandas DataFrame 'all_combinations' by generating all possible combinations
# of 'YEAR' from the list 'all_years' and 'GroupNumber' from the list 'all_group_numbers'.
# Each combination is represented as a tuple in the form (year, group_number).
# The resulting DataFrame has columns 'YEAR' and 'GroupNumber'.
all_combinations = pd.DataFrame([(year, group_number) for year in all_years for group_number in all_group_numbers],
                                 columns=['YEAR', 'GroupNumber'])



# # Merge all_combinations with final_merged_df to fill gaps
final_merged_df = pd.merge(all_combinations, final_merged_df, on=['YEAR', 'GroupNumber'], how='left')

# # Fill NaN values in 'PID' with 'GroupNumber' and convert back to int
final_merged_df['PID'] = final_merged_df['PID'].fillna(final_merged_df['GroupNumber']).astype(int)
# Fill NaN values in 'OC' with 0
final_merged_df['OC'] = final_merged_df['OC'].fillna(0.0)  

# Sort the final DataFrame by the "YEAR" column in ascending order
final_merged_df.sort_values(by=["GroupNumber", "YEAR"], inplace=True)

# Drop the temporary group number column
final_merged_df.drop(columns=['GroupNumber'], inplace=True)

# Save the final merged and sorted DataFra  prinme to a new CSV file
final_file_path = os.path.join(path_csv, "final_merged_sorted_data.csv")

final_merged_df.to_csv(final_file_path, index=False)

# print("Your file has been created! Find it in:", path_csv)


# #Now get the trait values from attrs (without considering the occupation)
# # Read file with all pls traits
pls_traits = pd.read_csv("/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/state_start/MAN_save_spin_1/pls_attrs-6000.csv")

# # Get the PIDs, that is, the alive PLSs
pids_to_select = final_merged_df['PID'].unique()

# # Filtrar pls_traits com base nos PIDs em pids_to_select
selected_PLS_traits = pls_traits[pls_traits['PLS_id'].isin(pids_to_select)]
# #PLS id, year and occupation
PLS_ocp_year = final_merged_df

# # Realiza a agregação
ocp_traits = pd.merge(PLS_ocp_year, selected_PLS_traits[['PLS_id', 'g1','sla_random','wd_random']], left_on='PID', right_on='PLS_id', how='left')
# # Remove a coluna 'PLS_id' da nova tabela
ocp_traits = ocp_traits.drop('PLS_id', axis=1)

# #Calculates the value for a trait multiplying it by the PLS occupation
ocp_traits['sla_ocp'] = ocp_traits['OC']*ocp_traits['sla_random']
ocp_traits['wd_ocp'] = ocp_traits['OC']*ocp_traits['wd_random']
ocp_traits['g1_ocp'] = ocp_traits['OC']*ocp_traits['g1']
# path_output = f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/{run_name}/"
# # Save the final merged and sorted DataFra  prinme to a new CSV file
# final_merged_path = os.path.join(path_output, f"PLS_alive_traits_{run_name}.csv")
ocp_traits.to_csv(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/src/PLS_alive_traits_{run_name}.csv", index=False)
ocp_traits.to_csv(f"/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/MAN/experiments/30perc_reduction/{run_name}/PLS_alive_traits_{run_name}.csv", index=False)






# # # # Select only data from living pls (those in the list of files)
# # new_pls_traits = pls_traits[pls_traits['PLS_id'].isin(pls_id['pls_id'])]
    

# # Merge the all_combinations DataFrame with the final DataFrame to fill gaps
# final_merged_df = pd.merge(all_combinations, final_merged_df, on=['YEAR', 'GroupNumber'], how='left')

# # Fill NaN values in columns other than 'YEAR' and 'GroupNumber' with 0
# final_merged_df.fillna(0, inplace=True)

# # Sort the final DataFrame by the "YEAR" column in ascending order
# final_merged_df.sort_values(by=["GroupNumber", "YEAR"], inplace=True)

# # Drop the temporary group number column
# final_merged_df.drop(columns=['GroupNumber'], inplace=True)

# # Save the final merged and sorted DataFrame to a new CSV file
# final_file_path = os.path.join(path_csv, "final_merged_sorted_data.csv")
# final_merged_df.to_csv(final_file_path, index=False)

# # # Sort the final DataFrame by the "YEAR" column in ascending order
# # final_merged_df.sort_values(by=["GroupNumber", "YEAR"], inplace=True)

# # # Drop the temporary group number column
# # final_merged_df.drop(columns=['GroupNumber'], inplace=True)

# # # Save the final merged and sorted DataFrame to a new CSV file
# # final_file_path = os.path.join(path_csv, "final_merged_sorted_data.csv")
# # final_merged_df.to_csv(final_file_path, index=False)



# # # Print the file groups
# # # for group_number, files in file_groups.items():
# #     # print(f"Files with group number {group_number}: {files}")






# # # # Create a DataFrame with the extracted numeric part
# # pls_id = pd.DataFrame({"pls_id": pd.to_numeric(s)})

# # # # Check the structure of pls_id to make sure it is as expected
# # print(pls_id.info())

# # # # Read file with all pls traits
# # pls_traits = pd.read_csv("/home/bianca/bianca/CAETE-DVM-alloc-allom/outputs/pls_attrs-3000.csv")

# # # # Select only data from living pls (those in the list of files)
# # new_pls_traits = pls_traits[pls_traits['PLS_id'].isin(pls_id['pls_id'])]


# #exclude files of PLSs csv for each spin
# temp_files = os.listdir(path_csv)

# for file in temp_files:
#     if file.startswith('baserun') and file.endswith('.csv'):
#         print(file)
#         file_path2 = os.path.join(path_csv, file)
#         os.remove(file_path2)



# #PLS id, year and occupation
# PLS_ocp_year = final_merged_df

# # Get a list of all files with .csv extension in the directory
# list_files = [file for file in os.listdir(path_csv) if file.startswith("sorted_merged_") and file.endswith(".csv")]

# # Delete temporary files
# for file in list_files:
#     print(file)
#     file_path = os.path.join(path_csv, file)
#     os.remove(file_path)

# print("Your file has been created! Find it in:", path_csv)






