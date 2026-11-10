import joblib
import os
import matplotlib.pyplot as plt


# Function to read the pickled file
def read_pkz(filename):
    with open(filename, 'rb') as fh:
        dt = joblib.load(fh)
    # print(dt.keys())  # List the available keys for the outputs
    return dt


# Get user input for the run name
# output_folder_path = input('Base CAETE path: ')
# run_name_output_folder = input('Run name output folder:')
# output_grid_subfolder = input('Output grid sub folder: ')

output_folder_path = "/home/luanabeckerdaluz/"
run_name_output_folder = "sombrero_outputs"
output_grid_subfolder = "gridcell186-239"


# Set the path to the data directory
path = f'{output_folder_path}/{run_name_output_folder}/{output_grid_subfolder}'
print(f"Changing directory to folder '{path}'...")
os.chdir(path)

# Filter files in the current path that start with "spin" to get all spins
spins = [item.replace("spin", "").replace(".pkz", "") for item in os.listdir() if item.startswith("spin")]

print(spins)
spins = [spins[0]]
print(spins)


# Iterate over all spins
for spin in spins:
    spin_filename = "spin{}.pkz".format(spin)
    print(f"Processing file '{spin_filename}'...")

    dt = read_pkz(spin_filename)

    # Plot the specified variables in a 4x4 grid
    ncols = 6
    nrows = 9
    # figsize = (20, 16)
    figsize = (40, 30)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Specify the variables to be plotted
    variables_to_plot = dt.keys()
    # variables_to_plot = ['emaxm', 'tsoil', 'photo', 'ar', 'npp', 'lai', 'rcm', 'f5', 'runom', 'evapm', 'wsoil', 'rm', 'rg', 'cleaf', 'csap', 'cheart', 'csto', 'wue', 'area', 'ls']

    for i, key in enumerate(variables_to_plot):
        # print(f"Plotting key {key}...")
        if key in ["lim_status", "u_strat"]:
            continue

        values = dt[key]
        row = i // ncols
        col = i % ncols
        axs[row, col].plot(values)
        axs[row, col].set_title(key)
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel(key)
        axs[row, col].grid(True)

    # Adjust the layout to prevent title overlap
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig('subplots_{}_{}_{}.png'.format(run_name_output_folder, output_grid_subfolder, spin))

    # Display the subplots
    plt.show()