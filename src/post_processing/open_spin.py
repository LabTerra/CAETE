import joblib
import matplotlib.pyplot as plt
import os

# Get user input for the run name
# run_name = input('run name: ')
run_name = "lu"
# grd = input('lat-long: ')
grd = "186-239"


outputs_folder = "/home/luana/Documents/outputs_JUSTRUNNED_CAETE_just_bia_and_precision/"
# outputs_folder = "/home/amazonfaceme/biancarius/CAETE-DVM-alloc-allom/outputs/"


# Set the path to the data directory
path = f'{outputs_folder}{run_name}/gridcell{grd}'

# Change the current working directory to the specified path
os.chdir(path)

# List the files and directories in the current path
contents = os.listdir()

# Filter files that start with "spin" to get all spins
spins = [item.replace("spin", "").replace(".pkz", "") for item in contents if item.startswith("spin")]
spins.sort()
print(spins)


# Iterate over all spins
for spin in spins:
    print(f"Processing spin {spin}")

    # Load data for the current spin
    with open("spin{}.pkz".format(spin), 'rb') as fh:
        dt = joblib.load(fh)

    # Specify the variables to be plotted
    variables_to_plot = ['emaxm', 'tsoil', 'photo', 'ar', 'npp', 'lai', 'rcm', 'f5', 'runom', 'evapm', 'wsoil', 'rm', 'rg', 'cleaf', 'cwood', 'croot', 'csap', 'cheart', 'csto', 'wue', 'area', 'ls']

    # Plot the specified variables in a 4x4 grid
    fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(20, 16))

    for i, key in enumerate(variables_to_plot):
        values = dt[key]
        row = i // 4
        col = i % 4
        axs[row, col].plot(values)
        axs[row, col].set_title(key)
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel(key)
        axs[row, col].grid(True)

    # Adjust the layout to prevent title overlap
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig('subplots_{}_{}.png'.format(run_name, spin))

    # Display the subplots
    plt.show()