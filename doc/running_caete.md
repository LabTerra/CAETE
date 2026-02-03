# Running CAETE fast to test

[⇦ Back](../README.md)

This section presents instructions to run CAETE model.

We assume here that you have already configured your machine to run CAETE, following [Build Instructions](./build_instructions.md) and [System Configuration](./system_config.md) sections previously!


## 1. Generating the PLS table

The PLS (Plant Life Strategy) table is required for running the CAETÊ model. You can generate the PLS table using the `plsgen.py` script located in the `src` folder. This script creates a PLS table based on the specified parameters and saves it as a CSV file.

In this example, we will generate a PLS table with 10,000 entries and save it to the `PLS_MAIN` folder:

```bash
python plsgen.py -n 10000 -f PLS_MAIN
```

After generating the PLS table, access the `caete_model.py` file and update the `PLS_TABLE_PATH` variable with the path to this generated CSV file!


## 2. Set input data

Download the input files following [Input Files](./input_data.md) section. Optionally, you can place them inside the `input` folder in this repository. For example, using **20CRv3-ERA5** data, you will have a directory structure like this:

```
$ $input/
$ -- 20CRv3-ERA5/
$ -- -- obsclim
$ -- -- spinclim
$ -- -- ...
```

You can edit the paths to the input data files inside `caete_driver_CMIP6.py` and `caete_driver.py` scripts before running them.


## 3. Model configuration

Model configuration parameters are set in the `caete.toml` file located in the `src` folder. You can edit this file to change model parameters before running the model.
  
- **Specify number of cores:** First, check the number of cores that your machine have. Then, we reccommend you to subtract 1 from this number. Use the result to set the variables `nprocs` and `max_processes` inside `src/caete.toml` file. **Example: If you have 4 cores, set variables to 3.**

### Tips to run CAETE faster

- You can set `grd/gridlist_test.csv` file or create a new one with just a few coordinates. **Using fewer coordinates will allow CAETE to run faster!**
- You can edit the `spinup` method inside `src/worker.py` file to reduce the spinups. To do this, just comment all `gridcell.run_gridcell()` invokations you want.


## 4. Compile Fortran code

This step is important always when:

- The `npls_max` variable was updated on `caete.toml`;
- You are running CAETE for the first time;
- The Fortran source code was updated;

In order to compile CAETE fortran modules, run:
  
  - Windows:
    ```Powershell
    nmake -f Makefile_win
    ```

  - Linux/MacOS:
    ```bash
    make
    ```

## 4. Run CAETE
  
You can run CAETE model from the `src` folder using the same python executable that is used in your Makefile.

```bash
$ cd src
$ python caete_driver.py

# or

$ cd src
$ python caete_driver_CMIP6.py
```

The idea is that when you want to do a experiment with the model, you create a script that mimics the caete_driver.py script but with your own configuration (e.g., different input files, different output files, different spinup configurations).

```caete_driver.py``` runs the model with a subset (n=16) of the gridcells in the input files.

```caete_driver_CMIP6.py``` runs the model with CMIP6 forcing data for 4 gridcells.

[⇦ Back](../README.md)