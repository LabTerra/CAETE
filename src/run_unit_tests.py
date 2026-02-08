import subprocess
import sys
from pathlib import Path

"""
Runner: Unit Tests - To be called from the Makefiles

NOTE: WORK in progress. Unit tests coverage is very low,
      but does crucial checks for the core functionality
      of the program. More unit tests will be added in the future.

This script will run the unit tests in the src folder.
The unit tests are in the following files:

- _geos.py (full coverage (unit tests) of cartographic functions in the _geos module)
- caete_jit.py (full coverage (unit tests) of the functions in the caete_jit module)
- caete.py (functional test, not a unit test. Tests basic functionality of the program,
  but does not test all the functions in the caete.py module. More unit tests will be added in the future.)

These tests must be as short as possible and should not take more than a few seconds to run.
They should test the basic functionality of the program and check for any errors or exceptions.
"""

def run_unit_test(command:str = sys.executable, module:str = ""):
    """Run a unit test for a given module using a subprocess."""

    if is_module_empty:= module == "" or module is None:
        assert not is_module_empty, "Must provide a module"

    module_path = Path(module)
    print(f"Testing {module_path}")
    if not module_path.exists():
        print("Module path must be a valid file")
        return

    module = str(module_path.resolve())

    error = subprocess.run([command, module])
    print("=======---===---===---===---=======")
    if error.returncode != 0:
        print(F"Error in {module} Unit Test")
    return

if __name__ == "__main__":

    print("UNIT TESTS + Functional test in region/grd_mt")

    # Unit tests in _geos.py
    run_unit_test(module="_geos.py")

    # Unit tests in caete_jit.py
    run_unit_test(module="caete_jit.py")

    # Functional test in caete.py.
    # Perfom a basic functionality test in the region and grd_mt modules.
    # This is not a unit test, but it is important to check that the basic
    # functionality of the program is working correctly.
    # TODO: Add unit tests in the caete.py module.
    # This is not trivial, but it is important to have unit
    # tests for the core functionality of the program.
    run_unit_test(module="caete.py")

    # This would raise an assertion error, since the module argument is required.
    # run_unit_test(module="") # will raise an assertion error
