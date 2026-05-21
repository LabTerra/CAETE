Requirements and Setup
======================


TERMINAR
TERMINAR
TERMINAR
TERMINAR
TERMINAR
TERMINAR
TERMINAR
./TEMP_system_config.md
TERMINAR
TERMINAR
TERMINAR
TERMINAR
TERMINAR
TERMINAR
TERMINAR


Install and configure compilers
-------------------------------

Windows
~~~~~~~

- Installation Order:
    - First: Install **C Compiler MSVC**
    - Second: Install **Fortran Compiler: Intel OneAPI**
- The gfortan from WinLibs (GCC 14.2.0 (with MCF threads) + MinGW-w64 12.0.0 (UCRT) - release 1) can also be used in windows along with MSVC.

Linux/macOS
~~~~~~~~~~~

- Install:
    - **C Compiler: gcc**
    - **Fortran Compiler: gfortran**
- Usually they are pre-installed or easily available.
- The flang AOCC compiler (Fortran language compiler and driver) is supported in linux. You will need to set some variables in the Makefile to use it.


Configure Python environment
----------------------------

There are several options to install and manage multiple Python versions in all platforms. For our purposes, we need a python installation that meets the requirements above. Additionally, the Python installation must be reachable from the terminal/shell where you will run the Makefile to build the Fortran/Python extension module. This means that the Python executable must be in the ``PATH`` environment variable or can be launched via an environment manager like ``pyenv``, ``uv``, ``PIM``, etc. 

.. warning::
    Python installation must be compatible with the C compiler installed in your system. More specifically, the Python installation must be compiled with the same C runtime library as the C compiler used to build the Fortran/Python extension module. 
    
.. warning::
    In Windows, this usually means that you should use the **official python installer** from **python.org**, since it is compiled with **MSVC**. Other Python distributions like Anaconda may use different C runtimes and may lead to issues when building the extension module with MSVC.


**Best Practices:**

Windows
~~~~~~~

    .. danger::
        **Avoid system Python! Use isolated Python environments!** System Python is used by core utilities and package managers. Modifying it can break your system.

    - Consider Python Install Manager (PIM) for multiple versions.
    - Python version must be **Python == 3.11**. Use official Python 3.11 installer!
    - Required **pip** for package installation and **numpy.f2py** for compiling Fortran modules.

Linux/macOS
~~~~~~~~~~~

    .. danger::
        **Avoid system Python! Use isolated Python environments!** System Python is used by core utilities and package managers. Modifying it can break your system.

    - Python version must be **Python >= 3.11 <= 3.13**
    - Required **pip** for package installation and **numpy.f2py** for compiling Fortran modules.
    - Use `pyenv` to manage Python packages. Other tools like ``conda`` or ``uv`` may work but are not guaranteed.
    - Package Management: Standard **pip**


Test f2py
---------

- Follow `NumPy f2py "quick way" <https://numpy.org/doc/stable/f2py/f2py.getting-started.html#>`_.
- If successful → CAETE compilation likely to work!