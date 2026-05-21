Install all requirements manually, create Python environment and run
====================================================================

Requirements
------------

Linux/macOS
~~~~~~~~~~~

- Install:
    - **C Compiler: gcc**
    - **Fortran Compiler: gfortran**
- Usually they are pre-installed or easily available.
- The flang AOCC compiler (Fortran language compiler and driver) is supported in linux. You will need to set some variables in the Makefile to use it.


Configure Python environment
----------------------------

.. danger::
    **Avoid system Python! Use isolated Python environments!** System Python is used by core utilities and package managers. Modifying it can break your system.

Required **pip** for package installation and **numpy.f2py** for compiling Fortran modules.

There are several options to install and manage multiple Python versions in all platforms. For our purposes, we need a python installation that meets the requirements above. Additionally, the Python installation must be reachable from the terminal/shell where you will run the Makefile to build the Fortran/Python extension module. This means that the Python executable must be in the ``PATH`` environment variable or can be launched via an environment manager like ``pyenv``, ``uv``, ``PIM``, etc. 

.. warning::
    Python installation must be compatible with the C compiler installed in your system. More specifically, the Python installation must be compiled with the same C runtime library as the C compiler used to build the Fortran/Python extension module. 
    
.. warning::
    In Windows, this usually means that you should use the **official python installer** from **python.org**, since it is compiled with **MSVC**. Other Python distributions like Anaconda may use different C runtimes and may lead to issues when building the extension module with MSVC.

**Windows:**

    - Consider Python Install Manager (PIM) for multiple versions.
    - Python version must be **Python == 3.11**. Use official Python 3.11 installer!
    
**Linux/macOS:**
    - Python version must be **Python >= 3.11 <= 3.13**
    - Use `pyenv` to manage Python packages. Other tools like ``conda`` or ``uv`` may work but are not guaranteed.


Test f2py
---------

- Follow `NumPy f2py "quick way" <https://numpy.org/doc/stable/f2py/f2py.getting-started.html#>`_.
- If successful → CAETE compilation likely to work!