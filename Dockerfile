# Use Miniconda as base image (Debian GNU/Linux 13 - trixie)
FROM anaconda/miniconda:latest@sha256:ffb09b25d6ba331b18f0a87f5b2f0fc1b6e661f99f765e4c2d7f61e015df0efd

# Copy local requirements.txt file to docker image
COPY requirements-pip.txt /tmp/

# Install Python, libs, compilers and Python packages on Conda base environment
RUN conda install --override-channels --channel conda-forge -y \
    python=3.13 \
    gfortran=15.2.0 \
    gcc=15.2.0 \
    libgomp=15.2.0 \
    make=4.4.1 \
    && pip install -r /tmp/requirements-pip.txt \
    && conda clean -afy

# Set CMD
CMD ["python", "--version"]