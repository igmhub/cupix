# cupix

Cosmology using Px (p-cross) - the Lyman-alpha cross-spectrum, which measures 3D small-scale clustering of the Lyman alpha forest.

Modeled on the cup1d (https://github.com/igmhub/cup1d) package for inference from the one-dimensional power spectrum, P1D.

This package is in development, and not yet ready for external use.

This repository currently uses the ForestFlow emulator (https://github.com/igmhub/ForestFlow) for cosmology and IGM modeling, and LaCE for cosmology computations (https://github.com/igmhub/LaCE). Alternative emulators may be added in the future. If you would like to collaborate, please email Martine Lokken (mlokken@ifae.es), Sindhu Satyavolu (ssatyavolu@ifae.es), Andreu Font-Ribera (afont@ifae.es) or Jonas Chaves-Montero (jchaves@ifae.es).
 

### Installation

- Download and install Conda. You can find the instructions here https://docs.anaconda.com/miniconda/miniconda-install/

- Create a new conda environment. In July 2025, we recommend to use Python 3.10 or 3.11.

```
conda create -n cupix -c conda-forge python=3.11 camb mpi4py
conda activate cupix
pip install --upgrade pip
```
- Navigate to the directory where you'd like to install all new software. Clone and install LaCE (do so within the `cupix` environment created above):

```
git clone https://github.com/igmhub/LaCE.git
cd LaCE
pip install -e .
``` 

- Navigate back to the base software directory (`cd ..`) and clone and install ForestFlow with Px routines (still within the `cupix` environment created above):

```
git clone https://github.com/igmhub/ForestFlow.git
cd ForestFlow
pip install -e .[px]
``` 

- Navigate back to the base software directory (`cd ..`) and clone and install cupix (still within the `cupix` environment created above):

```
git clone https://github.com/igmhub/cupix.git
cd cupix
pip install -e .
``` 

#### NERSC users:

- You need to compile ``mpi4py`` package on NERSC (see [here](https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment)).

```
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```

### Notebooks / tutorials


- All notebooks in the repository are in .py format. You need to install jupytext to generate the .ipynb versions:

```
conda install jupytext -c conda-forge
```
Then run:

```
jupytext --to ipynb notebooks/*/*.py
```

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name cupix --display-name cupix
```

There are no clean tutorials yet, but you can find some in-development notebooks with examples in `notebooks/development`. You will need to point them to your own data because there are no small test files included in the package, at the moment.