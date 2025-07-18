# cupix

Cosmology using Px (p-cross) - the Lyman-alpha cross-spectrum, which measures 3D small-scale clustering of the Lyman alpha forest.

Modeled on the cup1d (https://github.com/igmhub/cup1d) package for inference from the one-dimensional power spectrum, P1D.

This package is in development, and not yet ready for external use.

This repository currently uses the ForestFlow emulator (https://github.com/igmhub/ForestFlow) for cosmology and IGM modeling, and LaCE for cosmology computations (https://github.com/igmhub/LaCE). Alternative emulators may be added in the future. If you would like to collaborate, please email Martine Lokken (mlokken@ifae.es), Sindhu Satyavolu (ssatyavolu@ifae.es), Andreu Font-Ribera (afont@ifae.es) or Jonas Chaves-Montero (jchaves@ifae.es).
 

### Installation

- Download and install Conda. You can find the instructions here https://docs.anaconda.com/miniconda/miniconda-install/

- Create a new conda environment. It is usually better to follow python version one or two behind. In July 2025, the latest is 3.15, so we recommend 3.13.

```
conda create -n cupix -c conda-forge python=3.13 camb mpi4py
conda activate cupix
pip install --upgrade pip
```
- Clone and install LaCE (do so within the environment created above):

```
git clone https://github.com/igmhub/LaCE.git
cd LacE
pip install -e .
``` 

- Clone and install ForestFlow with Px routines (do so within the environment created above):

```
git clone https://github.com/igmhub/ForestFlow.git
cd ForestFlow
pip install -e .[px]
``` 

- Clone and install cupix (do so within the environment created above):

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


- All notebooks in the repository are in .py format. To generate the .ipynb version, run:

```
jupytext --to ipynb notebooks/*/*.py
```

- If you want to use notebooks via JupyterHub, you'll also need to download `ipykernel`:

```
pip install ipykernel
python -m ipykernel install --user --name cup1d --display-name cupix
```

There are no clean tutorials yet, but you can find some in-development notebooks with examples in `notebooks/development`. You will need to point them to your own data because there are no small test files included in the package, at the moment.