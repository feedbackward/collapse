
# collapse: a study of criterion collapse in machine learning

In this repository, we provide software and demonstrations related to the following paper:

- [Criterion collapse and loss distribution control](https://arxiv.org/abs/2402.09802). Matthew J. Holland. *Preprint*.

This repository contains code which can be used to faithfully reproduce all the experimental results given in the above paper, and it can be easily applied to more general machine learning tasks outside the examples considered here.


## Our setup

Below, we describe the exact procedure by which we set up a (conda-based) virtual environment and installed the software used for our numerical experiments.

First, we [install miniforge](https://github.com/conda-forge/miniforge) (i.e., a conda-forge aligned miniconda).

Just to be safe, we then update conda in the usual fashion, as follows.

```
$ conda update -n base conda
```

Next, create a new virtual environment which includes PyTorch.

```
$ conda create -n collapse pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Note that we have specified an older version of PyTorch to match our CUDA 11.7 installation. Feel free to adjust this installation for your own system (see [reference on using older versions](https://pytorch.org/get-started/previous-versions/)). Running this command on our machine took a few minutes. Next, a few more installs within the new environment.

```
$ conda activate collapse
(collapse) $ conda install jupyterlab
(collapse) $ conda install matplotlib
```

The matplotlib bit was slow (5-10 minutes), but it goes through. To wrap things up, let's use [pip](https://pypi.org/project/pip/) to get [mlflow](https://mlflow.org/).

```
(collapse) $ pip install mlflow
```

This concludes our basic setup.


## Getting started

Anyone visiting this repository is probably interested in the software used to obtain the results shown in our paper. Any files not described explicitly are helper files used in support of the experiments carried out in the files given below.

### Simple demonstrations

- `demo_simple_tests.ipynb`: Figures 2 and 5 are generated here.
- `demo_surrogate_nolink.ipynb`: Figure 1 is generated here.


### Main empirical tests

The core content of our empirical investigations (section 4 of the paper) is a set of experiments using four real-world benchmark datasets. All the core settings are given in `run_benchmarks.py` and `benchmarks.py`. The former is essentially an experiment configuration file, and the latter is just a driver script that processes arguments passed from the command line to all the various helper functions we have defined in other files (e.g., for data handling, model initialization, training loops, evaluation, etc.). All these "other files" are stored in the `setup` directory.

Running our full set of experiments, for all relevant methods and datasets in one shot can be done by simply running

```
(collapse) $ python run_benchmarks.py
```

and letting the scripts run (this takes some time). Regarding how long the tests take to run, we split the effort across four machines (one for each dataset, each equipped with NVIDIA A100), and it still took several days.

To view results once the above script has been executed, use the Jupyter notebook called `eval_benchmarks.ipynb`.


## Reference links

Introduction and documentation for conda-forge project.
https://conda-forge.org/docs/user/introduction.html

Uninstallation of conda-related materials.
https://docs.anaconda.com/free/anaconda/install/uninstall/

Getting miniforge.
https://github.com/conda-forge/miniforge

JupyterLab home page.
https://jupyter.org/

PyTorch, local installation.
https://pytorch.org/get-started/locally/
