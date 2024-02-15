'''Setup: generate data, set it in a loader, and return the loader.'''

# External modules.
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

# Internal modules.
from setup.cifar10 import CIFAR10
from setup.cifar100 import CIFAR100
from setup.fashionmnist import FashionMNIST
from setup.svhn import SVHN


###############################################################################


# Functions related to data loaders and generators.

def get_dataloader(dataset_name, dataset_paras, device, verbose=True):
    '''
    This function does the following three things.
    1. Generate data in numpy format.
    2. Convert that data into PyTorch (tensor) Dataset object.
    3. Initialize PyTorch loaders with this data, and return them.
    '''
    
    dp = dataset_paras
    
    # First get the data generators.
    data_tr, data_va, data_te = get_generator(dataset_name=dataset_name,
                                              dataset_paras=dp)
    
    # Generate data, and map from ndarray to tensor, sharing memory.
    X_tr, Y_tr = map(torch.from_numpy, data_tr())
    X_va, Y_va = map(torch.from_numpy, data_va())
    X_te, Y_te = map(torch.from_numpy, data_te())
    
    # Do a dtype check.
    if verbose:
        print("dtypes (tr): {}, {}".format(X_tr.dtype, Y_tr.dtype))
        print("dtypes (va): {}, {}".format(X_va.dtype, Y_va.dtype))
        print("dtypes (te): {}, {}".format(X_te.dtype, Y_te.dtype))
    
    # Organize tensors into PyTorch dataset objects.
    Z_tr = TensorDataset(X_tr.to(device), Y_tr.to(device))
    Z_va = TensorDataset(X_va.to(device), Y_va.to(device))
    Z_te = TensorDataset(X_te.to(device), Y_te.to(device))
    
    # Prepare the loaders to be returned.
    dl_tr = DataLoader(Z_tr, batch_size=dp["bs_tr"], shuffle=True)
    eval_dl_tr = DataLoader(Z_tr, batch_size=len(X_tr), shuffle=False)
    eval_dl_va = DataLoader(Z_va, batch_size=len(X_va), shuffle=False)
    eval_dl_te = DataLoader(Z_te, batch_size=len(X_te), shuffle=False)

    return (dl_tr, eval_dl_tr, eval_dl_va, eval_dl_te)


def get_generator(dataset_name, dataset_paras):
    
    dp = dataset_paras
    rg = dp["rg"]
    
    # Prepare the data generators for the dataset specified.
    if dataset_name == "cifar10":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = CIFAR10(
            rg=rg, tr_frac=dp["tr_frac"],
            noise_frac=dp["noise_frac"], clean_test=True,
            imbalance_factor=dp["imbalance_factor"],
            num_minority_classes=dp["num_minority_classes"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr)
        data_va = lambda : (X_va, Y_va)
        data_te = lambda : (X_te, Y_te)
    
    elif dataset_name == "cifar100":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = CIFAR100(
            rg=rg, tr_frac=dp["tr_frac"],
            noise_frac=dp["noise_frac"], clean_test=True,
            imbalance_factor=dp["imbalance_factor"],
            num_minority_classes=dp["num_minority_classes"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr)
        data_va = lambda : (X_va, Y_va)
        data_te = lambda : (X_te, Y_te)
    
    elif dataset_name == "fashionmnist":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = FashionMNIST(
            rg=rg, tr_frac=dp["tr_frac"], flatten=True,
            noise_frac=dp["noise_frac"], clean_test=True,
            imbalance_factor=dp["imbalance_factor"],
            num_minority_classes=dp["num_minority_classes"]
        )() # note call
        data_tr = lambda : (X_tr, Y_tr)
        data_va = lambda : (X_va, Y_va)
        data_te = lambda : (X_te, Y_te)

    elif dataset_name == "svhn":
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = SVHN(
            rg=rg, tr_frac=dp["tr_frac"],
            noise_frac=dp["noise_frac"], clean_test=True,
            imbalance_factor=dp["imbalance_factor"],
            num_minority_classes=dp["num_minority_classes"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr)
        data_va = lambda : (X_va, Y_va)
        data_te = lambda : (X_te, Y_te)
        
    else:
        raise ValueError("Unrecognized dataset name.")
    
    return (data_tr, data_va, data_te)


###############################################################################
