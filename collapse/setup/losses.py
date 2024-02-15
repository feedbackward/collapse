'''Setup: initialize and pass the desired loss function.'''

# External modules.
import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, BCELoss, L1Loss, Parameter
from torch.nn.functional import relu

# Internal modules.
from setup.sunhuber import rho_torch


###############################################################################


# Here we define some customized loss classes.


class Loss_OCElike(Module):
    '''
    Loss class for some (dual-form) OCE-like criteria.
    REF: https://github.com/daniellevy/fast-dro/blob/main/robust_losses.py
    '''
    def __init__(self, crit_name: str, crit_paras: dict,
                 loss_name: str, loss_paras: dict,
                 device, theta_init=0.0):
        super().__init__()
        self.crit_name = crit_name
        self.crit_paras = crit_paras
        if self.crit_name == "CVaR":
            qlevel = self.crit_paras["quantile_level"]
            if qlevel == 0.0:
                self.theta = theta_init # irrelevant in this case
            elif qlevel > 0.0:
                self.theta = Parameter(data=Tensor([theta_init])).to(device)
            else:
                raise ValueError("CVaR only works for non-negative qlevel.")
        elif self.crit_name == "DRO":
            radius = self.crit_paras["radius"]
            if radius == 0.0:
                self.theta = theta_init # irrelevant in this case
            elif radius > 0.0:
                self.theta = Parameter(data=Tensor([theta_init])).to(device)
            else:
                raise ValueError("DRO only works for non-negative radius.")
        else:
            raise ValueError("Please pass a recognized criterion name.")
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None

    def forward(self, input: Tensor, target: Tensor):
        
        # Compute individual losses.
        loss = self.loss_fn(input=input, target=target)
        
        # Process losses based on specified OCE-like criterion.
        if self.crit_name == "CVaR":
            qlevel = self.crit_paras["quantile_level"]
            if qlevel == 0.0:
                return loss.mean()
            elif qlevel > 0.0:
                return self.theta + relu(loss-self.theta).mean()/(1.0-qlevel)
            else:
                raise ValueError("CVaR only works for non-negative qlevel.")
        elif self.crit_name == "DRO":
            radius = self.crit_paras["radius"]
            if radius == 0.0:
                return loss.mean()
            elif radius > 0.0:
                sqd = (1+2*radius)*(relu(loss-self.theta)**2).mean()
                return self.theta + torch.sqrt(sqd)
            else:
                raise ValueError("DRO only works for non-negative radius.")
        else:
            raise ValueError("Please pass a recognized criterion name.")


class Loss_Tilted(Module):
    '''
    Loss class for tilted ERM.
    '''
    def __init__(self, tilt: float, loss_name: str, loss_paras: dict):
        super().__init__()
        self.tilt = tilt
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None
    
    def forward(self, input: Tensor, target: Tensor):
        loss = self.loss_fn(input=input, target=target)
        if self.tilt > 0.0:
            return torch.log(torch.exp(self.tilt*loss).mean()) / self.tilt
        elif self.tilt == 0.0:
            return loss.mean()
        else:
            raise ValueError("Only defined for non-negative tilt values.")


class Loss_Flood(Module):
    '''
    General purpose loss class for the "flooding" algorithm
    of Ishida et al. (2020).
    '''
    def __init__(self, flood_level: float, loss_name: str, loss_paras: dict):
        super().__init__()
        self.flood_level = flood_level
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="mean")
        return None

    def forward(self, input: Tensor, target: Tensor):
        fl = self.flood_level
        loss = self.loss_fn(input=input, target=target)
        return (loss-fl).abs()+fl


class Loss_SoftAD(Module):
    '''
    Soft ascent-descent (SoftAD), our most basic modified version of
    the flooding algorithm.
    '''
    def __init__(self, theta: float, sigma: float, eta: float,
                 loss_name: str, loss_paras: dict):
        super().__init__()
        self.theta = theta
        self.sigma = sigma
        self.eta = eta
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None
    
    def forward(self, input: Tensor, target: Tensor):
        theta = self.theta
        sigma = self.sigma + 1e-12 # to be safe.
        eta = self.eta
        loss = self.loss_fn(input=input, target=target)
        dispersion = (sigma**2) * rho_torch((loss-theta)/sigma).mean()
        return eta*theta + dispersion


# Here we define various loss function getters.

def get_loss(loss_name, loss_paras, device):
    '''
    Loss function getter for all methods.
    '''
    ln = loss_name
    lp = loss_paras
    
    if lp["method"] == "ERM":
        loss_fn = get_named_loss(loss_name=ln,
                                 loss_paras=lp,
                                 reduction="mean")
    elif lp["method"] == "Ishida":
        loss_fn = get_flood_loss(loss_name=ln,
                                 loss_paras=lp)
    elif lp["method"] in ["CVaR", "DRO"]:
        lp["crit_name"] = lp["method"]
        loss_fn = get_ocelike_loss(loss_name=ln,
                                   loss_paras=lp,
                                   device=device)
    elif lp["method"] == "SAM":
        loss_fn = get_named_loss(loss_name=ln,
                                 loss_paras=lp,
                                 reduction="mean")
    elif lp["method"] == "SoftAD":
        loss_fn = get_softad_loss(loss_name=ln,
                                  loss_paras=lp)
    elif lp["method"] == "Tilted":
        loss_fn = get_tilted_loss(loss_name=ln,
                                  loss_paras=lp)
    else:
        raise ValueError("Unrecognized method name.")
    
    return loss_fn


def get_flood_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_Flood.
    '''
    fl = loss_paras["flood_level"]
    loss_fn = Loss_Flood(flood_level=fl,
                         loss_name=loss_name,
                         loss_paras=loss_paras)
    return loss_fn


def get_softad_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_SoftAD.
    '''
    eta = loss_paras["eta"]
    sigma = loss_paras["sigma"]
    theta = loss_paras["theta"]
    loss_fn = Loss_SoftAD(theta=theta, sigma=sigma, eta=eta,
                          loss_name=loss_name, loss_paras={})
    return loss_fn


def get_ocelike_loss(loss_name, loss_paras, device):
    '''
    A simple wrapper for Loss_OCElike.
    '''
    crit_name = loss_paras["crit_name"]
    crit_paras = {"quantile_level": loss_paras["quantile_level"],
                  "radius": loss_paras["radius"]}
    loss_fn = Loss_OCElike(crit_name=crit_name, crit_paras=crit_paras,
                           loss_name=loss_name, loss_paras={},
                           theta_init=0.0, device=device)
    return loss_fn


def get_tilted_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_Tilted.
    '''
    loss_fn = Loss_Tilted(tilt=loss_paras["tilt"],
                          loss_name=loss_name, loss_paras={})
    return loss_fn


def get_named_loss(loss_name, loss_paras, reduction):
    
    if loss_name == "CrossEntropy":
        loss_fn = CrossEntropyLoss(reduction=reduction)
    elif loss_name == "BCELoss":
        loss_fn = BCELoss(reduction=reduction)
    elif loss_name == "L1Loss":
        loss_fn = L1Loss(reduction=reduction)
    else:
        raise ValueError("Unrecognized loss name.")
    
    return loss_fn


###############################################################################
