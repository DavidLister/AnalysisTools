# generic.py
#
# Holds generic models
# David Lister
# October 2023
#

# Naming convention is:
#   - s_name for single models
#   - c_name for compound models
#   - fm_name for functions that build up the single models
#   - fd_name for the domain functions
#   - fg_name for the initial guess functions

import numpy as np
from .. import model_classes


def fm_linear(x, params):
    m = params['m']
    b = params['b']
    return m*x + b


def fm_lorentz_distribution(x, params):
    # Not centered, that will be done by the fitting
    x0 = params['x0']
    fwhm = params['fwhm']
    fwhm = np.abs(fwhm)
    scale = params['scale']
    scale = np.abs(scale)
    gamma = fwhm/2
    return scale / (np.pi * gamma * (1 + ((x - x0)/gamma)**2))


def fd_all(x, params):
    return np.full(x.shape, True)



s_linear = model_classes.SingleModel(fm_linear, fd_all, ('m', 'b'))
s_lorentz = model_classes.SingleModel(fm_lorentz_distribution, fd_all, ('x0', 'fwhm', 'scale'))
