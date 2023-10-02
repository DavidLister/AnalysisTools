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

import common
import numpy as np
import model_classes


def fm_linear(x, m, b):
    return m*x + b


def fd_all(x):
    return np.full(x.shape, True)


s_linear = model_classes.SingleModel(fm_linear, fd_all, 'm', 'b')
