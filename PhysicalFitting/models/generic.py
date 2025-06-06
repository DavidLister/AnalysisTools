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
from scipy.interpolate import BSpline


def fm_linear(x, params):
    m = params['m']
    b = params['b']
    return m*x + b


def fm_lorentz_distribution(x, params):
    x0 = params['x0']
    fwhm = params['fwhm']
    fwhm = np.abs(fwhm)
    scale = params['scale']
    scale = np.abs(scale)
    gamma = fwhm/2
    return scale / (1 + ((x - x0)/gamma)**2)


def fm_lorentz_distribution_normalized(x, params):
    x0 = params['x0']
    fwhm = params['fwhm']
    fwhm = np.abs(fwhm)
    scale = params['scale']
    scale = np.abs(scale)
    gamma = fwhm/2
    return scale / (np.pi * gamma * (1 + ((x - x0)/gamma)**2))


def fm_gaussian_distribution(x, params):
    x0 = params['x0']
    stdev = params['stdev']
    stdev = np.abs(stdev)
    scale = params['scale']
    scale = np.abs(scale)
    return scale * np.exp(-(x - x0)**2/(2 * stdev**2))



def fm_gaussian_distribution_normalized(x, params):
    x0 = params['x0']
    stdev = params['stdev']
    stdev = np.abs(stdev)
    scale = params['scale']
    scale = np.abs(scale)
    return scale * (1/(np.sqrt(2 * np.pi * stdev**2))) * np.exp(-(x - x0)**2/(2 * stdev**2))


def fm_spline(x, params): ## Doesn't work, needs the parameter tracking system to be expanded to allow arrays.
    bounds = params['bounds']
    n = params['n_points']
    c_arr = params['c_arr']
    k = params['order']

    spacing = (bounds[1] - bounds[0]) / (n - 1)
    t = np.linspace(bounds[0] - spacing*(k-1), bounds[1] + spacing*(k-1), int(n + k + 1))
    spline = BSpline(t, c_arr, k)
    return spline(x)


# Todo - Clean this up!
def fm_spline_5pt_cubic(x, params):  # Kludge for speed of testing.
    bounds = params['bounds']
    c_0 = params['c_0']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_3 = params['c_3']
    c_4 = params['c_4']
    c_arr = [c_0, c_1, c_2, c_3, c_4]
    n = 5
    k = 3

    spacing = (bounds[1] - bounds[0]) / (n - 1)
    t = np.linspace(bounds[0] - spacing*(k-1), bounds[1] + spacing*(k-1), int(n + k + 1))
    spline = BSpline(t, c_arr, k)
    return spline(x)

def fm_spline_7pt_cubic(x, params):  # Kludge for speed of testing.
    bounds = params['bounds']
    c_0 = params['c_0']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_3 = params['c_3']
    c_4 = params['c_4']
    c_5 = params['c_5']
    c_6 = params['c_6']
    c_arr = [c_0, c_1, c_2, c_3, c_4, c_5, c_6]
    n = 7
    k = 3

    spacing = (bounds[1] - bounds[0]) / (n - 1)
    t = np.linspace(bounds[0] - spacing*(k-1), bounds[1] + spacing*(k-1), int(n + k + 1))
    spline = BSpline(t, c_arr, k)
    return spline(x)

def fm_spline_9pt_cubic(x, params):  # Kludge for speed of testing.
    bounds = params['bounds']
    c_0 = params['c_0']
    c_1 = params['c_1']
    c_2 = params['c_2']
    c_3 = params['c_3']
    c_4 = params['c_4']
    c_5 = params['c_5']
    c_6 = params['c_6']
    c_7 = params['c_7']
    c_8 = params['c_8']
    c_arr = [c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8]
    n = 7
    k = 3

    spacing = (bounds[1] - bounds[0]) / (n - 1)
    t = np.linspace(bounds[0] - spacing*(k-1), bounds[1] + spacing*(k-1), int(n + k + 1))
    spline = BSpline(t, c_arr, k)
    return spline(x)


def fd_all(x, params):
    return np.full(x.shape, True)



s_linear = model_classes.SingleModel(fm_linear, fd_all, ('m', 'b'))
s_lorentz = model_classes.SingleModel(fm_lorentz_distribution, fd_all, ('x0', 'fwhm', 'scale'))
s_lorentz_nomralized = model_classes.SingleModel(fm_lorentz_distribution_normalized, fd_all, ('x0', 'fwhm', 'scale'))
s_gaussian = model_classes.SingleModel(fm_gaussian_distribution, fd_all, ('x0', 'stdev', 'scale'))
s_gaussian_normalized = model_classes.SingleModel(fm_gaussian_distribution_normalized, fd_all, ('x0', 'stdev', 'scale'))
s_spline = model_classes.SingleModel(fm_spline, fd_all, ('bounds', 'n_points', 'c_arr', 'order'))
s_spline_5pt_cubic = model_classes.SingleModel(fm_spline_5pt_cubic, fd_all, ('bounds', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4'))
s_spline_7pt_cubic = model_classes.SingleModel(fm_spline_7pt_cubic, fd_all, ('bounds', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6'))
s_spline_9pt_cubic = model_classes.SingleModel(fm_spline_9pt_cubic, fd_all, ('bounds', 'c_0', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5', 'c_6', 'c_7', 'c_8'))
