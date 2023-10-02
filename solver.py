# solver.py
#
# General purpose curve-fitting tool
# David Lister
# October 2023
#

from scipy.optimize import minimize
import numpy as np

def error_l2norm(data1, data2):
    # print(data1.units)
    # print(data2.units)
    delta = data1 - data2
    return np.linalg.norm(delta.magnitude)


def fit_model(x_data, y_data, model, error_model=None, method='Nelder-Mead'):
    if error_model is None:
        error_model = error_l2norm

    def internal_fit(params, x_vals, y_vals):
        y_model = model.run_optimizer(x_vals, params)
        return error_model(y_model, y_vals)

    initial_array = model.get_initial_array(x_data)

    result = minimize(internal_fit, initial_array, method=method)

    return model.get_param_dict_from_array(result.x), result
