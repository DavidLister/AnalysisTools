# solver.py
#
# General purpose curve-fitting tool
# David Lister
# October 2023
#

from scipy.optimize import minimize
import numpy as np
import model_classes
import common

class SolverConfigurationError(Exception):
    """"Solver Configuration Error"""

def error_l2norm(data1, data2):
    delta = data1 - data2
    if isinstance(delta, common.pint.Quantity):
        return np.linalg.norm(delta.magnitude)
    out = np.linalg.norm(delta)
    return out


def error_l2norm_in_semilogy_space(data1, data2):
    """Experimental"""
    delta = np.log(data1 + 1) - np.log(data2 + 1)
    return np.linalg.norm(delta)


def fit_model(x_data, y_data, model, error_model=None, method='Nelder-Mead', tol=1e-6, max_iteration=1000):
    if error_model is None:
        error_model = error_l2norm

    if not isinstance(model, model_classes.CompositeModel):
        raise SolverConfigurationError(f"Solver can only fit composite models, type given is {type(model)}")
    def internal_fit(params, x_vals, y_reference):
        y_model = model.run_optimizer(x_vals, params)
        return error_model(y_model, y_reference)

    initial_array = model.get_initial_array(x_data)
    print("Initial Guess:")
    for i in range(len(initial_array)):
        print(f"\t{model.parameter_fit_lst[i]}: {initial_array[i]}")

    result = minimize(internal_fit, initial_array,
                      method=method, args=(x_data, y_data), tol=tol, options={"maxiter": max_iteration})

    return model.get_param_dict_from_array(result.x), result
