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
    # print(data1.units)
    # print(data2.units)
    delta = data1 - data2
    if isinstance(delta, common.pint.Quantity):
        return np.linalg.norm(delta.magnitude)
    out = np.linalg.norm(delta)
    return out


def fit_model(x_data, y_data, model, error_model=None, method='Nelder-Mead', tol=1e-6):
    if error_model is None:
        error_model = error_l2norm

    if not isinstance(model, model_classes.CompositeModel):
        raise SolverConfigurationError(f"Solver can only fit composite models, type given is {type(model)}")
    def internal_fit(params, x_vals, y_reference):
        y_model = model.run_optimizer(x_vals, params)
        return error_model(y_model, y_reference)

    initial_array = model.get_initial_array(x_data)

    result = minimize(internal_fit, initial_array, method=method, args=(x_data, y_data), tol=tol)

    return model.get_param_dict_from_array(result.x), result
