# model_classes.py
#
# A utility package for parsing model dictionaries into objects that can be used internally.
# David Lister
# October 2023
#
import numpy as np
import common
from scipy.optimize import root
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from functools import partial


def logistic_function(x, lower, upper):
    return (upper - lower) / (1 + np.exp(-x)) + lower


def identity(x):
    return x


def robust_solve(func, target, test_range=5):
    """Robustly solve a single-valued function f(x)=y. Assumes domain is soft-constrained."""
    x_test = np.linspace(-test_range, test_range, 100)
    error = func(x_test) - target
    error = np.abs(error)
    x_initial_guess = x_test[list(error).index(np.min(error))]
    sol = root(lambda x: func(x) - target, np.array(x_initial_guess))

    if common.DEBUG:
        print("Robust solver: ", func, target, test_range)
        print(sol)
        print('\n\n\n')

    if abs(sol.x) < 0.1:
        return np.array(0.1)  # small numbers near zero can mess up the minimizer
    return sol.x[0]


def direct_sampling(model, x, params, width=None):
    return model(x, params)


def normal_distribution(x, mu, sigma):
    dist = np.exp((-1/2) * ((x - mu)/sigma)**2)
    return dist * (1/np.sum(dist))


def gaussian_AA10X(model, x, params, width=None):
    """Samples the model with a resolution of 10x the median x spacing and convolves
       with a gaussian with a sigma equal to the median x spacing."""
    delta_x = np.median(x[1:] - x[:-1])
    oversample = 10
    x_oversample = np.arange(min(x), max(x) + delta_x/(oversample-1), delta_x/oversample)
    if width is None:
        width = delta_x/2

    x_kernel = np.arange(-width*5, width*5 + delta_x/(oversample-1), delta_x/oversample)
    kernel = normal_distribution(x_kernel, 0, width)

    raw_samples = model(x_oversample, params)
    convolved_samples = fftconvolve(raw_samples, kernel, 'same')
    interpolator = interp1d(x_oversample, convolved_samples)
    return interpolator(x)



class ModelDefinitionError(Exception):
    """"Model Definition Error"""


class ModelConfigurationError(Exception):
    """Model Configuration Error"""


class SingleModel:
    def __init__(self, model_function, domain_function, parameter_map, initial_guess_map=None):
        """
        Initializes a class to hold a model.
        :param model_function: Must be of the form mfunc(x, parameter_dict)
        :param domain_function: Of the form rfunc(x, parameter_dict) and returns a mask for x, corresponding to the valid domain of the model
        :param parameter_map: Tuple of parameter dict keys for the model.
        :param initial_guess_map: Dict of initial guess generating functions. Format is param:funct(x_values).
        """
        self.model_function = model_function
        self.domain_function = domain_function
        self.parameter_map = parameter_map
        self.n_params = len(self.parameter_map)
        if initial_guess_map is None:
            self.initial_guess_map = {}
        else:
            self.initial_guess_map = initial_guess_map

    def run(self, x_data, parameter_dict, domain_restriction=None):
        """
        Runs the model and returns a y-value array for each x value.
        :param x_data: x values to run the model. Will be zero for any values outside of domain.
        :param parameter_dict: Dictionary of parameters for the model, free and fixed
        :param domain_restriction: tuple of the form (min x, max x) corresponding to the open interval to restrict the domain, if allowed by the model. np.inf is valid input
        :return: np array of y values that the model generates.
        """

        domain = self.domain_function(x_data, parameter_dict)
        if domain_restriction is not None:
            restricted_domain = np.logical_and(x_data > domain_restriction[0], x_data < domain_restriction[1])
            domain = np.logical_and(domain, restricted_domain)

        out = np.zeros(x_data.shape)
        out[domain] = self.model_function(x_data[domain], parameter_dict)

        return out


class CompositeModel:
    def __init__(self, composite_model_dict, domain_restriction=None):
        """
        Builds a composite model class from a composite model dictionary.
        :param composite_model_dict: A composite model dictionary definition. See initial_model in "Planning/use_cases.py" for an example.
        """
        self.raw_composite_model_dict = composite_model_dict
        self.model_dict = {}

        self.parameter_name_to_model = {}  # Maps composite model parameters to which sub-model they belong to
        self.parameter_classification_dict = {}  # Holds all the parameter names and their types
        self.parameter_fixed_lookup = {}  # LUT to hold fixed values
        self.parameter_fit_initial_lookup = {}  # LUT to hold initial guess for fit parameters
        self.parameter_fit_lst = []  # List that defines parameter order that is used by minimizer
        self.optimizer_to_internal_mapping = {}
        self.internal_to_model_mapping = {}
        self.domain_restriction = domain_restriction

        keys = self.raw_composite_model_dict.keys()

        for key in keys:
            if key == common.FIT_PARAMETERS:
                for param in self.raw_composite_model_dict[common.FIT_PARAMETERS].keys():
                    self.parameter_classification_dict[param] = common.FIT_PARAMETERS
                    self.parameter_fit_lst.append(param)
                    if isinstance(self.raw_composite_model_dict[common.FIT_PARAMETERS][param], tuple):
                        constraints = self.raw_composite_model_dict[common.FIT_PARAMETERS][param]
                        self.parameter_fit_initial_lookup[param] = constraints[0]
                        match len(constraints):
                            case 3:
                                self.optimizer_to_internal_mapping[param] = partial(logistic_function, lower=constraints[1], upper=constraints[2])
                                self.internal_to_model_mapping[param] = identity

                            case 4:
                                self.internal_to_model_mapping[param] = constraints[3]
                                internal_low = robust_solve(constraints[3], constraints[1], test_range=50)
                                internal_high = robust_solve(constraints[3], constraints[2], test_range=50)
                                self.optimizer_to_internal_mapping[param] = partial(logistic_function, lower=internal_low, upper=internal_high)


                            case _:
                                raise ModelDefinitionError(f"Fit parameter initial guess not defined properly for {param}")
                    else:
                        self.parameter_fit_initial_lookup[param] = self.raw_composite_model_dict[common.FIT_PARAMETERS][param]
                        self.optimizer_to_internal_mapping[param] = identity
                        self.internal_to_model_mapping[param] = identity

            elif key == common.FIXED_PARAMETERS:
                for param in self.raw_composite_model_dict[common.FIXED_PARAMETERS].keys():
                    self.parameter_classification_dict[param] = common.FIXED_PARAMETERS
                    self.parameter_fixed_lookup[param] = self.raw_composite_model_dict[common.FIXED_PARAMETERS][param]

            else:  # it's a model!
                # Register model into model dict
                self.model_dict[key] = self.raw_composite_model_dict[key]

                # Verify that all required parameters are defined
                if self.model_dict[key][common.MODEL].n_params != len(self.model_dict[key][common.PARAMETERS]):
                    raise ModelDefinitionError(f"The number of parameters given doesn't match {key} model requirements.")

                # Register model parameters in LUT
                for model_param in self.model_dict[key][common.PARAMETERS].keys():
                    param_name = self.model_dict[key][common.PARAMETERS][model_param]
                    if param_name in self.parameter_name_to_model:
                        self.parameter_name_to_model[param_name].append(key)
                    else:
                        self.parameter_name_to_model[param_name] = [key, ]

                # Set domain to None if it's not defined. If it is defined, check that it's valid
                if common.DOMAIN in self.model_dict[key]:
                    if len(self.model_dict[key][common.DOMAIN]) != 2:
                        raise ModelDefinitionError(f"Domain for {key} has the wrong number of values, should be only 2.")
                    elif self.model_dict[key][common.DOMAIN][0] > self.model_dict[key][common.DOMAIN][1]:
                        raise ModelDefinitionError(f"Domain for {key} is incorrectly defined")

                else:
                    self.model_dict[key][common.DOMAIN] = None

        self.parameter_fit_lst = tuple(self.parameter_fit_lst)  # Make this immutable.

        # Now verify that all the registered parameters are also defined as fixed or free parameters
        for key in self.parameter_name_to_model.keys():
            in_lut = False
            if key in self.parameter_fixed_lookup:
                in_lut = True
            if key in self.parameter_fit_initial_lookup:
                in_lut = True

            if not in_lut:
                raise ModelDefinitionError(f"Model parameter, {key}, not found in top level definition.")

        # Verify all the given parameters are used in at least one model
        for param in self.parameter_fixed_lookup.keys():
            if param not in self.parameter_name_to_model:
                raise ModelDefinitionError(f"Fixed parameter {param} not found in any model definition.")

        for param in self.parameter_fit_initial_lookup.keys():
            if param not in self.parameter_name_to_model:
                raise ModelDefinitionError(f"Fit parameter {param} not found in any model definition.")

        # Raise flag to automatically determine initial guess, if applicable
        self.auto_initial_guess_flag = False
        for param in self.parameter_fit_initial_lookup.keys():
            if self.parameter_fit_initial_lookup[param] == common.AUTO:
                self.auto_initial_guess_flag = True

    def sample(self, x_data, parameter_dict):
        y_data = np.zeros(x_data.shape)
        for model in self.model_dict.keys():
            if self.domain_restriction is not None:
                if self.model_dict[model][common.DOMAIN] is not None:
                    domain = (min(self.domain_restriction[0], self.model_dict[model][common.DOMAIN][0]),
                              max(self.domain_restriction[1], self.model_dict[model][common.DOMAIN][1]))
                else:
                    domain = self.domain_restriction
            else:
                if self.model_dict[model][common.DOMAIN] is not None:
                    domain = self.model_dict[model][common.DOMAIN]
                else:
                    domain = None
            submodel_parameters = {sub_param: parameter_dict[self.model_dict[model][common.PARAMETERS][sub_param]] for sub_param in self.model_dict[model][common.PARAMETERS].keys()}
            y_data = y_data + self.model_dict[model][common.MODEL].run(x_data, submodel_parameters, domain_restriction=domain)

        return y_data

    def run(self, x_data, parameter_dict, antialiasing=False, width=None):
        """
        Run the model given x-data and a dictionary with all the required parameters.
        :param x_data: np array of the x-data to run the model over.
        :param parameter_dict: Dictionary containing all the model fit parameters.
        :return: np array of the y data, in the same shape as x.
        """

        match antialiasing:
            case False:
                sampler = direct_sampling
            case common.GAUSSIAN_AA10X:
                sampler = gaussian_AA10X
            case _:
                raise ModelConfigurationError(f"Improper anti aliasing configuration, set to {antialiasing}")

        return sampler(self.sample, x_data, parameter_dict, width=width)


    def get_param_dict_from_array(self, parameter_array):
        parameter_dict = {}
        for i in range(len(self.parameter_fit_lst)):
            internal = self.optimizer_to_internal_mapping[self.parameter_fit_lst[i]](parameter_array[i])
            parameter_dict[self.parameter_fit_lst[i]] = self.internal_to_model_mapping[self.parameter_fit_lst[i]](internal)

        return {**parameter_dict, **self.parameter_fixed_lookup}

    def run_optimizer(self, x_data, parameter_array, antialiasing=False):
        parameter_dict = self.get_param_dict_from_array(parameter_array)
        return self.run(x_data, parameter_dict, antialiasing=antialiasing)

    def get_initial_array(self, x_values):
        """
        Returns a numpy array of initial parameters in optimizer-reduced variables
        :param x_values: x_values that it will be fit over. Used for parameters with AUTO parameter flag.
        :return: np array of initial guess values.
        """
        out = []
        for parameter in self.parameter_fit_lst:
            if self.parameter_fit_initial_lookup[parameter] == common.AUTO:
                # Determine automatic initial guess
                values = []
                models = self.parameter_name_to_model[parameter]
                for model in models:
                    if parameter in self.model_dict[model].initial_guess_map:
                        val = robust_solve(self.optimizer_to_internal_mapping[parameter], self.model_dict[model].initial_guess_map[parameter](x_values))
                        values.append(val)
                if len(values) > 0:
                    out.append(np.mean(values))
                else:
                    print(f"Warning, no initial guess function for {parameter}")
                    out.append(1.0)

            else:
                val = robust_solve(self.optimizer_to_internal_mapping[parameter], self.parameter_fit_initial_lookup[parameter])
                out.append(val)
        return np.array(out)

    def get_parameter_array_order(self):
        return self.parameter_fit_lst
