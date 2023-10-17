# model_classes.py
#
# A utility package for parsing model dictionaries into objects that can be used internally.
# David Lister
# October 2023
#
import numpy as np
import common

class ModelDefinitionError(Exception):
    """"Model Definition Error"""

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
    def __init__(self, composite_model_dict):
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

        keys = self.raw_composite_model_dict.keys()

        for key in keys:
            if key == common.FIT_PARAMETERS:
                for param in self.raw_composite_model_dict[common.FIT_PARAMETERS].keys():
                    self.parameter_classification_dict[param] = common.FIT_PARAMETERS
                    self.parameter_fit_initial_lookup[param] = self.raw_composite_model_dict[common.FIT_PARAMETERS][param]
                    self.parameter_fit_lst.append(param)

            elif key == common.FIXED_PARAMETERS:
                for param in self.raw_composite_model_dict[common.FIXED_PARAMETERS].keys():
                    self.parameter_classification_dict[param] = common.FIXED_PARAMETERS
                    self.parameter_fixed_lookup[param] = self.raw_composite_model_dict[common.FIXED_PARAMETERS][param]

            else:  # it's a model!
                # Register model into model dict
                self.model_dict[key] = self.raw_composite_model_dict[key]

                # Verify that all required parameters are defined
                if self.model_dict[key].n_params != len(self.model_dict[key][common.PARAMETERS]):
                    raise ModelDefinitionError(f"The number of parameters given doesn't match {key} model requirements.")

                # Register model parameters in LUT
                for model_param in self.model_dict[common.PARAMETERS].keys():
                    param_name = self.model_dict[common.PARAMETERS][model_param]
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

    def run(self, x_data, parameter_dict):
        """
        Run the model given x-data and a dictionary with all the required parameters.
        :param x_data: np array of the x-data to run the model over.
        :param parameter_dict: Dictionary containing all the model fit parameters.
        :return: np array of the y data, in the same shape as x.
        """
        y_data = np.zeros(x_data.shape)
        for model in self.model_dict.keys():
            submodel_parameters = {sub_param: parameter_dict[self.model_dict[model][common.PARAMETERS][sub_param]] for sub_param in self.model_dict[model][common.PARAMETERS].keys()}
            y_data = y_data + self.model_dict[model][common.MODEL].run(x_data, submodel_parameters, domain_restriction=self.model_dict[model][common.MODEL][common.DOMAIN])

        return y_data

    def get_param_dict_from_array(self, parameter_array):
        parameter_dict = {}
        for i in range(len(self.parameter_fit_lst)):
            parameter_dict[self.parameter_fit_lst[i]] = parameter_array[i]

        return {**parameter_dict, **self.parameter_fixed_lookup}

    def run_optimizer(self, x_data, parameter_array):
        parameter_dict = self.get_param_dict_from_array(parameter_array)
        return self.run(x_data, parameter_dict)

    def get_initial_array(self, x_values):
        """
        Returns a numpy array of initial parameters
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
                        values.append(self.model_dict[model].initial_guess_map[parameter](x_values))
                if len(values) > 0:
                    out.append(np.mean(values))
                else:
                    print(f"Warning, no initial guess function for {parameter}")
                    out.append(1.0)

            else:
                out.append(self.parameter_fit_initial_lookup[parameter])

        return np.array(out)

    def get_parameter_array_order(self):
        return self.parameter_fit_lst
