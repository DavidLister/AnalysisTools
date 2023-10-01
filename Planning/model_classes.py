# model_classes.py
#
# A utility package for parsing model dictionaries into objects that can be used internally.
# David Lister
# October 2023
#
import numpy as np


class SingleModel:
    def __init__(self, model_function, range_function, parameter_map):
        """
        Initializes a class to hold a model.
        :param model_function: Must be of the form mfunc(x, parameter_dict)
        :param range_function: Of the form rfunc(x, parameter_dict) and returns a mask for x, corresponding to the valid domain of the model
        :param parameter_map: Tuple of parameter dict keys for the model.
        """
        self.model_function = model_function
        self.range_function = range_function
        self.parameter_map = parameter_map
        self.n_params = len(self.parameter_map)

    def run(self, x_data, parameter_dict, domain_restriction=None):
        """
        Runs the model and returns a y-value array for each x value.
        :param x_data: x values to run the model. Will be zero for any values outside of domain.
        :param domain_restriction: tuple of the form (min x, max x) corresponding to the open interval to restrict the domain, if allowed by the model. np.inf is valid input
        :return: np array of y values that the model generates.
        """

        domain = self.range_function(x_data, parameter_dict)
        if domain_restriction is not None:
            restricted_domain = np.logical_and(x_data > domain_restriction[0], x_data < domain_restriction[1])
            domain = np.logical_and(domain, restricted_domain)

        out = np.zeros(x_data.shape)
        out[domain] = self.model_function(x_data[domain], parameter_dict)

        return out



class CompositeModel:
    def __init__(self):
        pass

