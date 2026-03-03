import numpy as np


class LikelihoodParameter(object):
    """Base class for likelihood parameter"""

    def __init__(
        self,
        name,
        min_value,
        max_value,
        ini_value=None,
        value=None,
        Gauss_priors_width=None,
        fixed=False,
    ):
        """Base class for parameter used in likelihood"""
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.Gauss_priors_width = Gauss_priors_width
        self.fixed = False
        if ini_value is not None:
            self.ini_value = ini_value
        return

    def value_in_cube(self):
        """Normalize parameter value to [0,1]."""
        assert self.value is not None, "value not set in parameter " + self.name
        return (self.value - self.min_value) / (self.max_value - self.min_value)

    def get_value_in_cube(self, value):
        """Normalize parameter value to [0,1]."""
        return (value - self.min_value) / (self.max_value - self.min_value)

    def set_from_cube(self, x):
        """Set parameter value from value in cube [0,1]."""
        value = self.value_from_cube(x)
        self.value = value
        return

    def set_without_cube(self, value):
        """Set parameter value without cube"""
        ## Check to make sure parameter is within min/max
        assert self.min_value < value < self.max_value, (
            "Parameter name: %s" % self.name
        )
        self.value = value
        return

    def info_str(self, all_info=False):
        """Return a string with parameter name and value, for debugging"""

        info = self.name + " = " + str(self.value)
        if all_info:
            info += " , " + str(self.min_value) + " , " + str(self.max_value)

        return info

    def value_from_cube(self, x):
        """Given the value in range (xmin,xmax), return absolute value"""

        return self.min_value + x * (self.max_value - self.min_value)

    def err_from_cube(self, err):
        """Return scaled covariance"""

        return err * (self.max_value - self.min_value)

    def get_new_parameter(self, value_in_cube):
        """Return copy of parameter, with updated value from cube"""

        par = LikelihoodParameter(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            Gauss_priors_width=self.Gauss_priors_width,
        )
        par.set_from_cube(value_in_cube)

        return par

def likeparam_from_dict(params_dict):
    # takes a dictionary of parameter names and values, and returns a list of LikelihoodParameter objects
    like_params = []
    for name, value in params_dict.items():
        like_params.append(LikelihoodParameter(name=name, min_value=-1000, max_value=1000, value=value))
        
    return like_params

def dict_from_likeparam(like_params):
    # takes a list of LikelihoodParameter objects, and returns a dictionary of parameter names and values
    params_dict = {}
    for param in like_params:
        params_dict[param.name] = param.value
        
    return params_dict


def format_like_params_dict(iz_choice, param_dict):
    """ Ensure that the parameters are consistent with param_{iz} format"""
    formatted_param_dict = {}
    for iz in range(20): # assume there will never be more than 20 redshift bins
        for key in param_dict.keys():
            if key.endswith(f"_{iz}"):
                formatted_param_dict[key] = param_dict[key]
            elif iz_choice.size == 1:
                # if there is only 1 redshift bin, allow parameters without _{iz} format, and apply to the single redshift bin
                formatted_param_dict[key+f"_{iz_choice[0]}"] = param_dict[key]
            else:
                print("Warning: parameter", key, "not in correct format, must end in _{integer} to specify redshift bin when evaluating multiple redshift bins. This parameter will be ignored.")

    return formatted_param_dict

def par_index(like_param_list, par_name):
    """ Return the index of the parameter with name par_name in the list of LikelihoodParameter objects like_param_list"""
    for i, param in enumerate(like_param_list):
        if param.name == par_name:
            return i
    raise ValueError("Parameter with name " + par_name + " not found in list of LikelihoodParameter objects.")