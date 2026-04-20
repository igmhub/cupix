import numpy as np


class FreeParameter(object):
    """Basic class describing a free parameter"""

    def __init__(self,
        name,
        min_value=None,
        max_value=None,
        ini_value=None,
        true_value=None,
        delta=None,
        gauss_prior_mean=None,
        gauss_prior_width=None,
        latex_label=None
    ):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.ini_value = ini_value
        self.true_value = true_value
        self.delta = delta
        self.gauss_prior_mean = gauss_prior_mean
        self.gauss_prior_width = gauss_prior_width
        if latex_label is not None:
            self.latex_label = latex_label
        else:
            self.latex_label = self.name

        return


    def out_of_bounds(self, value):
        """Return True if value out of bounds"""
        if value < self.min_value:
            return True
        if value > self.max_value:
            return True
        return False


    def get_prior_chi2(self, value):
        # if both are None, return 0
        if self.gauss_prior_mean is None and self.gauss_prior_width is None:
            return 0.0
        # if one is not None, the other shouldn't
        assert self.gauss_prior_mean is not None
        assert self.gauss_prior_width is not None
        return ( (value-self.gauss_prior_mean) / self.gauss_prior_width)**2


