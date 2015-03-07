import numpy as np


class Likelihood(object):

    def __init__(self, invert=False):
        self.parameters = {}
        self.invert = invert


class Likelihood1DFromFile(Likelihood):

    def load_likelihood(self, parameter, input_filename):
        if parameter not in self.parameters:
            self.parameters[parameter] = {}
        values, likelihood = np.loadtxt(input_filename, unpack=True)
        self.parameters[parameter]['values'] = values
        self.parameters[parameter]['likelihood'] = likelihood

    def chi_squared(self, **kwargs):
        if set(kwargs.keys()) != set(self.parameters.keys()):
            print 'Parameter names do not match.'
            return None
        chisq = 0.0
        for parameter in self.parameters:
            if not np.all(np.diff(self.parameters[parameter]['values']) > 0):
                print 'Likelihood for ' + str(parameter) + \
                    ' not in increasing order of parameter values.'
                return None
            else:
                likelihood = np.interp(kwargs[parameter], \
                                   self.parameters[parameter]['values'], \
                                   self.parameters[parameter]['likelihood'])
                chisq += -2.*np.log(likelihood)
        if self.invert:
            chisq = -chisq
        return chisq


class GaussianLikelihood(Likelihood):

    def set_parameter_means(self, **kwargs):
        for key in kwargs:
            if key not in self.parameters:
                self.parameters[key] = {}
            self.parameters[key]['mean'] = kwargs[key]

    def set_covariance(self, cov_array, parameter_order):
        if len(cov_array) != len(self.parameters) or \
                len(cov_array) != len(parameter_order):
            print 'Covariance matrix size not equal to number of parameters.'
        elif set(parameter_order) != set(self.parameters.keys()):
            print 'Parameter names do not match.'
        elif not np.array_equal(cov_array, np.transpose(cov_array)):
            print 'Covariance matrix is not symmetric.'
        elif np.linalg.det(cov_array) <= 0.0:
            print 'Covariance matrix must have positive determinant.'
        else:
            inv_cov_array = np.linalg.inv(cov_array)
            for i, par1 in enumerate(parameter_order):
                self.parameters[par1]['inverse_covariance'] = {}
                for j, par2 in enumerate(parameter_order):
                    self.parameters[par1]['inverse_covariance'][par2] = \
                        inv_cov_array[i,j]

    def chi_squared(self, **kwargs):
        if set(kwargs.keys()) != set(self.parameters.keys()):
            print 'Parameter names do not match.'
            return None
        chisq = 0.0
        for par1 in self.parameters:
            for par2 in self.parameters[par1]['inverse_covariance']:
                chisq += (kwargs[par1] - self.parameters[par1]['mean']) * \
                         (kwargs[par2] - self.parameters[par2]['mean']) * \
                         self.parameters[par1]['inverse_covariance'][par2]
        if self.invert:
            chisq = -chisq
        return chisq


class GaussianSNLikelihood(GaussianLikelihood):
    # subclass for supernova Ia constraints
    # includes analytic marginalization over nuisance parameter

    def set_covariance(self, cov_array, parameter_order):
        if len(cov_array) != len(self.parameters) or \
                len(cov_array) != len(parameter_order):
            print 'Covariance matrix size not equal to number of parameters.'
        elif set(parameter_order) != set(self.parameters.keys()):
            print 'Parameter names do not match.'
        elif not np.array_equal(cov_array, np.transpose(cov_array)):
            print 'Covariance matrix is not symmetric.'
        elif np.linalg.det(cov_array) <= 0.0:
            print 'Covariance matrix must have positive determinant.'
        else:
            inv_cov_array = np.linalg.inv(cov_array)
            for i, par1 in enumerate(parameter_order):
                self.parameters[par1]['inverse_covariance'] = {}
                for j, par2 in enumerate(parameter_order):
                    self.parameters[par1]['inverse_covariance'][par2] = \
                        inv_cov_array[i,j]
            self.inv_cov_sum = np.sum(inv_cov_array)

    def chi_squared(self, **kwargs):
        if set(kwargs.keys()) != set(self.parameters.keys()):
            print 'Parameter names do not match.'
            return None
        chisq = 0.
        chisq_offset = 0.
        for par1 in self.parameters:
            for par2 in self.parameters[par1]['inverse_covariance']:
                dp1 = (kwargs[par1] - self.parameters[par1]['mean'])
                dp2 = (kwargs[par2] - self.parameters[par2]['mean'])
                inv_cov = self.parameters[par1]['inverse_covariance'][par2]
                chisq += dp1 * dp2 * inv_cov
                chisq_offset += dp2 * inv_cov
        chisq -= chisq_offset**2 / self.inv_cov_sum
        if self.invert:
            chisq = -chisq
        return chisq
