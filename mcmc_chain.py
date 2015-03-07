import sys
import textwrap
import numpy as np
import utils


class MCMCChain(object):

    def __init__(self, name, chain_files, burn_in=None,
                 mult_column=0, lnlike_column=1, first_par_column=2,
                 paramname_file=None, params_in_header=False):
        self.rename(name)
        self.chain_files = chain_files
        first_file = True
        for chain_file in chain_files:
            # does this raise an error if the file doesn't exist???
            reader = utils.open_if_exists(chain_file, 'r')

            new_samples = np.loadtxt(reader)
            reader.close()
            if first_file:
                self.samples = np.copy(new_samples)
                first_file = False
            else:
                # check that number of columns are the same

                self.samples = np.vstack((self.samples, new_samples))
        if mult_column is None:
            self.mult_column = 0
            self.multiplicity = np.ones(len(self.samples))
            self.samples = np.vstack((self.multiplicity, self.samples.T)).T
            if lnlike_column is not None:
                self.lnlike_column = lnlike_column + 1
            else:
                self.lnlike_column = None
            self.first_par_column = first_par_column + 1            
        else:
            self.mult_column = mult_column
            self.multiplicity = self.samples[:,mult_column]
            self.lnlike_column = lnlike_column
            self.first_par_column = first_par_column
        if (paramname_file is None) and (not params_in_header):
            paramname_file = '_'.join(chain_files[0].split('_')[:-1]) + \
                '.paramnames'
        self.parameters = self.get_parameter_names(paramname_file, 
                                                   params_in_header)
        self.column_names = list(self.parameters)
        self.column_names.insert(mult_column, 'mult')
        if lnlike_column is not None:
            self.column_names.insert(lnlike_column, '-ln(L)')
        self.burn_in = burn_in
        if burn_in is not None:
            self.remove_burn_in_samples()

    def get_parameter_names(self, paramname_file, params_in_header=False):
        if (paramname_file is None) and params_in_header:
            reader = utils.open_if_exists(self.chain_files[0], 'r')
            parameters = reader.readline().strip('#').split()
            reader.close()
        else:
            paramname_reader = utils.open_if_exists(paramname_file, 'r')
            lines = paramname_reader.readlines()
            paramname_reader.close()
            parameters = []
            for line in lines:
                # remove trailing * used to denote derived parameters
                parameters.append( line.strip().split()[0].split('*')[0] )
        return parameters

    def display_parameters(self):
        print textwrap.fill(', '.join(self.parameters))

    def remove_burn_in_samples(self):
        if 0. <= self.burn_in < 1.:
            n_burn = self.burn_in * np.sum(self.multiplicity)
        elif self.burn_in >= 1.:
            n_burn = int(self.burn_in)
        else:
            n_burn = 0.
            print 'Invalid burn-in value: ', self.burn_in
            print 'Using full chain.'
        # delta is negative for samples to be removed;
        # 1st positive value is the multiplicity for the 1st retained sample
        delta = np.cumsum(self.multiplicity) - n_burn
        burn_index = np.where(delta > 0)[0][0]
        self.multiplicity = self.multiplicity[burn_index:]
        self.multiplicity[0] = delta[burn_index]
        self.samples = self.samples[burn_index:,:]
        self.samples[0,self.mult_column] = delta[burn_index]
        
    def thin(self, thinning_factor):
        # would be more accurate to account for varying multiplicities
        self.samples = self.samples[::thinning_factor,:]
        self.multiplicity = self.multiplicity[::thinning_factor]

    def importance_sample(self, likelihood, 
                          invert=False, print_status=False):
        # *********************************************************
        # need to update anything else that depends on multiplicity
        # after running this
        # *********************************************************
        parameter_values = {}
        n_samples = len(self.multiplicity)
        for i in range(n_samples):
            for p in likelihood.parameters.keys():
                j = np.where(np.array(self.parameters) == p)[0][0] + \
                    self.first_par_column
                parameter_values[p] = self.samples[i, j]
            chisq = likelihood.chi_squared(**parameter_values)
            if invert:
                chisq = -chisq
            self.multiplicity[i] *= np.exp(-0.5*chisq)
            self.samples[i,self.mult_column] = self.multiplicity[i]
            if self.lnlike_column is not None:
                lnl = self.samples[i,self.lnlike_column]
                self.samples[i,self.lnlike_column] = np.sign(lnl) * \
                    (np.abs(lnl) + 0.5*chisq)
            if print_status:
                print '    sample', i+1, 'of', n_samples, \
                    ': chi squared =', chisq,
                sys.stdout.flush()
                print '\r',

    def extend(self, parameters, priors):
        self.parameters.extend(parameters)
        n_samples = len(self.multiplicity)
        columns = []
        for pr in priors:
            columns.append(np.random.random(n_samples)*(pr[1]-pr[0]) + pr[0])
        self.samples = np.vstack([self.samples.T] + columns).T

    def rename(self, new_name):
        # check that name is unique

        self.name = new_name


class PriorChain(MCMCChain):

    def __init__(self, parameters, priors, n_samples=100000):
        self.rename(None)
        self.parameters = list(parameters)

        columns = [np.ones(n_samples), np.zeros(n_samples)]
        for pr in priors:
            columns.append(np.random.random(n_samples)*(pr[1]-pr[0]) + pr[0])
        self.samples = np.vstack(columns).T

        self.mult_column = 0
        self.multiplicity = np.ones(n_samples)
        self.lnlike_column = 1
        self.first_par_column = 2

        self.column_names = list(self.parameters)
        self.column_names.insert(self.mult_column, 'mult')
        self.column_names.insert(self.lnlike_column, '-ln(L)')


class ChainParameter(object):

    def __init__(self, chain, index):
        self.chain = chain
        self.column = index + chain.first_par_column
        self.values = chain.samples[:,self.column]
        # make arrays of sorted values, and 
        # cumulative fraction of multiplicities sorted by values
        sorted_values_and_mult = zip(*np.sort(
                np.array(zip(self.values, self.chain.multiplicity), 
                         dtype=[('v', float), ('m', float)]), 
                order='v'
                )
                                      )
        self.sorted_values = np.array(sorted_values_and_mult[0])
        sorted_mult = np.array(sorted_values_and_mult[1])
        self.mult_fraction = np.cumsum(sorted_mult)/np.sum(sorted_mult)

    def mean(self):
        return np.average(self.values, weights=self.chain.multiplicity)

    def median(self):
        index = np.abs(self.mult_fraction-0.5).argmin()
        return self.sorted_values[index]

    def variance(self):
        return np.average((self.values - self.mean())**2, \
                              weights=self.chain.multiplicity)

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def upper_limit(self, percent_cl):
        index = np.abs(self.mult_fraction-0.01*percent_cl).argmin()
        return self.sorted_values[index]

    def lower_limit(self, percent_cl):
        index = np.abs(self.mult_fraction-(1.-0.01*percent_cl)).argmin()
        return self.sorted_values[index]

    def equal_tail_limits(self, percent_cl):
        lower_fraction = 0.5*(1.-0.01*percent_cl)
        upper_fraction = 0.5*(1.+0.01*percent_cl)
        lower_index = np.abs(self.mult_fraction-lower_fraction).argmin()
        upper_index = np.abs(self.mult_fraction-upper_fraction).argmin()
        return (self.sorted_values[lower_index], 
                self.sorted_values[upper_index])

    def fraction_less_than(self, value):
        index = np.abs(self.sorted_values-float(value)).argmin()
        return self.mult_fraction[index]

    def fraction_greater_than(self, value):
        return 1.-self.fraction_less_than(value)
