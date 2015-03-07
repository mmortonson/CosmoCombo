import os.path
import time
import textwrap
import numpy as np
from sympy import symbols, sympify, lambdify
import utils
import cosmology
from menu import Menu
from mcmc_chain import MCMCChain, PriorChain, ChainParameter
from likelihood import GaussianLikelihood, GaussianSNLikelihood


class PostPDF(object):

    def __init__(self, name, model):
        self.settings = {}
        self.rename(name)
        self.model = model
        self.chain = None
        self.chain_files = []
        self.likelihoods = {}
        self.cosmology_parameter_samples = []
        for attr in ['name', 'model', 'chain_files']:
            self.settings[attr] = getattr(self, attr)
        self.settings['parameters'] = {}
        self.settings['likelihoods'] = {}
        self.settings['contour_data_files'] = []
        self.settings['color'] = None
        self.settings['derived_parameters'] = {}
        self.settings['order'] = []
        self.settings['chain_to_cosmology'] = {}

    def has_mcmc(self):
        if len(self.chain_files) > 0:
            return True
        else:
            return False

    # break up into more functions
    def add_chain(self, name, files, burn_in=None, 
                  mult_column=0, lnlike_column=1, first_par_column=2,
                  paramname_file=None, params_in_header=False,
                  update_order=True):
        # check if already have chain, if files exist, if name is unique

        # check if model is consistent?

        self.chain_files.extend(files)
        self.settings['chain_name'] = name
        self.settings['chain_files'] = self.chain_files
        self.settings['chain_burn_in'] = burn_in
        self.settings['chain_mult_column'] = mult_column
        self.settings['chain_lnlike_column'] = lnlike_column
        self.settings['chain_first_par_column'] = first_par_column
        self.settings['chain_paramname_file'] = paramname_file
        self.settings['chain_params_in_header'] = params_in_header
        self.chain = MCMCChain(name, files, burn_in, 
                               mult_column, lnlike_column, first_par_column,
                               paramname_file, params_in_header)

        for pdf_element in self.settings['order']:
            if pdf_element[0] == 'chain':
                # only add likelihoods and/or derived parameters
                # that come before the chain
                break
            elif pdf_element[0] == 'likelihood':
                lk_name = pdf_element[1]
                lk_settings = self.settings['likelihoods'][lk_name]
                lk_parameters = lk_settings['parameters']
                lk_priors = lk_settings['priors']                    
                if lk_name not in self.likelihoods:
                    self.add_likelihood(lk_name, update_order=False,
                                        **lk_settings)
                    # if likelihood(s) already included in constraint
                    # when chain is added, check if likelihood parameters
                    # are also in the chain (if so, rename chain parameters)
                extra_parameters = []
                lk_chain_parameters = []
                for i, p in enumerate(lk_parameters):
                    if 'chain_parameters' in lk_settings:
                        if lk_settings['chain_parameters'][i] is None:
                            extra_parameters.append(p)
                        else:
                            self.chain.parameters[ \
                                self.chain.parameters.index( \
                                    lk_settings['chain_parameters'][i])] = p
                    else:
                        m = Menu(options=self.chain.parameters, 
                                 exit_str='New',
                                 header='If the parameter ' + p + \
                                     ' from likelihood ' + lk_name + \
                                     '\nis in the chain, select the ' + \
                                     'corresponding parameter;\n' + \
                                     'otherwise choose "New":')
                        m.get_choice()
                        if m.choice == m.exit:
                            extra_parameters.append(p)
                            lk_chain_parameters.append(None)
                        else:
                            self.chain.parameters[ \
                                self.chain.parameters.index(m.choice)] = p
                            lk_chain_parameters.append(m.choice)
                if 'chain_parameters' not in lk_settings:
                    lk_settings['chain_parameters'] = \
                        list(lk_chain_parameters)
                # add extra parameters defined in likelihoods
                self.chain.extend(extra_parameters, 
                                  [lk_priors[lk_parameters.index(p)] for \
                                       p in extra_parameters])
                # importance sample
                if lk_settings['form'] != 'Flat':
                    self.chain.importance_sample(self.likelihoods[lk_name])

            # recompute derived parameters if necessary
            elif pdf_element[0] == 'derived_parameter':
                par_name = pdf_element[1]
                par_settings = self.settings['derived_parameters'][par_name]
                if 'chain_parameter' in par_settings:
                    if par_settings['chain_parameter'] is None:
                        self.add_derived_parameter( \
                            par_name, par_settings['function'],
                            par_settings['parameters'], update_order=False)
                    else:
                        self.chain.parameters[ \
                            self.chain.parameters.index( \
                                par_settings['chain_parameter'])] = par_name
                else:
                    m = Menu(options=self.chain.parameters, exit_str='New',
                             header='If the derived parameter ' + \
                                 par_name + \
                                 ' is in the chain,\n' + \
                                 'select the corresponding ' + \
                                 'parameter;\notherwise choose "New":')
                    m.get_choice()
                    if m.choice == m.exit:
                        self.add_derived_parameter( \
                            par_name, par_settings['function'],
                            par_settings['parameters'], update_order=False)
                        par_settings['chain_parameter'] = None
                    else:
                        self.chain.parameters[ \
                            self.chain.parameters.index(m.choice)] = \
                            par_name
                        par_settings['chain_parameter'] = m.choice

        self.add_parameters(self.chain.parameters)
        if update_order:
            self.settings['order'].append(('chain', name))
            self.settings['contour_data_files'] = []

    def add_likelihood(self, name, update_order=True, **kwargs):
        # check if name is unique (not already in self.likelihoods)

        self.settings['likelihoods'][name] = kwargs
        if update_order:
            self.settings['order'].append(('likelihood', name))
            self.settings['contour_data_files'] = []
        kwargs['invert'] = False
        if kwargs['form'] == 'Gaussian':
            self.add_gaussian_likelihood(name, **kwargs)
        elif kwargs['form'] == 'SNe':
            for p in kwargs['parameters']:
                # assumes parameter names end in 'z' + the SN redshift
                sn_z = p.split('z')[-1]
                self.add_derived_parameter(p, 'mu_SN(' + str(sn_z) + ')', 
                                           [], [], update_order=False)
                print '  z = ' + sn_z
            self.add_gaussian_SN_likelihood(name, **kwargs)
        elif kwargs['form'] == 'Inverse Gaussian':
            kwargs['invert'] = True
            self.add_gaussian_likelihood(name, **kwargs)
        
        # if this is a new PDF, create a "chain" with 
        # random samples within the priors
        if self.chain is None:
            self.chain = PriorChain(kwargs['parameters'], kwargs['priors'])
        else:
            # check if there are extra parameters not in the current chain;
            # if so, extend the chain using random samples within priors
            new_parameters = []
            new_priors = []
            for p, p_range in zip(kwargs['parameters'], kwargs['priors']):
                if p not in self.chain.parameters:
                    new_parameters.append(p)
                    new_priors.append(p_range)
            if len(new_parameters) > 0:
                self.chain.extend(new_parameters, new_priors)

        if kwargs['form'] != 'Flat':
            print 'Importance sampling...'
            self.chain.importance_sample(self.likelihoods[name])

        self.add_parameters(kwargs['parameters'])
        for p, p_range in zip(kwargs['parameters'], kwargs['priors']):
            self.settings['parameters'][p] = p_range
        
    def add_gaussian_likelihood(self, name, **kwargs):
        self.likelihoods[name] = GaussianLikelihood(invert=kwargs['invert'])
        self.likelihoods[name].set_parameter_means( \
            **dict(zip(kwargs['parameters'], kwargs['means'])))
        self.likelihoods[name].set_covariance(kwargs['covariance'],
                                              kwargs['parameters'])

    def add_gaussian_SN_likelihood(self, name, **kwargs):
        self.likelihoods[name] = GaussianSNLikelihood(invert=kwargs['invert'])
        self.likelihoods[name].set_parameter_means( \
            **dict(zip(kwargs['parameters'], kwargs['means'])))
        self.likelihoods[name].set_covariance(kwargs['covariance'],
                                              kwargs['parameters'])

    def add_parameters(self, new_parameters):
        for p in new_parameters:
            if p not in self.settings['parameters']:
                self.settings['parameters'][p] = []

    def display_parameters(self, parameters=None):
        if parameters is None:
            parameters = self.settings['parameters'].keys()
        print textwrap.fill(', '.join(parameters))

    def choose_parameters(self, options=None, num=None,
                          allow_extra_parameters=False):
        if options is None:
            options = list(self.settings['parameters'].keys())
        # choose one or more parameters 
        # (enter names or pick from chain column list)
        if num is None:
            print '\nEnter parameter names (all on one line),'
        elif num == 1:
            print '\nEnter a parameter name,'
        else:
            print '\nEnter ' + str(num) + ' parameter names (on one line),'
        print 'or enter "l" to see a list of available parameters:'
        parameters = raw_input('> ').split()
        if len(parameters) == 1 and parameters[0].strip() == 'l':
            self.display_parameters(options)
            parameters = self.choose_parameters(options, num,
                                                allow_extra_parameters)
        elif num is not None and len(parameters) != num:
            print 'Wrong number of parameters.'
            parameters = self.choose_parameters(options, num,
                                                allow_extra_parameters)
        else:
            extra_parameters = list(np.array(parameters)[ \
                    np.where([p not in options for p in parameters])[0]])
            if not allow_extra_parameters and len(extra_parameters) > 0:
                print '\nThe constraint ' + self.name + \
                    ' does not have the following parameters:'
                print ', '.join(extra_parameters)
                parameters = self.choose_parameters(options, num)
        return parameters

    def get_chain_parameter(self, parameter):
        if parameter not in self.settings['parameters']:
            print '\nThe constraint ' + self.name + \
                ' does not have the parameter ' + str(parameter) + '.'
            print 'Choose a parameter to use instead.'
            parameter = self.choose_parameters(num=1)[0]
        # find the index for the parameter
        index = np.where(np.array(self.chain.parameters) == parameter)[0][0]
        # create a ChainParameter object for each parameter
        return ChainParameter(self.chain, index)

    def add_derived_parameter(self, new_name, f_str, par_names, 
                              par_indices=None, update_order=True):

        input_f_str = f_str

        if par_indices is None:
            par_indices = []
            for name in par_names:
                index = np.where(np.array(self.chain.parameters) == name)[0]
                if len(index) == 0:
                    print 'Parameter ' + str(name) + ' not found in chain.'
                    print 'Use the name of an existing parameter or ' + \
                        'supply a list of chain indices.'
                else:
                    par_indices.append(index[0])

        # check whether any cosmology functions are required
        cosm_fns_required = []
        for cf in cosmology.functions.keys():
            if cf in f_str:
                cosm_fns_required.append(cf)

        self.set_cosmology_model()
            
        if len(cosm_fns_required) > 0:
            have_cp_samples = bool(len(self.cosmology_parameter_samples))
            model_parameters = ['h', 'omegam', 'omegabhh', 'omegagamma', 
                                'mnu', 'neff', 'omegak', 'sigma8', 'ns']
            if self.model == 'wCDM':
                model_parameters.append('w')
            for p in model_parameters:
                # check whether the chain_to_cosmology mapping is defined
                if p in self.settings['chain_to_cosmology']:
                    p_map = self.settings['chain_to_cosmology'][p]
                    pf_str = p_map['function']
                    if pf_str != 'default':
                        if p not in cosmology.parameters:
                            cosmology.parameters.append(p)
                        chain_params_used = p_map['chain_parameters']
                        chain_indices = p_map['chain_indices']
                else:
                    self.settings['chain_to_cosmology'][p] = {}
                    pf_str = ''
                    while pf_str == '':
                        pf_str = raw_input('\nWhat is ' + p + ' in terms of ' \
                                               + 'the chain parameters?\n' + \
                                               '(Press Enter to see the ' + \
                                               'list of chain parameters,\n' + \
                                               'or enter "d" to use a fixed' + \
                                               ' default value.)\n> ')
                        if pf_str == '':
                            self.display_parameters()
                    if pf_str.strip() == 'd':
                        self.settings['chain_to_cosmology'][p] \
                            ['chain_parameters'] = []
                        self.settings['chain_to_cosmology'][p] \
                            ['chain_indices'] = []
                        self.settings['chain_to_cosmology'][p] \
                            ['function'] = 'default'
                    else:
                        if p not in cosmology.parameters:
                            cosmology.parameters.append(p)
                        chain_indices = []
                        chain_params_used = []
                        for i, cp in enumerate(self.chain.parameters):
                            if cp in pf_str:
                                chain_indices.append(i)
                                chain_params_used.append(cp)
                        self.settings['chain_to_cosmology'][p] \
                            ['chain_parameters'] = chain_params_used
                        self.settings['chain_to_cosmology'][p] \
                            ['chain_indices'] = chain_indices
                        self.settings['chain_to_cosmology'][p] \
                            ['function'] = pf_str

                if p in cosmology.parameters and not have_cp_samples:
                    pf = lambdify(symbols(chain_params_used), 
                                  sympify(pf_str), 'numpy')
                    # handle sympy errors

                    self.cosmology_parameter_samples.append( \
                        pf(*[self.chain.samples[:, \
                                    i+self.chain.first_par_column] \
                                 for i in chain_indices]))

            for cf in cosm_fns_required:
                # insert cosmology parameters as extra arguments
                substrings = [s.strip() for s in f_str.split(')')]
                new_substrings = []
                for sub in substrings[:-1]:
                    if cf in sub:
                        if sub[-1] != '(':
                            sub += ', '
                        sub += ','.join(cosmology.parameters)
                    new_substrings.append(sub)
                f_str = ')'.join(new_substrings + [substrings[-1]])

        # need to fix this for cases where both cosmology functions
        # and regular parameters are required

        
        f = lambdify(symbols([str(x) for x in \
                                  par_names + cosmology.parameters]), 
                     sympify(f_str), [cosmology.functions, 'numpy'])
        # handle errors from the sympy operations


        if len(par_indices) == len(par_names):
            self.chain.parameters.append(new_name)
            self.chain.column_names.append(new_name)
            if len(cosm_fns_required) > 0:
                new_column = f(\
                    *[self.chain.samples[:,i+self.chain.first_par_column] \
                      for i in par_indices] + self.cosmology_parameter_samples)
            else:
                new_column = f(\
                    *[self.chain.samples[:,i+self.chain.first_par_column] \
                      for i in par_indices])
            # need to test this more: crashes in some cases when
            # adding a chain to existing likelihoods that depend
            # on cosmology functions
            self.chain.samples = np.hstack((self.chain.samples, 
                                      np.array([new_column]).T))
            if update_order:
                self.settings['parameters'][new_name] = []
                self.settings['order'].append(('derived_parameter', 
                                               new_name))
                self.settings['derived_parameters'][new_name] = {
                    'function': input_f_str,
                    'parameters': par_names,
                    'indices': par_indices}

    def set_cosmology_model(self):
        cosmology.parameters = []
        if self.model == 'LCDM':
            cosmology.model_class = cosmology.LCDM
        elif self.model == 'wCDM':
            cosmology.model_class = cosmology.wCDM            
        else:
            print 'Model class ' + self.model + ' is not implemented.'

    def compute_1d_stats(self, parameters, stats='mean and standard deviation'):
        print
        for p in parameters:
            cp = self.get_chain_parameter(p)
            if stats == 'mean and standard deviation':
                fmt_str = '{0:s} = {1:.3g} +/- {2:.3g}'
                print fmt_str.format(p, cp.mean(), cp.standard_deviation())
            elif stats == 'equal-tail limits':
                med = cp.median()
                limits_68 = cp.equal_tail_limits(68.27)
                limits_95 = cp.equal_tail_limits(95.45)
                fmt_str = '{0:s} = {1:.3g} -{2:.3g} +{3:.3g} (68.27%), ' + \
                    '-{4:.3g} +{5:.3g} (95.45%)'
                print fmt_str.format(p, med, 
                                     med-limits_68[0], limits_68[1]-med,
                                     med-limits_95[0], limits_95[1]-med)
            elif stats == 'upper limit':
                limit_68 = cp.upper_limit(68.27)
                limit_95 = cp.upper_limit(95.45)
                fmt_str = '{0:s} < {1:.3g} (68.27%), {2:.3g} (95.45%)'
                print fmt_str.format(p, limit_68, limit_95)
            elif stats == 'lower limit':
                limit_68 = cp.lower_limit(68.27)
                limit_95 = cp.lower_limit(95.45)
                fmt_str = '{0:s} > {1:.3g} (68.27%), {2:.3g} (95.45%)'
                print fmt_str.format(p, limit_68, limit_95)
            else:
                print 'The statistic "' + stats + '" is not implemented.'

    def save_contour_data(self, parameters, n_samples, grid_size, smoothing, 
                          contour_pct, contour_levels, X, Y, Z):
        filename = os.path.join('Data', time.strftime('%Y-%m-%d'), 
                                time.strftime('%Z.%H.%M.%S') + '_contour.txt')
        writer = utils.open_with_path(filename, 'w')
        header = ' '.join(parameters) + ' # parameters\n' + \
            str(n_samples) + ' # n_samples\n' + \
            str(grid_size[0]) + ' ' + str(grid_size[1]) + ' # grid_size\n' + \
            str(smoothing) + ' # smoothing\n' + \
            ' '.join([str(cp) for cp in contour_pct]) + ' # contour_pct\n' + \
            ' '.join([str(cl) for cl in contour_levels]) + \
                         ' # contour_levels\n'
        np.savetxt(writer, 
                   np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T, 
                   fmt='%.3e', header=header, comments='')
        writer.close()
        self.settings['contour_data_files'].append(filename)

    def load_contour_data(self, parameters, n_samples, grid_size, 
                          smoothing, contour_pct):
        contour_data = None
        print self.settings['contour_data_files']
        for f in self.settings['contour_data_files']:
            reader = utils.open_if_exists(f, 'r')
            test_parameters = reader.readline().split('#')[0].split()
            test_n_samples = int(float(reader.readline().split('#')[0]))
            test_grid_size = [int(float(x)) for x in \
                                  reader.readline().split('#')[0].split()]
            test_smoothing = float(reader.readline().split('#')[0])
            test_contour_pct = [float(x) for x in \
                                  reader.readline().split('#')[0].split()]

            match = (test_parameters == parameters) and \
                (test_n_samples == n_samples) and \
                np.all([x == y for x, y in \
                            zip(test_grid_size, grid_size)]) and \
                test_smoothing == smoothing and \
                np.all([x == y for x, y in \
                            zip(test_contour_pct, contour_pct)])

            if match:
                print 'Plotting contours from ' + f
                contour_levels = [float(x) for x in \
                                      reader.readline().split('#')[0].split()]
                X, Y, Z = np.loadtxt(reader, skiprows=1, unpack=True)
                contour_data = (contour_levels, X.reshape(grid_size),
                                Y.reshape(grid_size), Z.reshape(grid_size))
            reader.close()
        return contour_data

    def rename(self, new_name):
        # check that name is unique

        self.name = new_name

    def set_color(self, color=None):
        if color is None:
            color = utils.get_input_float('\nRGB values for main color?\n' + \
                                              '(3 numbers between 0 and 1 ' + \
                                              'separated by spaces)\n> ', 
                                          num=3)
        if not np.all([0.<=x<=1. for x in list(color)]):
            print 'Values must be between 0 and 1.'
            self.set_color()
        self.settings['color'] = tuple(color)
