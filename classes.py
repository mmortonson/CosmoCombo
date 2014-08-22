import sys
import os.path
import glob
import json
import textwrap
import numpy as np

class Session(object):

    def __init__(self, name, path=None):
        self.settings = Settings()
        self.history_file = '.session_history'
        self.name = name
        self.plot = None
        self.pdfs = []
        # if path is given, search it for the named script (error if not found)

        # if no path given, search multiple paths for named script

        # if existing script found, set up environment using old settings

        # if script not found, notify that new script is assumed

        # check for file with inputs from all previous sessions
        # (e.g. chains, likelihoods, joint pdfs) and load it
        self.load_history()

    def load_settings(self):
        pass

    def load_history(self):
        if os.path.isfile(self.history_file):
            reader = open(self.history_file, 'r')
            self.history = json.loads(reader.read())
        else:
            self.history = {'chains': [], 'likelihoods': []}
        
    def save_and_exit(self):
        # save settings

        # save history
        history_writer = open(self.history_file, 'w')
        history_writer.write(json.dumps(self.history))
        history_writer.close()

    def start_interactive(self, options=None):
        print 'Session name:', self.name
        menu = Menu()
        menu.get_choice(options=self.active_options(options))
        while menu.choice != menu.exit:
            if options[menu.choice]['condition']():
                options[menu.choice]['action']()
            menu.get_choice(options=self.active_options(options))
        self.save_and_exit()

    def active_options(self, all_options):
        if all_options is None:
            options = None
        else:
            options = []
            for key in all_options:
                if all_options[key]['condition']():
                    options.append(key)
        return options

    def get_old_pdf(self):
        pass

    def set_up_pdf(self):
        name = raw_input('\nLabel for constraint?\n> ')
        model = raw_input('Model?\n> ')
        self.pdfs += [PostPDF(name, model)]

    def choose_pdf(self, require_data=False, require_no_chain=False):
        pdfs = self.pdfs
        for pdf in pdfs:
            if (require_no_chain and pdf.chain) or \
                    (require_data and (pdf.chain is None) and \
                         len(pdf.likelihoods) == 0):
                pdfs.remove(pdf)
        m = Menu(options=[pdf.name for pdf in pdfs], exit_str='Cancel',
                 header='Choose a constraint:')
        m.get_choice()
        if m.choice == m.exit:
            return None
        else:
            return pdfs[m.i_choice]

    def add_chain(self):
        pdf = self.choose_pdf(require_no_chain=True)
        # select from list of currently defined chains (from all pdfs),
        # previously used chains (read from file),
        # or get files and name for new chain (if new, save to file)
        if pdf is not None:
            if len(self.history['chains']) > 0:
                options = [ch[0] + ' (' + ', '.join(ch[1]) + ')' for \
                               ch in self.history['chains']]
                m = Menu(options=options, exit_str='New chain',
                         header='Choose a chain:')
                m.get_choice()
                if m.choice == m.exit:
                    pdf.add_chain(*self.define_new_chain())
                else:
                    pdf.add_chain(*self.history['chains'][m.i_choice])
            else:
                pdf.add_chain(*self.define_new_chain())

    def define_new_chain(self):
        files = []
        for f in raw_input('\nChain file names?\n> ').split():
            files += glob.glob(f)
        name = raw_input('Label for chain?\n> ')
        # check if name is already in history; if so, replace with new
        for chain in self.history['chains']:
            if chain[0] == name:
                self.history['chains'].remove(chain)
        self.history['chains'].append((name, files))
        return (name, files)

    def choose_parameters(self, pdf):
        # choose one or more parameters 
        # (enter names or pick from chain column list)
        print '\nEnter one or more parameter names (on one line),'
        print 'or press Enter to see a list of available parameters:'
        parameters = raw_input('> ').split()
        if len(parameters) == 0:
            pdf.display_parameters()
            parameters = self.choose_parameters(pdf)
        else:
            extra_parameters = list(np.array(parameters)[np.where([
                    p not in pdf.parameters for p in parameters])[0]])
            if len(extra_parameters) > 0:
                print 'The PDF ' + pdf.name + \
                    ' does not have the following parameters:'
                print ', '.join(extra_parameters)
                parameters = self.choose_parameters(pdf)
        return parameters

    def compute_1d_stats(self):
        pdf = self.choose_pdf(require_data=True)
        parameters = self.choose_parameters(pdf)
        pdf.compute_1d_stats(parameters)

    def set_up_plot(self):
        print 'Setting up plot'
        self.plot = Plot()

    def plot_constraint(self):
        print 'Plotting constraint'

    def pdf_without_chain_exists(self):
        answer = False
        for pdf in self.pdfs:
            if pdf.chain is None:
                answer = True
        return answer

    def pdf_with_data_exists(self):
        answer = False
        for pdf in self.pdfs:
            if pdf.chain or len(pdf.likelihoods) > 0:
                answer = True
        return answer

    def plot_exists(self):
        if self.plot:
            return True
        else:
            return False


class Settings(object):

    def __init__(self):
        pass


class Menu(object):

    def __init__(self, options=None, exit_str='Exit', header=None):
        self.choice = None
        self.i_choice=None
        self.exit = exit_str
        self.header = header
        self.prompt = '----\n> '
        if options:
            self.update_options(options)
        else:
            self.options = [self.exit]

    def update_options(self, options):
        self.options = list(options) + [self.exit]

    def get_choice(self, options=None):
        if options:
            self.update_options(options)
        print
        if self.header:
            print self.header
        for i, opt in enumerate(self.options):
            print textwrap.fill(str(i) + ': ' + str(opt), 
                                initial_indent='',
                                subsequent_indent='    ')
        # get an integer
        try:
            i_choice = int(raw_input(self.prompt))
        except ValueError as e:
            i_choice = -1
        # check that the integer corresponds to a valid option
        if i_choice >=0 and i_choice < len(self.options):
            self.i_choice = i_choice
            self.choice = self.options[i_choice]
        else:
            print 'Not a valid option. Enter a number between 0 and ' + \
                str(len(self.options)-1) + ':'
            self.get_choice()

    def add_option(self, position=None):
        pass


class Plot(object):

    def __init__(self):
        pass


class PostPDF(object):

    def __init__(self, name, model):
        self.rename(name)
        self.model = model
        self.parameters = []
        self.chain = None
        self.likelihoods = []

    def add_chain(self, name, files):
        # check if already have chain, if files exist, if name is unique

        # check if model is consistent?

        self.chain = MCMCChain(name, files)
        self.add_parameters(self.chain.parameters)

    def add_parameters(self, new_parameters):
        for p in new_parameters:
            if p not in self.parameters:
                self.parameters.append(p)

    def display_parameters(self):
        print textwrap.fill(', '.join(self.parameters))

    def compute_1d_stats(self, parameters):
        # how to do this if there is no chain, only likelihoods?

        for p in parameters:
            if p not in self.parameters:
                sys.exit('The PDF ' + self.name + \
                             ' does not have the parameter ' + str(p))
            # find the index for the parameter
            index = np.where(np.array(self.chain.parameters) == p)[0][0]
            # create a ChainParameter object for each parameter
            cp = ChainParameter(self.chain, index)
            # print stats
            fmt_str = '{0:s} = {1:.3g} +/- {2:.3g}'
            print fmt_str.format(p, cp.mean(), cp.standard_deviation())

    def rename(self, new_name):
        # check that name is unique

        self.name = new_name


class MCMCChain(object):

    def __init__(self, name, chain_files, 
                 mult_column=0, lnlike_column=1, first_par_column=2,
                 paramname_file=None):
        self.rename(name)
        first_file = True
        for chain_file in chain_files:
            if os.path.isfile(chain_file):
                reader = open(chain_file, 'r')
                new_samples = np.loadtxt(reader)
                reader.close()
            else:
                sys.exit('File ' + str(chain_file) + ' not found.')
            if first_file:
                self.samples = np.copy(new_samples)
                first_file = False
            else:
                # check that number of columns are the same

                self.samples = np.vstack((self.samples, new_samples))
        self.mult_column = mult_column
        self.multiplicity = self.samples[:,mult_column]
        self.lnlike_column = lnlike_column
        self.first_par_column = first_par_column
        if paramname_file is None:
            paramname_file = '_'.join(chain_files[0].split('_')[:-1]) + \
                '.paramnames'
        self.parameters = self.get_parameter_names(paramname_file)
        self.column_names = list(self.parameters)
        self.column_names.insert(mult_column, 'mult')
        self.column_names.insert(lnlike_column, '-ln(L)')

    def get_parameter_names(self, paramname_file):
        if os.path.isfile(paramname_file):
            paramname_reader = open(paramname_file, 'r')
            lines = paramname_reader.readlines()
            paramname_reader.close()
            parameters = []
            for line in lines:
                # remove trailing * used to denote derived parameters
                parameters.append( line.strip().split()[0].split('*')[0] )
        else:
            sys.exit('File ' + str(paramname_file) + ' not found.')
        return parameters

    """
    def thin(self, thinning_factor):
        # would be more accurate to account for varying multiplicities
        self.samples = self.samples[::thinning_factor,:]
        self.multiplicity = self.multiplicity[::thinning_factor]

    def importance_sample(self, likelihood, parameter_functions, 
                          invert=False, print_status=False):
        # *********************************************************
        # need to update anything else that depends on multiplicity
        # after running this
        # *********************************************************
        parameters = {}
        n_samples = len(self.multiplicity)
        for i in range(n_samples):
            for par in parameter_functions:
                parameters[par] = parameter_functions[par](\
                    self.samples[i,:], self.column_names)
            chisq = likelihood.chi_squared(**parameters)
            if invert:
                self.multiplicity[i] *= np.exp(0.5*chisq)
            else:
                self.multiplicity[i] *= np.exp(-0.5*chisq)
            if print_status:
                print '    sample', i+1, 'of', n_samples, \
                    ': chi squared =', chisq,
                sys.stdout.flush()
                print '\r',
    """

    def rename(self, new_name):
        # check that name is unique

        self.name = new_name


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

    def pdf_1d(self, bins_per_sigma, p_min_frac=0.01):
        bin_width = self.standard_deviation() / float(bins_per_sigma)
        number_of_bins = (self.values.max()-self.values.min())/ \
                             bin_width
        pdf, bin_edges = np.histogram(self.values, bins=number_of_bins,\
                                          weights=self.chain.multiplicity, \
                                          density=True)
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        # trim bins at either end with prob./max(pdf) < p_min_frac
        while pdf[0] < p_min_frac*pdf.max():
            pdf = np.delete(pdf, 0)
            bin_centers = np.delete(bin_centers, 0)
        while pdf[-1] < p_min_frac*pdf.max():
            pdf = np.delete(pdf, -1)
            bin_centers = np.delete(bin_centers, -1)
        return (bin_centers, pdf)


class DerivedParameter(ChainParameter):
# use supplied function to combine columns from the chain
    def __init__(self):
        pass

