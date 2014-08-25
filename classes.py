import sys
import os.path
import glob
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import default_plot_settings
import utils

class Session(object):

    def __init__(self, name, path=None, save_log=True):
        self.settings = Settings()
        self.history_file = '.session_history'
        self.name = name
        self.path = path
        self.save_log = save_log
        self.plot = None
        self.pdfs = []
        # if path is given, search it for the named log (error if not found)

        # if no path given, search multiple paths for named log file

        # if existing log found, set up environment using old settings

        # if not found, notify that new log file is assumed

        if self.save_log:
            print 'Session name:', self.name

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

    def set_up_pdf(self):
        name = raw_input('\nLabel for constraint?\n> ')
        model = raw_input('Model?\n> ')
        self.pdfs += [PostPDF(name, model)]

    def choose_pdf(self, require_data=False, require_no_chain=False):
        pdfs = list(self.pdfs)
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

    def rename_pdf(self):
        pdf = self.choose_pdf()
        if pdf is not None:
            new_name = raw_input('\nNew name?\n> ')
            pdf.rename(new_name)

    def add_chain(self):
        pdf = self.choose_pdf(require_no_chain=True)
        # select from list of currently defined chains (from all pdfs),
        # previously used chains (read from file),
        # or get files and name for new chain (if new, save to file)
        if pdf is not None:
            if len(self.history['chains']) > 0:
                options = [ch[0] for ch in self.history['chains']]
                details = ['Chains:\n' + '\n'.join(
                        [textwrap.fill(s, initial_indent='    ',
                                       subsequent_indent='        ') \
                             for s in sorted(ch[1])]) \
                               for ch in self.history['chains']]
                m = Menu(options=options, more=details,
                         exit_str='New chain',
                         header=['Choose a chain: ' \
                                     '(add ? to the number to get ' \
                                     'more info on a chain)'][0])
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
        if pdf is not None:
            parameters = self.choose_parameters(pdf)
            pdf.compute_1d_stats(parameters)

    def set_up_plot(self):
        n_rows = self.get_input_integer('\nNumber of subplot rows?\n> ',
            error_text='Number of rows must be an integer > 0.')
        n_cols = self.get_input_integer('Number of subplot columns?\n> ',
            error_text='Number of columns must be an integer > 0.')
        if n_rows < 1 or n_cols < 1:
            print 'Must have > 0 rows and columns.'
            self.set_up_plot()
        self.plot = Plot()
        self.plot.set_up_plot_grid(n_rows, n_cols)
        plt.show(block=False)
        print '(If you cannot see the plot, try changing the '
        print 'matplotlib backend. Current backend is ' + \
            plt.get_backend() + '.)'

    def plot_constraint(self):
        pdf = self.choose_pdf(require_data=True)
        if pdf is not None:
            # get row and column of subplot
            n_rows = self.plot.n_rows
            n_cols = self.plot.n_cols
            if n_rows > 1:
                row = self.get_input_integer( \
                    '\nSubplot row (0-' + str(self.plot.n_rows - 1) + ')?\n> ',
                    error_text='Must choose an integer.')
            else:
                row = 0
            if n_cols > 1:
                col = self.get_input_integer( \
                    '\nSubplot column (0-' + str(str(self.plot.n_cols - 1)) + \
                        ')?\n> ',
                    error_text='Must choose an integer.')
            else:
                col = 0
            if row < 0 or row > self.plot.n_rows - 1 or \
                    col < 0 or col > self.plot.n_cols - 1:
                print 'Row or column number is out of required range.'
                self.plot_constraint()
            ax = self.plot.axes[row][col]
            if len(ax.pdfs) == 0:
                self.set_up_subplot(row, col, pdf)
            n_dim = len(ax.parameters)
            if n_dim == 1:
                self.plot.plot_1d_pdf(ax, pdf)
            elif n_dim == 2:
                self.plot.plot_2d_pdf(ax, pdf)
            plt.draw()

    def set_up_subplot(self, row, col, pdf):
        ax = self.plot.axes[row][col]
        ax.parameters = self.choose_parameters(pdf)
        if len(ax.parameters) > 2:
            print 'Number of parameters must be 1 or 2.'
            self.set_up_subplot(row, col)

    def pdf_exists(self):
        if len(self.pdfs) > 0:
            return True
        else:
            return False

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

    def plot_and_pdf_with_data_exist(self):
        if self.plot and self.pdf_with_data_exists():
            return True
        else:
            return False

    def get_input_integer(self, prompt, 
                          error_text='Input must be an integer.',
                          error_action='retry'):
        value = 0
        try:
            value = int(raw_input(prompt))
        except ValueError as e:
            if error_action == 'retry':
                print error_text
                value = self.get_input_integer(prompt, error_text=error_text,
                                               error_action=error_action)
            else:
               sys.exit(error_text)
        return value

class Settings(object):

    def __init__(self):
        pass


class Menu(object):

    def __init__(self, options=None, more=None, exit_str='Exit', header=None):
        self.choice = None
        self.i_choice=None
        self.exit = exit_str
        self.header = header
        self.prompt = '----\n> '
        if options:
            self.update_options(options)
        else:
            self.options = [self.exit]
        self.more = more

    def update_options(self, options):
        self.options = list(options) + [self.exit]

    def get_choice(self, options=None):
        more_info = False
        if options:
            self.update_options(options)
        print
        if self.header:
            print self.header
        for i, opt in enumerate(self.options):
            print textwrap.fill(str(i) + ': ' + str(opt), 
                                initial_indent='',
                                subsequent_indent='    ')
        response = raw_input(self.prompt).strip()
        # check whether more info is requested
        if self.more and response[-1] == '?':
            more_info = True
            response = response[:-1]
        # get an integer
        try:
            i_choice = int(response)
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
        # provide more info on a particular choice
        if more_info:
            print str(self.more[self.i_choice])
            self.get_choice()

    def add_option(self, position=None):
        pass


class Plot(object):

    def __init__(self):
        self.n_rows = 1
        self.n_cols = 1

    def set_up_plot_grid(self, n_rows, n_cols):
        # assume all subplots occupy a single row and column for now
        # (also possible to use gridspec for plots that span multiple
        #  rows/columns - see http://matplotlib.org/users/gridspec.html)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.plot_grid = matplotlib.gridspec.GridSpec(n_rows, n_cols)
        self.axes = []
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                row.append(plt.subplot(self.plot_grid[i, j]))
            self.axes.append(row)
        for ax_row in self.axes:
            for ax in ax_row:
                ax.pdfs = []

    def plot_1d_pdf(self, ax, pdf, bins_per_sigma=5, p_min_frac=0.01):
        parameter = pdf.get_chain_parameter(ax.parameters[0])
        bin_width = parameter.standard_deviation() / float(bins_per_sigma)
        number_of_bins = (parameter.values.max()-parameter.values.min())/ \
                             bin_width
        pdf_1d, bin_edges = np.histogram(parameter.values, bins=number_of_bins,
                                         weights=parameter.chain.multiplicity, 
                                         density=True)
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
        # trim bins at either end with prob./max(pdf_1d) < p_min_frac
        while pdf_1d[0] < p_min_frac*pdf_1d.max():
            pdf_1d = np.delete(pdf_1d, 0)
            bin_centers = np.delete(bin_centers, 0)
        while pdf_1d[-1] < p_min_frac*pdf_1d.max():
            pdf_1d = np.delete(pdf_1d, -1)
            bin_centers = np.delete(bin_centers, -1)
        ax.plot(bin_centers, pdf_1d)

    def plot_2d_pdf(self, ax, pdf):
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

    def get_chain_parameter(self, parameter):
        if parameter not in self.parameters:
            sys.exit('The PDF ' + self.name + \
                         ' does not have the parameter ' + str(parameter))
        # find the index for the parameter
        index = np.where(np.array(self.chain.parameters) == parameter)[0][0]
        # create a ChainParameter object for each parameter
        return ChainParameter(self.chain, index)

    def compute_1d_stats(self, parameters):
        # how to do this if there is no chain, only likelihoods?

        for p in parameters:
            cp = self.get_chain_parameter(p)
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
            reader = utils.open_if_exists(chain_file, 'r')
            new_samples = np.loadtxt(reader)
            reader.close()
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
        paramname_reader = utils.open_if_exists(paramname_file, 'r')
        lines = paramname_reader.readlines()
        paramname_reader.close()
        parameters = []
        for line in lines:
            # remove trailing * used to denote derived parameters
            parameters.append( line.strip().split()[0].split('*')[0] )
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

    """
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
    """

class DerivedParameter(ChainParameter):
# use supplied function to combine columns from the chain
    def __init__(self):
        pass

