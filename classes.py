import sys
import os.path
import time
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
        self.settings = {}
        self.history_file = '.session_history'
        self.name = name
        self.save_log = save_log
        if save_log:
            self.log_file = os.path.join('Logs', time.strftime('%Y-%m-%d'), 
                                         name)
        self.plot = None
        self.pdfs = []

        for attr in ['name', 'log_file']:
            self.settings[attr] = getattr(self, attr)
        self.settings['pdfs'] = {}

        log_reader = None
        # if path is given, search it for the named log (error if not found)
        if path:
            log_reader = utils.open_if_exists(os.path.join(path, name), 'rb')
        # if no path given, search multiple paths for named log file
        else:
            log_paths = os.listdir('Logs')
            for x in log_paths:
                p = os.path.join('Logs', x)
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, name)):
                    log_reader = open(os.path.join(p, name), 'rb')
        # if existing log found, set up environment using old settings
        if log_reader:
            self.load_log(log_reader)
        # if not found, notify that new log file is assumed
        elif self.save_log:
            print 'No log file found. Creating new file:\n    ' + \
                self.log_file

        # check for file with inputs from all previous sessions
        # (e.g. chains, likelihoods, joint pdfs) and load it
        self.load_history()

    def load_log(self, reader):
        log_settings = json.loads(reader.read())
        reader.close()
        for pdf in log_settings['pdfs']:
            d = log_settings['pdfs'][pdf]
            self.set_up_pdf(name=d['name'], model=d['model'])
            if d['chain_name']:
                self.choose_pdf(name=d['name']).add_chain(d['chain_name'],
                                                          d['chain_files'])
        # add likelihoods

        if log_settings['plot']:
            d = log_settings['plot']
            self.set_up_plot((d['n_rows'], d['n_cols']))
            for row in range(d['n_rows']):
                for col in range(d['n_cols']):
                    ax_settings = d['{0:d}.{1:d}'.format(row, col)]
                    if ax_settings['pdfs']:
                        for pdf in ax_settings['pdfs']:
                            self.plot_constraint(row=row, col=col,
                                                 pdf=self.choose_pdf(pdf),
                                                 parameters=ax_settings \
                                                     ['parameters'])

    def load_history(self):
        if os.path.isfile(self.history_file):
            reader = open(self.history_file, 'r')
            self.history = json.loads(reader.read())
        else:
            self.history = {'chains': [], 'likelihoods': []}
        
    def save_and_exit(self):
        # save settings
        if self.save_log:
            writer = utils.open_with_path(self.log_file, 'wb')
            writer.write(json.dumps(self.settings, sort_keys=True, indent=4))
            writer.close()
            print '\nLog file saved as ' + self.log_file

        # save history
        history_writer = utils.open_with_path(self.history_file, 'w')
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

    def set_up_pdf(self, name=None, model=None):
        if name:
            print '\nConstraint name: ' + str(name)
        else:
            name = raw_input('\nLabel for constraint?\n> ')
        if model:
            print 'Model name: ' + str(model)
        else:
            model = raw_input('Model?\n> ')
        new_pdf = PostPDF(name, model)
        self.pdfs += [new_pdf]
        self.settings['pdfs'][name] = new_pdf.settings

    def choose_pdf(self, name=None, 
                   require_data=False, require_no_chain=False):
        chosen_pdf = None
        pdfs = list(self.pdfs)
        for pdf in pdfs:
            if name and pdf.name == name:
                chosen_pdf = pdf
            if (require_no_chain and pdf.chain) or \
                    (require_data and (pdf.chain is None) and \
                         len(pdf.likelihoods) == 0):
                pdfs.remove(pdf)
        if not chosen_pdf:
            m = Menu(options=[pdf.name for pdf in pdfs], exit_str='Cancel',
                     header='Choose a constraint:')
            m.get_choice()
            if m.choice != m.exit:
                chosen_pdf = pdfs[m.i_choice]
        return chosen_pdf

    def rename_pdf(self):
        pdf = self.choose_pdf()
        if pdf is not None:
            old_name = pdf.name
            new_name = raw_input('\nNew name?\n> ')
            pdf.rename(new_name)
            set_pdfs = self.settings['pdfs']
            if old_name in set_pdfs:
                set_pdfs[new_name] = set_pdfs[old_name]
                del set_pdfs[old_name]
            if self.plot:
                for ax_row in self.plot.axes:
                    for ax in ax_row:
                        subplot_pdfs = self.plot.settings \
                            ['{0:d}.{1:d}'.format(ax.row, ax.col)] \
                            ['pdfs']
                        if old_name in subplot_pdfs:
                            subplot_pdfs[new_name] = subplot_pdfs[old_name]
                            del subplot_pdfs[old_name]

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

    def set_up_plot(self, size=None):
        if size:
            n_rows = size[0]
            n_cols = size[1]
            print '\nSetting up {0:d}x{1:d} plot.'.format(n_rows, n_cols)
        else:
            e_str = 'Number of {0:s} must be an integer > 0.'
            n_rows = self.get_input_integer('\nNumber of subplot rows?\n> ',
                                            error_text=e_str.format('rows'))
            n_cols = self.get_input_integer('Number of subplot columns?\n> ',
                                            error_text=e_str.format('columns'))
        if n_rows < 1 or n_cols < 1:
            print 'Must have > 0 rows and columns.'
            self.set_up_plot()
        self.plot = Plot()
        self.settings['plot'] = self.plot.settings
        self.plot.set_up_plot_grid(n_rows, n_cols)
        plt.show(block=False)
        print '(If you cannot see the plot, try changing the '
        print 'matplotlib backend. Current backend is ' + \
            plt.get_backend() + '.)'

    # merge with get_col function?
    def get_row(self, default=None):
        n_rows = self.plot.settings['n_rows']
        if default is None:
            if n_rows > 1:
                row = self.get_input_integer( \
                    '\nSubplot row (0-' + str(n_rows - 1) + ')?\n> ',
                    error_text='Must choose an integer.')
            else:
                row = 0
        else:
            row = default
        if row < 0 or row > n_rows - 1:
            print 'Row number is out of required range.'
            row = self.get_row()
        return row

    def get_col(self, default=None):
        n_cols = self.plot.settings['n_cols']
        if default is None:
            if n_cols > 1:
                col = self.get_input_integer( \
                    '\nSubplot column (0-' + str(n_cols - 1) + ')?\n> ',
                    error_text='Must choose an integer.')
            else:
                col = 0
        else:
            col = default
        if col < 0 or col > n_cols - 1:
            print 'Column number is out of required range.'
            col = self.get_col()
        return col

    def change_plot(self):
        # change to menu of options
        row = self.get_row()
        col = self.get_col()
        ax = self.plot.axes[row][col]
        # write Plot method to do this and also save new labels to settings?
        ax.set_xlabel(raw_input('New x-axis label?\n> '))
        ax.set_ylabel(raw_input('New y-axis label?\n> '))
        plt.draw()

    def plot_constraint(self, row=None, col=None, pdf=None, parameters=None):
        if pdf is None:
            pdf = self.choose_pdf(require_data=True)
        if pdf is not None:
            row = self.get_row(default=row)
            col = self.get_col(default=col)
            ax = self.plot.axes[row][col]
            if len(ax.pdfs) == 0:
                self.set_up_subplot(row, col, pdf, parameters)
            n_dim = len(ax.parameters)
            if n_dim == 1:
                self.plot.plot_1d_pdf(ax, pdf)
            elif n_dim == 2:
                self.plot.plot_2d_pdf(ax, pdf)
            plt.draw()

    def set_up_subplot(self, row, col, pdf, parameters):
        ax = self.plot.axes[row][col]
        if parameters is None:
            ax.parameters = self.choose_parameters(pdf)
        else:
            ax.parameters = parameters
        if len(ax.parameters) > 2:
            print 'Number of parameters must be 1 or 2.'
            self.set_up_subplot(row, col, pdf, parameters)
        self.plot.settings['{0:d}.{1:d}'.format(row, col)]['parameters'] = \
            ax.parameters

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
        self.settings = {'n_rows': 1, 'n_cols': 1}

    def set_up_plot_grid(self, n_rows, n_cols):
        # assume all subplots occupy a single row and column for now
        # (also possible to use gridspec for plots that span multiple
        #  rows/columns - see http://matplotlib.org/users/gridspec.html)
        self.settings['n_rows'] = n_rows
        self.settings['n_cols'] = n_cols
        self.plot_grid = matplotlib.gridspec.GridSpec(n_rows, n_cols)
        self.axes = []
        for i in range(n_rows):
            row = []
            for j in range(n_cols):
                row.append(plt.subplot(self.plot_grid[i, j]))
            self.axes.append(row)
        for i, ax_row in enumerate(self.axes):
            for j, ax in enumerate(ax_row):
                ax.row = i
                ax.col = j
                ax.pdfs = []
                self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)] = \
                    {'pdfs': {}}

    def plot_1d_pdf(self, ax, pdf, bins_per_sigma=5, p_min_frac=0.01):
        ax.pdfs += [pdf]
        self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)] \
            ['pdfs'][pdf.name] = {}
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
        if not ax.get_xlabel():
            ax.set_xlabel(ax.parameters[0])
            ax.set_ylabel('P(' + ax.parameters[0] + ')')

    def plot_2d_pdf(self, ax, pdf):
        ax.pdfs += [pdf]
        self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)] \
            ['pdfs'][pdf.name] = {}
        pass


class PostPDF(object):

    def __init__(self, name, model):
        self.settings = {}
        self.rename(name)
        self.model = model
        self.parameters = []
        self.chain = None
        self.chain_files = []
        self.likelihoods = []
        for attr in ['name', 'model', 'parameters', 
                     'chain_files', 'likelihoods']:
            self.settings[attr] = getattr(self, attr)

    def add_chain(self, name, files):
        # check if already have chain, if files exist, if name is unique

        # check if model is consistent?

        self.chain_files.extend(files)
        self.settings['chain_name'] = name
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

