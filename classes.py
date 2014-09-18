import sys
import os.path
import time
import glob
import json
from copy import deepcopy
import textwrap
from collections import OrderedDict
import itertools
import numpy as np
from scipy import stats
from sympy import symbols, sympify, lambdify
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import default_plot_settings
import utils

class Session(object):

    def __init__(self, name, path=None, save=True):
        self.settings = {}
        self.history_file = '.session_history'
        self.name = name
        self.save = save
        if save:
            self.log_file = os.path.join('Logs', time.strftime('%Y-%m-%d'), 
                                         name)
        self.plot = None
        self.pdfs = []

        for attr in ['name', 'log_file']:
            self.settings[attr] = getattr(self, attr)
        self.settings['pdfs'] = {}

        # check for file with inputs from all previous sessions
        # (e.g. chains, likelihoods, joint pdfs) and load it
        self.load_history()

        log_reader = None
        # if path is given, search it for the named log (error if not found)
        if path:
            old_log_file = os.path.join(path, name)
            log_reader = utils.open_if_exists(old_log_file, 'rb')
        # if no path given, search multiple paths for named log file
        # (sort date directories in reverse order so newest 
        #  log files are found first)
        else:
            log_paths = sorted(os.listdir('Logs'), reverse=True)
            for x in log_paths:
                p = os.path.join('Logs', x)
                old_log_file = os.path.join(p, name)
                if os.path.isdir(p) and os.path.isfile(old_log_file):
                    log_reader = open(old_log_file, 'rb')
                    break
        # if existing log found, set up environment using old settings
        if log_reader:
            print 'Using settings from ' + old_log_file
            self.load_log(log_reader)
        # if not found, notify that new log file is assumed
        elif self.save:
            print 'No log file found. Creating new file:\n    ' + \
                self.log_file

    def load_log(self, reader):
        log_settings = json.loads(reader.read())
        reader.close()
        for pdf_name in log_settings['pdfs']:
            d = log_settings['pdfs'][pdf_name]
            self.set_up_pdf(settings=d)
            pdf = self.choose_pdf(name=d['name'])
            if 'chain_name' in d:
                print 'Adding chain: ' + d['chain_name']
                pdf.add_chain(d['chain_name'], d['chain_files'])
                for chain in list(self.history['chains']):
                    if chain[0] == d['chain_name']:
                        self.history['chains'].remove(chain)
                self.history['chains'].append((d['chain_name'],
                                               d['chain_files']))
            for p in d['derived_parameters']:
                p_dict = d['derived_parameters'][p]
                pdf.add_derived_parameter(p, p_dict['function'],
                                          p_dict['parameters'],
                                          p_dict['indices'])
            if 'likelihoods' in d:
                for lk in d['likelihoods']:
                    print 'Adding likelihood: ' + lk
                    pdf.add_likelihood(lk, **d['likelihoods'][lk])
                    for hist_lk in list(self.history['likelihoods']):
                        if hist_lk[0] == lk:
                            self.history['likelihoods'].remove(hist_lk)
                    self.history['likelihoods'].append( \
                        (lk, d['likelihoods'][lk]))
            if 'contour_data_files' in d:
                pdf.settings['contour_data_files'] = d['contour_data_files']
            if ('color' in d) and (d['color'] is not None):
                pdf.settings['color'] = tuple(d['color'])

        if 'plot' in log_settings:
            d = log_settings['plot']
            self.set_up_plot(d)
            for row in range(d['n_rows']):
                for col in range(d['n_cols']):
                    ax_settings = d['{0:d}.{1:d}'.format(row, col)]
                    limits = None
                    if 'x_limits' in ax_settings:
                        limits = [ax_settings['x_limits']]
                        if 'y_limits' in ax_settings:
                            limits.append(ax_settings['y_limits'])
                    if 'pdfs' in ax_settings:
                        for pdf in ax_settings['pdfs']:
                            self.plot_constraint(row=row, col=col,
                                                 pdf=self.choose_pdf(pdf),
                                                 parameters=ax_settings \
                                                     ['parameters'],
                                                 limits=limits)
                        if 'legend' in ax_settings and ax_settings['legend']:
                            self.plot.add_legend(ax=self.plot.axes[row][col])

            self.plot.plot_grid.tight_layout(self.plot.figure)
            plt.draw()

    def load_history(self):
        if os.path.isfile(self.history_file):
            reader = open(self.history_file, 'r')
            try:
                self.history = json.loads(reader.read())
            except ValueError:
                self.history = {'chains': [], 'likelihoods': []}
            reader.close()
        else:
            self.history = {'chains': [], 'likelihoods': []}

    def save_log(self, filename=None):
        if filename is None:
            filename = self.log_file
        writer = utils.open_with_path(filename, 'wb')
        writer.write(json.dumps(self.settings, sort_keys=True, indent=4))
        writer.close()
        print '\nLog file saved as ' + filename

    def save_and_exit(self):
        # save settings
        if self.save:
            self.save_log()

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

    def set_up_pdf(self, settings=None):
        if (settings is not None) and \
                ('name' in settings) and ('model' in settings):
            name = settings['name']
            model = settings['model']
            print '\nConstraint name: ' + str(name)
            print 'Model name: ' + str(model)
        else:
            name = raw_input('\nLabel for constraint?\n> ')
            model = raw_input('Model?\n> ')
        new_pdf = PostPDF(name, model)
        if settings is not None:
            new_pdf.settings = settings
        self.pdfs += [new_pdf]
        self.settings['pdfs'][name] = new_pdf.settings

    def choose_pdf(self, name=None, 
                   require_data=False, require_no_chain=False):
        chosen_pdf = None
        pdfs = list(self.pdfs)
        for pdf in list(pdfs):
            if name and pdf.name == name:
                chosen_pdf = pdf
            if (require_no_chain and pdf.has_mcmc()) or \
                    (require_data and (pdf.chain is None)):
                pdfs.remove(pdf)
        if not chosen_pdf:
            m = Menu(options=[pdf.name for pdf in pdfs], exit_str='Cancel',
                     header='Choose a constraint:')
            m.get_choice()
            if m.choice != m.exit:
                chosen_pdf = pdfs[m.i_choice]
        return chosen_pdf

    def copy_pdf(self):
        pdf = self.choose_pdf()
        if pdf is not None:
            new_pdf = deepcopy(pdf)
            name = raw_input('\nLabel for copied constraint?\n> ')
            new_pdf.name = name
            new_pdf.settings['name'] = name
            self.pdfs += [new_pdf]
            self.settings['pdfs'][name] = new_pdf.settings

    def print_pdf_settings(self):
        pdf = self.choose_pdf()
        if pdf is not None:
            for key in sorted(pdf.settings):
                print textwrap.fill(key + ': ' + str(pdf.settings[key]),
                                    subsequent_indent='    ')

    def rename_pdf(self):
        pdf = self.choose_pdf()
        if pdf is not None:
            old_name = pdf.name
            new_name = raw_input('\nNew name?\n> ')
            pdf.rename(new_name)
            set_pdfs = self.settings['pdfs']
            if old_name in set_pdfs:
                set_pdfs[new_name] = set_pdfs[old_name]
                set_pdfs[new_name]['name'] = new_name
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
                         header='Choose a chain:\n' + \
                             '(add ? to the number to get ' + \
                             'more info on a chain)')
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
        for chain in list(self.history['chains']):
            if chain[0] == name:
                self.history['chains'].remove(chain)
        self.history['chains'].append((name, files))
        return (name, files)

    # merge with add_chain?
    def add_likelihood(self):
        pdf = self.choose_pdf()
        if pdf is not None:
            if len(self.history['likelihoods']) > 0:
                options = [lk[0] for lk in self.history['likelihoods']]
                details = ['Likelihoods:\n' + '\n'.join(
                        ['    ' + s + ': ' +  str(lk[1][s]) \
                             for s in sorted(lk[1].keys())]) \
                               for lk in self.history['likelihoods']]
                m = Menu(options=options, more=details,
                         exit_str='New likelihood',
                         header='Choose a likelihood function:\n' + \
                             '(add ? to the number to get ' + \
                             'more info on a likelihood)')
                m.get_choice()
                if m.choice == m.exit:
                    new_lk = self.define_new_likelihood(pdf)
                    pdf.add_likelihood(new_lk[0], **new_lk[1])
                else:
                    lk = self.history['likelihoods'][m.i_choice]
                    pdf.add_likelihood(lk[0], **lk[1])
            else:
                new_lk = self.define_new_likelihood(pdf)
                pdf.add_likelihood(new_lk[0], **new_lk[1])

    def define_new_likelihood(self, pdf):
        # if chain exists, choose some parameters from there
        # can also add new parameters
        name = raw_input('Label for likelihood?\n> ')
        m = Menu(options=['Flat', 'Gaussian', 'Inverse Gaussian'], 
                 exit_str=None,
                 header='Choose the form of the likelihood:\n> ')
        m.get_choice()
        form = m.choice
        parameters = self.choose_parameters(pdf, 
                                            allow_extra_parameters=True)
        pdf.add_parameters(parameters)
        priors = []
        for p in parameters:
            if len(pdf.settings['parameters'][p]) != 2:
                new_prior = utils.get_input_float( \
                    'Enter lower and upper limits of the prior on ' + \
                        p + ':\n> ', num=2)
                pdf.settings['parameters'][p] = new_prior
                # check that min value < max value

            priors.append(pdf.settings['parameters'][p])

            lk_dict = {'form': form, 'parameters': parameters, 
                       'priors': priors}

        if form in ['Gaussian', 'Inverse Gaussian']:

            means = utils.get_input_float('Enter mean values:\n> ',
                                          num=len(parameters))
            # check whether means are within priors

            variances = utils.get_input_float('Enter variances:\n> ',
                                              num=len(parameters))
            covariance = np.diag(variances)
            for i, j in itertools.combinations(range(len(parameters)), 2):
                covariance[i, j] = utils.get_input_float( \
                    'Cov(' + parameters[i] + ', ' + parameters[j] + ')?\n> ')[0]
                covariance[j, i] = covariance[i, j]
            covariance = covariance.tolist()
            lk_dict['means'] = means
            lk_dict['covariance'] = covariance

        new_lk = (name, lk_dict)

        # check if name is already in history; if so, replace with new
        for lk in list(self.history['likelihoods']):
            if lk[0] == name:
                self.history['likelihoods'].remove(lk)
        self.history['likelihoods'].append(new_lk)
        return new_lk

    def choose_parameters(self, pdf, allow_extra_parameters=False):
        # choose one or more parameters 
        # (enter names or pick from chain column list)
        print '\nEnter one or more parameter names (on one line),'
        print 'or press Enter to see a list of available parameters:'
        parameters = raw_input('> ').split()
        if len(parameters) == 0:
            pdf.display_parameters()
            parameters = self.choose_parameters(pdf, allow_extra_parameters)
        else:
            extra_parameters = list(np.array(parameters)[np.where([
                    p not in pdf.settings['parameters'].keys() \
                        for p in parameters])[0]])
            if not allow_extra_parameters and len(extra_parameters) > 0:
                print allow_extra_parameters, len(extra_parameters)>0
                print 'The PDF ' + pdf.name + \
                    ' does not have the following parameters:'
                print ', '.join(extra_parameters)
                parameters = self.choose_parameters(pdf)
        return parameters

    def add_derived_parameter(self):
        # how does this work with likelihoods without a chain?
        # add option to add parameter to all chains?
        pdf = self.choose_pdf(require_data=True)
        if pdf is not None:
            name = raw_input('\nName of the new parameter?\n> ')
            # check that this is a new name

            par_names = raw_input('\nExisting parameters required to ' + \
                                      'compute the new parameter?\n' + \
                                      '(all on one line, separated ' + \
                                      'by spaces)\n> ').split()
            f_str = raw_input('\nEnter the function to use for the new ' + \
                                  'parameter,\nusing the existing ' + \
                                  'parameter names and standard functions:\n> ')
            pdf.add_derived_parameter(name, f_str, par_names)

    def compute_1d_stats(self):
        pdf = self.choose_pdf(require_data=True)
        if pdf is not None:
            parameters = self.choose_parameters(pdf)
            pdf.compute_1d_stats(parameters)

    def set_up_plot(self, settings=None):
        self.plot = Plot()
        if settings is not None:
            for key in settings:
                self.plot.settings[key] = settings[key]
        self.settings['plot'] = self.plot.settings
        n_rows = self.plot.settings['n_rows']
        n_cols = self.plot.settings['n_cols']
        if n_rows is not None and n_cols is not None:
            print '\nSetting up {0:d}x{1:d} plot.'.format(n_rows, n_cols)
        else:
            e_str = 'Number of {0:s} must be an integer > 0.'
            n_rows = utils.get_input_integer('\nNumber of subplot rows?\n> ',
                                             error_text=e_str.format('rows'))[0]
            n_cols = utils.get_input_integer('Number of subplot columns?\n> ',
                                             error_text= \
                                                 e_str.format('columns'))[0]
        if n_rows < 1 or n_cols < 1:
            print 'Must have > 0 rows and columns.'
            self.set_up_plot()
        self.plot.set_up_plot_grid(n_rows, n_cols)
        self.plot.plot_grid.tight_layout(self.plot.figure)
        plt.show(block=False)
        print '(If you cannot see the plot, try changing the '
        print 'matplotlib backend. Current backend is ' + \
            plt.get_backend() + '.)'

    def save_plot(self):
        # add options to change file format, background color/transparency,
        # resolution, padding, etc.
        file_prefix = os.path.join('Plots', time.strftime('%Y-%m-%d'), 
                                   time.strftime('%Z.%H.%M.%S'))
        plot_file = file_prefix + '.eps'
        utils.check_path(plot_file)
        plt.savefig(plot_file, format='eps', bbox_inches='tight')
        print '\nPlot saved as ' + plot_file
        plot_log_file = file_prefix + '.log'
        self.save_log(filename=plot_log_file)

    def change_plot(self):
        # options that apply to all subplots
        options = [('Change axis limits', self.plot.change_limits),
                   ('Change axis labels', self.plot.label_axes),
                   ('Add legend', self.plot.add_legend)]
        # change appearance of a PDF
        if self.pdf_exists():
            options.extend([('Change constraint color', 
                             self.change_pdf_color)])
        # options specific to 1D or 2D plots
        subplot_1d_exists = False
        subplot_2d_exists = False
        subplot_2d_has_multiple_contours = False
        for ax_row in self.plot.axes:
            for ax in ax_row:
                if hasattr(ax, 'parameters'):
                    if len(ax.parameters) == 1:
                        subplot_1d_exists = True
                    elif len(ax.parameters) == 2:
                        subplot_2d_exists = True
                        if len(ax.pdfs) > 1:
                            subplot_2d_has_multiple_contours = True
        if subplot_2d_has_multiple_contours:
            options.extend([('Change order of contour layers', 
                             self.plot.change_layer_order)])
        options = OrderedDict(options)
        m = Menu(options=options.keys(), exit_str='Cancel')
        m.get_choice()
        if m.choice != m.exit:
            options[m.choice]()
            self.plot.plot_grid.tight_layout(self.plot.figure)
            plt.draw()

    def change_pdf_color(self):
        pdf = self.choose_pdf()
        if pdf is not None:
            pdf.set_color()
            if self.plot:
                for ax_row in self.plot.axes:
                    for ax in ax_row:
                        subplot_pdfs = self.plot.settings \
                            ['{0:d}.{1:d}'.format(ax.row, ax.col)] \
                            ['pdfs']
                        if pdf.name in subplot_pdfs:
                            self.plot_constraint(ax.row, ax.col, pdf=pdf)

    def plot_constraint(self, row=None, col=None, pdf=None, 
                        parameters=None, limits=None):
        if pdf is None:
            pdf = self.choose_pdf(require_data=True)
        if pdf is not None:
            ax = self.plot.select_subplot(row=row, col=col)
            if len(ax.pdfs) == 0:
                self.set_up_subplot(ax.row, ax.col, pdf, parameters, limits)
            if pdf.settings['color'] is None:
                pdf.set_color()
            n_dim = len(ax.parameters)
            if n_dim == 1:
                self.plot.plot_1d_pdf(ax, pdf)
            elif n_dim == 2:
                self.plot.plot_2d_pdf(ax, pdf)
            plt.draw()

    def set_up_subplot(self, row, col, pdf, parameters, limits):
        ax = self.plot.axes[row][col]
        ax_settings = self.plot.settings['{0:d}.{1:d}'.format(row, col)]
        if parameters is None:
            ax.parameters = self.choose_parameters(pdf)
        else:
            ax.parameters = parameters
        if len(ax.parameters) > 2:
            print 'Number of parameters must be 1 or 2.'
            self.set_up_subplot(row, col, pdf, parameters, limits)
        ax_settings['parameters'] = ax.parameters
        self.plot.change_limits(ax=ax, limits=limits)

    def pdf_exists(self):
        if len(self.pdfs) > 0:
            return True
        else:
            return False

    def pdf_without_chain_exists(self):
        answer = False
        for pdf in self.pdfs:
            if not pdf.has_mcmc():
                answer = True
        return answer

    def pdf_with_data_exists(self):
        answer = False
        for pdf in self.pdfs:
            if pdf.chain is not None:
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


class Menu(object):

    def __init__(self, options=None, more=None, exit_str='Exit', header=None):
        self.choice = None
        self.i_choice=None
        self.exit = exit_str
        self.header = header
        self.prompt = '----\n> '
        if options:
            self.update_options(options)
        elif self.exit is not None:
            self.options = [self.exit]
        else:
            sys.exit('Cannot create Menu object without options' + \
                         'or exit string.')
        self.more = more

    def update_options(self, options):
        self.options = list(options)
        if self.exit is not None:
            self.options += [self.exit]

    def print_options(self):
        print
        if self.header:
            print self.header
        for i, opt in enumerate(self.options):
            print textwrap.fill(str(i) + ': ' + str(opt), 
                                initial_indent='',
                                subsequent_indent='    ')

    def get_choice(self, options=None):
        more_info = False
        if options:
            self.update_options(options)
        self.print_options()
        response = raw_input(self.prompt).strip()
        # check whether more info is requested
        if self.more and response[-1] == '?':
            more_info = True
            response = response[:-1]
        # get an integer
        try:
            i_choice = int(response)
        except ValueError:
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

    def get_order(self):
        self.print_options()
        response = raw_input(self.prompt).split()
        new_order = []
        for s in response:
            try:
                i = int(s)
            except ValueError:
                i = -1
            new_order.append(i)
        if sorted(new_order) != range(len(self.options)):
            print 'Not a valid order.'
            new_order = self.get_order()
        return new_order

    def add_option(self, position=None):
        pass


class Plot(object):

    def __init__(self):
        self.settings = {'n_rows': None, 'n_cols': None}
        self.figure = None

    def set_up_plot_grid(self, n_rows, n_cols):
        # assume all subplots occupy a single row and column for now
        # (also possible to use gridspec for plots that span multiple
        #  rows/columns - see http://matplotlib.org/users/gridspec.html)
        self.settings['n_rows'] = n_rows
        self.settings['n_cols'] = n_cols
        self.plot_grid = matplotlib.gridspec.GridSpec(n_rows, n_cols)
        figsize = plt.rcParams['figure.figsize']
        self.figure = plt.figure(figsize=(n_cols*figsize[0], 
                                          n_rows*figsize[1]))
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
                ax.pdfs = {}
                row_col_str = '{0:d}.{1:d}'.format(ax.row, ax.col)
                if row_col_str not in self.settings:
                    self.settings[row_col_str] = {'pdfs': {}}

    def select_subplot(self, row=None, col=None):
        row = self.get_row(default=row)
        col = self.get_col(default=col)
        return self.axes[row][col]

    # merge with get_col function?
    def get_row(self, default=None):
        n_rows = self.settings['n_rows']
        if default is None:
            if n_rows > 1:
                row = utils.get_input_integer( \
                    '\nSubplot row (0-' + str(n_rows - 1) + ')?\n> ',
                    error_text='Must choose an integer.')[0]
            else:
                row = 0
        else:
            row = default
        if row < 0 or row > n_rows - 1:
            print 'Row number is out of required range.'
            row = self.get_row()
        return row

    def get_col(self, default=None):
        n_cols = self.settings['n_cols']
        if default is None:
            if n_cols > 1:
                col = utils.get_input_integer( \
                    '\nSubplot column (0-' + str(n_cols - 1) + ')?\n> ',
                    error_text='Must choose an integer.')[0]
            else:
                col = 0
        else:
            col = default
        if col < 0 or col > n_cols - 1:
            print 'Column number is out of required range.'
            col = self.get_col()
        return col

    def change_limits(self, ax=None, limits=None):
        if ax is None:
            ax = self.select_subplot()
        ax_settings = self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)]
        if limits is None:
            x_limits = utils.get_input_float('\nPlot limits for ' + \
                                                 ax.parameters[0] + \
                                                 ' (lower upper)?\n> ', num=2)
        else:
            x_limits = limits[0]
        ax.set_xlim(x_limits)
        ax_settings['x_limits'] = x_limits
        if len(ax.parameters) == 2:
            if limits is None:
                y_limits = utils.get_input_float('\nPlot limits for ' + \
                                                     ax.parameters[1] + \
                                                     ' (lower upper)?\n> ', 
                                                 num=2)
            else:
                y_limits = limits[1]
            ax.set_ylim(y_limits)
            ax_settings['y_limits'] = y_limits

    def label_axes(self, ax=None, xlabel=None, ylabel=None):
        if ax is None:
            ax = self.select_subplot()
        if xlabel is None:
            new_label = raw_input('New x-axis label? (Press Enter to ' + \
                                      'keep the current label.)\n> ')
            if len(new_label) > 0:
                xlabel = new_label
        if xlabel is not None:
            ax.set_xlabel(xlabel)
            self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)]['xlabel'] = \
                xlabel
        if ylabel is None:
            new_label = raw_input('New y-axis label? (Press Enter to ' + \
                                      'keep the current label.)\n> ')
            if len(new_label) > 0:
                ylabel = new_label
        if ylabel is not None:
            ax.set_ylabel(ylabel)
            self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)]['ylabel'] = \
                ylabel

    def add_legend(self, ax=None):
        if ax is None:
            ax = self.select_subplot()
        if len(ax.parameters) == 1:
            ax.legend(frameon=False)
        elif len(ax.parameters) == 2:
            patches = []
            for pdf_name in ax.pdfs:
                pdf = ax.pdfs[pdf_name]
                if pdf.settings['color'] is None:
                    pdf.set_color()
                patches.append(mpatches.Patch(color=pdf.settings['color'], 
                                              label=pdf_name))
            ax.legend(handles=patches, frameon=False)
        self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)]['legend'] = True

    def plot_1d_pdf(self, ax, pdf, bins_per_sigma=5, p_min_frac=0.01,
                    color=None):
        ax.pdfs[pdf.name] = pdf
        ax_settings = self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)]
        set_pdfs = ax_settings['pdfs']
        if pdf.name not in set_pdfs:
            set_pdfs[pdf.name] = {}
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

        if color is None:
            if pdf.settings['color'] is None:
                color = (0, 0, 0)
            else:
                color = pdf.settings['color']

        ax.plot(bin_centers, pdf_1d, color=color, label=pdf.name)

        if 'xlabel' in ax_settings:
            xlabel = ax_settings['xlabel']
        else:
            xlabel = ax.parameters[0]
        if 'ylabel' in ax_settings:
            ylabel = ax_settings['ylabel']
        else:
            ylabel = 'P(' + ax.parameters[0] + ')'
        self.label_axes(ax, xlabel=xlabel, ylabel=ylabel)

    # break up into multiple methods and/or separate functions
    def plot_2d_pdf(self, ax, pdf, n_samples=5000, grid_size=(100, 100), 
                    smoothing=1.0, contour_pct=(95.45, 68.27),
                    colors=None, layer=None):
        ax.pdfs[pdf.name] = pdf
        ax_settings = self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)]
        set_pdfs = ax_settings['pdfs']
        if pdf.name not in set_pdfs:
            set_pdfs[pdf.name] = {}
        # plot new contour over others unless a layer is specified
        ax.set_rasterization_zorder(0)
        if layer is None:
            if 'layer' not in set_pdfs[pdf.name]:
                for p in set_pdfs:
                    if 'layer' in set_pdfs[p]:
                        set_pdfs[p]['layer'] -= 1
                set_pdfs[pdf.name]['layer'] = -1
        else:
            for p in set_pdfs:
                if 'layer' in set_pdfs[p] and set_pdfs[p]['layer'] <= layer:
                    set_pdfs[p]['layer'] -= 1
            set_pdfs[pdf.name]['layer'] = layer

        contour_data = pdf.load_contour_data(n_samples, grid_size, smoothing, 
                                             contour_pct)
        if contour_data is None:

            par_x = pdf.get_chain_parameter(ax.parameters[0])
            par_y = pdf.get_chain_parameter(ax.parameters[1])

            # draw random samples from the chains with probability
            # proportional to multiplicity weight
            mult = pdf.chain.multiplicity
            indices = np.random.choice(len(mult), n_samples,
                                       p=mult/np.sum(mult))
            x_samples = par_x.values[indices]
            y_samples = par_y.values[indices]

            #ax.scatter(x_samples, y_samples)

            # estimate PDF with KDE
            xy_samples = np.vstack((x_samples, y_samples))
            kde = stats.gaussian_kde(xy_samples)
            kde_bw = smoothing * kde.covariance_factor()
            kde.set_bandwidth(kde_bw)
            pdf_values = kde(xy_samples)

            # evaluate the PDF on a regular grid
            x_border = 0.05*(np.max(x_samples) - np.min(x_samples))
            x_limits = [np.min(x_samples)-x_border, np.max(x_samples)+x_border]
            y_border = 0.05*(np.max(y_samples) - np.min(y_samples))
            y_limits = [np.min(y_samples)-y_border, np.max(y_samples)+y_border]
            x_grid = np.linspace(*x_limits, num=grid_size[0])
            y_grid = np.linspace(*y_limits, num=grid_size[1])
            X_2d, Y_2d = np.meshgrid(x_grid, y_grid)
            xy_grid = np.transpose(np.vstack((X_2d.flatten(), Y_2d.flatten())))
            pdf_grid = np.array([kde(xy) for xy in xy_grid])
            Z_2d = np.reshape(pdf_grid, X_2d.shape)

            # compute contour levels
            contour_levels = []
            for cl in contour_pct:
                contour_levels.append(stats.scoreatpercentile(pdf_values, 
                                                              100.0-cl))
        
            pdf.save_contour_data(n_samples, grid_size, smoothing, 
                                  contour_pct, contour_levels,
                                  X_2d, Y_2d, Z_2d)

        else:
            contour_levels, X_2d, Y_2d, Z_2d = contour_data

        # set contour colors
        if colors is None:
            if pdf.settings['color'] is None:
                colors = utils.color_gradient((0, 0, 0), len(contour_pct))
            else:
                colors = utils.color_gradient(pdf.settings['color'], 
                                              len(contour_pct))

        # plot contours
        ax.contourf(X_2d, Y_2d, Z_2d, levels=sorted(contour_levels)+[np.inf],
                    colors=colors, zorder=set_pdfs[pdf.name]['layer'],
                    alpha=0.7, rasterized=True)

        if 'xlabel' in ax_settings:
            xlabel = ax_settings['xlabel']
        else:
            xlabel = ax.parameters[0]
        if 'ylabel' in ax_settings:
            ylabel = ax_settings['ylabel']
        else:
            ylabel = ax.parameters[1]
        self.label_axes(ax, xlabel=xlabel, ylabel=ylabel)

    def change_layer_order(self, ax=None):
        if ax is None:
            ax = self.select_subplot()
        set_pdfs = self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)]['pdfs']
        # find current order
        pdfs = []
        layers = []
        for pdf in set_pdfs:
            if 'layer' in set_pdfs[pdf]:
                layers.append(set_pdfs[pdf]['layer'])
                pdfs.append(pdf)
        old_layers = list(zip(*sorted(zip(pdfs, layers), 
                                      key=lambda x: x[1],
                                      reverse=True))[0])
        # get new order (e.g. 1 0 2)
        m = Menu(options=old_layers, exit_str=None, 
                 header='Enter a new order for the constraints.\n' + \
                     'The current order (top to bottom) is:')
        new_layer_order = m.get_order()
        new_layer_order.reverse()

        # should clear subplot before plotting contours again,
        # but don't want to change other plot elements

        # re-plot pdfs in new order
        for i in new_layer_order:
            self.plot_2d_pdf(ax, ax.pdfs[old_layers[i]], layer=-1)


class PostPDF(object):

    def __init__(self, name, model):
        self.settings = {}
        self.rename(name)
        self.model = model
        self.chain = None
        self.chain_files = []
        self.likelihoods = {}
        for attr in ['name', 'model', 'chain_files']:
            self.settings[attr] = getattr(self, attr)
        self.settings['parameters'] = {}
        self.settings['likelihoods'] = {}
        self.settings['contour_data_files'] = []
        self.settings['color'] = None
        self.settings['derived_parameters'] = {}

    def has_mcmc(self):
        if len(self.chain_files) > 0:
            return True
        else:
            return False

    def add_chain(self, name, files):
        # check if already have chain, if files exist, if name is unique

        # check if model is consistent?

        self.chain_files.extend(files)
        self.settings['chain_name'] = name
        self.chain = MCMCChain(name, files)

        # if likelihood already included in constraint
        # when chain is added, need to check whether likelihood parameters
        # are also in the chain; also importance sample

        self.add_parameters(self.chain.parameters)
        self.settings['contour_data_files'] = []

    def add_likelihood(self, name, **kwargs):
        # check if name is unique (not already in self.likelihoods)

        self.settings['likelihoods'][name] = kwargs
        kwargs['invert'] = False
        if kwargs['form'] == 'Gaussian':
            self.add_gaussian_likelihood(name, **kwargs)
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
        self.settings['contour_data_files'] = []

    def add_gaussian_likelihood(self, name, **kwargs):
        self.likelihoods[name] = GaussianLikelihood(invert=kwargs['invert'])
        self.likelihoods[name].set_parameter_means( \
            **dict(zip(kwargs['parameters'], kwargs['means'])))
        self.likelihoods[name].set_covariance(kwargs['covariance'],
                                              kwargs['parameters'])

    def add_parameters(self, new_parameters):
        print self.settings['parameters']
        for p in new_parameters:
            if p not in self.settings['parameters']:
                self.settings['parameters'][p] = []
        print self.settings['parameters']

    def display_parameters(self):
        print textwrap.fill(', '.join(self.settings['parameters'].keys()))

    def get_chain_parameter(self, parameter):
        if parameter not in self.settings['parameters']:
            sys.exit('The PDF ' + self.name + \
                         ' does not have the parameter ' + str(parameter))
        # find the index for the parameter
        index = np.where(np.array(self.chain.parameters) == parameter)[0][0]
        # create a ChainParameter object for each parameter
        return ChainParameter(self.chain, index)

    def add_derived_parameter(self, new_name, f_str, par_names, 
                              par_indices=None):
        # only works for simple parameter combinations - what about
        # more complicated functions like D_A(z)?
        f = lambdify(symbols([str(x) for x in par_names]), 
                     sympify(f_str), 'numpy')
        # handle errors from the sympy operations

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
        if len(par_indices) == len(par_names):
            self.settings['parameters'][new_name] = []
            self.chain.parameters.append(new_name)
            self.chain.column_names.append(new_name)
            new_column = f(\
                *[self.chain.samples[:,i+self.chain.first_par_column] \
                      for i in par_indices])
            self.chain.samples = np.hstack((self.chain.samples, 
                                      np.array([new_column]).T))
            self.settings['derived_parameters'][new_name] = {
                'function': f_str,
                'parameters': par_names,
                'indices': par_indices}

    def compute_1d_stats(self, parameters):
        # how to do this if there is no chain, only likelihoods?

        for p in parameters:
            cp = self.get_chain_parameter(p)
            fmt_str = '{0:s} = {1:.3g} +/- {2:.3g}'
            print fmt_str.format(p, cp.mean(), cp.standard_deviation())

    def save_contour_data(self, n_samples, grid_size, smoothing, 
                          contour_pct, contour_levels, X, Y, Z):
        filename = os.path.join('Data', time.strftime('%Y-%m-%d'), 
                                time.strftime('%Z.%H.%M.%S') + '_contour.txt')
        writer = utils.open_with_path(filename, 'w')
        header = str(n_samples) + ' # n_samples\n' + \
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

    def load_contour_data(self, n_samples, grid_size, smoothing, contour_pct):
        contour_data = None
        for f in self.settings['contour_data_files']:
            reader = utils.open_if_exists(f, 'r')
            test_n_samples = int(float(reader.readline().split('#')[0]))
            test_grid_size = [int(float(x)) for x in \
                                  reader.readline().split('#')[0].split()]
            test_smoothing = float(reader.readline().split('#')[0])
            test_contour_pct = [float(x) for x in \
                                  reader.readline().split('#')[0].split()]

            match = (test_n_samples == n_samples) and \
                np.all([x == y for x, y in \
                            zip(test_grid_size, grid_size)]) and \
                test_smoothing == smoothing and \
                np.all([x == y for x, y in \
                            zip(test_contour_pct, contour_pct)])

            if match:
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


class MCMCChain(object):

    def __init__(self, name, chain_files, 
                 mult_column=0, lnlike_column=1, first_par_column=2,
                 paramname_file=None):
        self.rename(name)
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
                j = np.where(np.array(self.parameters) == p)[0] + \
                    self.first_par_column
                parameter_values[p] = self.samples[i, j]
            chisq = likelihood.chi_squared(**parameter_values)
            if invert:
                self.multiplicity[i] *= np.exp(0.5*chisq)
            else:
                self.multiplicity[i] *= np.exp(-0.5*chisq)
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
        self.parameters = parameters

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

