import os.path
import time
import glob
import json
from copy import deepcopy
import textwrap
from collections import OrderedDict
import itertools
import numpy as np
import matplotlib.pyplot as plt
import utils
import cosmology
from menu import Menu
from post_pdf import PostPDF
from plot import Plot


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
            self.log_file = old_log_file
            self.load_log(log_reader)
        # if not found, notify that new log file is assumed
        elif self.save:
            print 'No log file found. Creating new file:\n    ' + \
                self.log_file

        for attr in ['name', 'log_file']:
            self.settings[attr] = getattr(self, attr)
            
    def load_log(self, reader):
        log_settings = json.loads(reader.read())
        reader.close()
        for pdf_name in log_settings['pdfs']:
            d = log_settings['pdfs'][pdf_name]
            self.set_up_pdf(settings=d)
            pdf = self.choose_pdf(name=d['name'])
            for pdf_element in list(d['order']):
                if pdf_element[0] == 'chain':
                    print 'Adding chain: ' + d['chain_name']
                    pdf.add_chain(d['chain_name'], d['chain_files'],
                                  d['chain_burn_in'], d['chain_mult_column'],
                                  d['chain_lnlike_column'],
                                  d['chain_first_par_column'],
                                  d['chain_paramname_file'],
                                  d['chain_params_in_header'],
                                  update_order=False)
                    count = 0
                    for chain in list(self.history['chains']):
                        if chain[1] == d['chain_name']:
                            count = chain[0]
                            self.history['chains'].remove(chain)
                    self.history['chains'].append([count, d['chain_name'],
                                                   d['chain_files'],
                                                   d['chain_burn_in'], 
                                                   d['chain_mult_column'],
                                                   d['chain_lnlike_column'],
                                                   d['chain_first_par_column'],
                                                   d['chain_paramname_file'],
                                                   d['chain_params_in_header']])
                elif pdf_element[0] == 'likelihood':
                    lk = pdf_element[1]
                    print 'Adding likelihood: ' + lk
                    pdf.add_likelihood(lk, update_order=False,
                                       **d['likelihoods'][lk])
                    count = 0
                    for hist_lk in list(self.history['likelihoods']):
                        if hist_lk[0] == lk:
                            count = hist_lk[1]
                            self.history['likelihoods'].remove(hist_lk)
                    self.history['likelihoods'].append( \
                        [lk, count, d['likelihoods'][lk]])
                elif pdf_element[0] == 'derived_parameter':
                    p = pdf_element[1]
                    print 'Adding derived parameter: ' + p
                    p_dict = d['derived_parameters'][p]
                    pdf.add_derived_parameter(p, p_dict['function'],
                                              p_dict['parameters'],
                                              p_dict['indices'],
                                              update_order=False)
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

            #self.plot.plot_grid.tight_layout(self.plot.figure)
            self.plot.figure.set_tight_layout(True)
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

    def delete_history(self):
        m = Menu(options=['Likelihoods', 'Chains', 'All'], exit_str='Cancel')
        m.get_choice()
        if m.choice == 'Likelihoods' or m.choice == 'All':
            self.history['likelihoods'] = []
        if m.choice == 'Chains' or m.choice == 'All':
            self.history['chains'] = []

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
            m = Menu(options=['LCDM', 'wCDM'], exit_str=None,
                     header='Choose a model class:')
            m.get_choice()
            model = m.choice
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

    def save_pdf(self):
        pdf = self.choose_pdf(require_data=True)
        if pdf is not None:
            file_prefix = os.path.join('Chains', time.strftime('%Y-%m-%d'),
                                       'chain_' + '_'.join(pdf.name.split()) \
                                       + time.strftime('_%Z.%H.%M.%S'))

            param_filename = file_prefix + '.paramnames'
            writer = utils.open_with_path(param_filename, 'w')
            writer.write('\n'.join(pdf.chain.parameters) + '\n')
            writer.close()
            pdf_filename = file_prefix + '_1.txt'
            writer = utils.open_with_path(pdf_filename, 'w')
            header = ''
            for key in sorted(pdf.settings):
                header += textwrap.fill(key + ': ' + str(pdf.settings[key]),
                                    subsequent_indent='    ') + '\n'
            np.savetxt(writer, pdf.chain.samples, fmt='%.6e', header=header)
            writer.close()
            print 'Saved as ' + pdf_filename

    def add_chain(self):
        pdf = self.choose_pdf(require_no_chain=True)
        # select from list of currently defined chains (from all pdfs),
        # previously used chains (read from file),
        # or get files and name for new chain (if new, save to file)
        if pdf is not None:
            if len(self.history['chains']) > 0:
                chain_history = sorted(self.history['chains'],
                                       key=lambda x: x[0], reverse=True)
                sort_indices = sorted(range(len(self.history['chains'])),
                                      key=lambda i: \
                                          self.history['chains'][i][0],
                                      reverse=True)
                options = [ch[1] for ch in chain_history]
                details = ['Chains:\n' + '\n'.join(
                        [textwrap.fill(s, initial_indent='    ',
                                       subsequent_indent='        ') \
                             for s in sorted(ch[2])]) \
                               for ch in chain_history]
                m = Menu(options=options, more=details,
                         exit_str='New chain',
                         header='Choose a chain:\n' + \
                             '(add ? to the number to get ' + \
                             'more info on a chain)')
                m.get_choice()
                if m.choice == m.exit:
                    pdf.add_chain(*self.define_new_chain())
                else:
                    self.history['chains'][sort_indices[m.i_choice]][0] += 1
                    pdf.add_chain(*self.history['chains'] \
                                       [sort_indices[m.i_choice]][1:])
            else:
                pdf.add_chain(*self.define_new_chain())

    def define_new_chain(self):
        files = []
        while len(files) == 0:
            for f in raw_input('\nChain file names?\n> ').split():
                new_files = glob.glob(f)
                if len(new_files) == 0:
                    print 'No files matching ' + f + ' found.'
                elif not np.array([os.path.isfile(nf) \
                                   for nf in new_files]).all():
                    print 'Some of the files specified are not valid.'
                else:
                    files += new_files
        name = raw_input('Label for chain?\n> ')
        burn_in = utils.get_input_float( \
            'Burn-in fraction or number of samples?\n> ')[0]
        mult_column = utils.get_input_integer( \
            'Multiplicity column? (Enter -1 if none.)\n> ')[0]
        if mult_column < 0:
            mult_column = None
        lnlike_column = utils.get_input_integer( \
            'Log likelihood column? (Enter -1 if none.)\n> ')[0]
        if lnlike_column < 0:
            lnlike_column = None
        first_par_column = utils.get_input_integer( \
            'Column of first chain parameter?\n> ')[0]
        m = Menu(options=['File named as chain label + .paramnames',
                          'A different file',
                          'Header of chain files'],
                 exit_str=None,
                 header='Where are the chain parameter names?')
        m.get_choice()
        paramname_file = None
        params_in_header = False
        if m.i_choice == 1:
            paramname_file = raw_input('Enter file name:\n> ')
        elif m.i_choice == 2:
            params_in_header = True
        chain_settings = (name, files, burn_in, 
                          mult_column, lnlike_column, first_par_column,
                          paramname_file, params_in_header)
        # check if name is already in history; if so, replace with new
        for chain in list(self.history['chains']):
            if chain[0] == name:
                self.history['chains'].remove(chain)
        self.history['chains'].append([1] + list(chain_settings))
        return chain_settings

    # merge with add_chain?
    def add_likelihood(self):
        pdf = self.choose_pdf()
        if pdf is not None:
            if len(self.history['likelihoods']) > 0:
                lk_history = sorted(self.history['likelihoods'], 
                                    key=lambda x: x[1], reverse=True)
                sort_indices = sorted(range(len(self.history['likelihoods'])),
                                      key=lambda i: \
                                          self.history['likelihoods'][i][1],
                                      reverse=True)
                options = [lk[0] for lk in lk_history]
                details = ['Likelihoods:\n' + '\n'.join(
                        ['    ' + s + ': ' +  str(lk[2][s]) \
                             for s in sorted(lk[2].keys())]) \
                               for lk in lk_history]
                m = Menu(options=options, more=details,
                         exit_str='New likelihood',
                         header='Choose a likelihood function:\n' + \
                             '(add ? to the number to get ' + \
                             'more info on a likelihood)')
                m.get_choice()
                if m.choice == m.exit:
                    new_lk = self.define_new_likelihood(pdf)
                    pdf.add_likelihood(new_lk[0], **new_lk[2])
                else:
                    self.history['likelihoods'][sort_indices[m.i_choice]][1] \
                        += 1
                    lk = self.history['likelihoods'][sort_indices[m.i_choice]]
                    pdf.add_likelihood(lk[0], **lk[2])
            else:
                new_lk = self.define_new_likelihood(pdf)
                pdf.add_likelihood(new_lk[0], **new_lk[2])

    def define_new_likelihood(self, pdf):
        # if chain exists, choose some parameters from there
        # can also add new parameters
        name = raw_input('Label for likelihood?\n> ')
        m = Menu(options=['Flat', 'Gaussian', 
                          'SNe (Gaussian with analytic marginalization)',
                          'Inverse Gaussian'], 
                 exit_str=None,
                 header='Choose the form of the likelihood:')
        m.get_choice()
        form = m.choice
        if form[:3] == 'SNe':
            form = 'SNe'
        lk_dict = {'form': form}
        priors = []
        if form != 'SNe':
            parameters = pdf.choose_parameters(allow_extra_parameters=True)
            pdf.add_parameters(parameters)
            lk_dict['parameters'] = parameters

        if form == 'Flat':
            for p in parameters:
                if len(pdf.settings['parameters'][p]) != 2:
                    new_prior = utils.get_input_float( \
                        'Enter lower and upper limits of the prior on ' + \
                            p + ':\n> ', num=2)
                    priors.append(sorted(new_prior))
                else:
                    priors.append(pdf.settings['parameters'][p])

        elif form in ['Gaussian', 'Inverse Gaussian']:
            means = utils.get_input_float('Enter mean values:\n> ',
                                          num=len(parameters))
            variances = utils.get_input_float('Enter variances:\n> ',
                                              num=len(parameters))
            for i, p in enumerate(parameters):
                if len(pdf.settings['parameters'][p]) != 2:
                    priors.append([means[i] - 5.*np.sqrt(variances[i]),
                                   means[i] + 5.*np.sqrt(variances[i])])
                else:
                    priors.append(pdf.settings['parameters'][p])
            covariance = np.diag(variances)
            for i, j in itertools.combinations(range(len(parameters)), 2):
                covariance[i, j] = utils.get_input_float( \
                    'Cov(' + parameters[i] + ', ' + parameters[j] + ')?\n> ')[0]
                covariance[j, i] = covariance[i, j]
            lk_dict['means'] = means
            lk_dict['covariance'] = covariance.tolist()

        elif form == 'SNe':
            sn_z_mu_file = raw_input('Name of file with mean SN distance ' + \
                                         'modulus vs. z?\n> ')
            sn_z, sn_mu = np.loadtxt(sn_z_mu_file, unpack=True)
            lk_dict['means'] = list(sn_mu)
            sn_cov_file = raw_input('Name of file with SN covariance ' + \
                                        'matrix?\n> ')
            sn_cov = np.loadtxt(sn_cov_file)
            lk_dict['covariance'] = sn_cov.tolist()
            sn_parameters = []
            for i, z in enumerate(sn_z):
                p_sn_z = 'mu_SN_z' + str(z)
                sn_parameters.append(p_sn_z)
                priors.append([sn_mu[i] - 5.*np.sqrt(sn_cov[i,i]),
                               sn_mu[i] + 5.*np.sqrt(sn_cov[i,i])])
            pdf.add_parameters(sn_parameters)
            lk_dict['parameters'] = sn_parameters


        lk_dict['priors'] = priors
        new_lk = (name, 1, lk_dict)

        # check if name is already in history; if so, replace with new
        for lk in list(self.history['likelihoods']):
            if lk[0] == name:
                self.history['likelihoods'].remove(lk)
        self.history['likelihoods'].append(list(new_lk))
        return new_lk

    def add_derived_parameter(self):
        # add option to add parameter to all chains?
        pdf = self.choose_pdf(require_data=True)
        if pdf is not None:
            name_is_new = False
            while not name_is_new:
                name = raw_input('\nName of the new parameter?\n> ')
                if name in pdf.settings['parameters'].keys():
                    print 'There is already a parameter with that name.'
                else:
                    name_is_new = True
            print '\nExisting parameters required to ' + \
                'compute the new parameter?'
            par_names = pdf.choose_parameters()
            f_str = ''
            while f_str == '':
                f_str = raw_input('\nEnter the function to use for the new' + \
                                      ' parameter,\nusing the existing ' + \
                                      'parameter names, standard functions,' + \
                                      ' and\npredefined cosmology functions' + \
                                      ' (press Enter to see a list):\n> ')
                if f_str == '':
                    print 'Available cosmology functions:'
                    print '  ' + '\n  '.join(cosmology.functions.keys())
            pdf.add_derived_parameter(name, f_str, par_names)

    def compute_1d_stats(self):
        pdf = self.choose_pdf(require_data=True)
        if pdf is not None:
            parameters = pdf.choose_parameters()
            m = Menu(options=['mean and standard deviation',
                              'equal-tail limits',
                              'upper limit',
                              'lower limit'], 
                     exit_str=None, header='Choose statistics to compute:')
            m.get_choice()
            pdf.compute_1d_stats(parameters, stats=m.choice)

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
            n_rows, n_cols = (0, 0)
            while n_rows < 1 or n_cols < 1:
                n_rows = utils.get_input_integer( \
                    '\nNumber of subplot rows?\n> ',
                    error_text=e_str.format('rows'))[0]
                n_cols = utils.get_input_integer( \
                    'Number of subplot columns?\n> ',
                    error_text=e_str.format('columns'))[0]
                if n_rows < 1 or n_cols < 1:
                    print 'Must have > 0 rows and columns.'

        self.plot.set_up_plot_grid(n_rows, n_cols)
        #self.plot.plot_grid.tight_layout(self.plot.figure)
        self.plot.figure.set_tight_layout(True)
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
        try:
            # bug in matplotlib 1.4.0 prevents this from working
            # (https://github.com/matplotlib/matplotlib/pull/3434)
            plt.savefig(plot_file, format='eps', bbox_inches='tight')
        except:
            plt.savefig(plot_file, format='eps')
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
            #self.plot.plot_grid.tight_layout(self.plot.figure)
            self.plot.figure.set_tight_layout(True)
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
            ax.parameters = pdf.choose_parameters()
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
