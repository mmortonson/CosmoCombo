import numpy as np
from scipy import stats
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils
from menu import Menu


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
        if limits is None:
            if len(ax.parameters) == 2:
                y_limits = utils.get_input_float('\nPlot limits for ' + \
                                                     ax.parameters[1] + \
                                                     ' (lower upper)?\n> ', 
                                                 num=2)
            else:
                y_limits = utils.get_input_float('\nPlot limits for ' + \
                                                     'y axis' + \
                                                     ' (lower upper)?\n> ', 
                                                 num=2)
        elif len(limits) == 2:
            y_limits = limits[1]
        else:
            y_limits = None
        if y_limits is not None:
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

    def plot_1d_pdf(self, ax, pdf, n_samples=5000, grid_size=100,
                    smoothing=1.0, p_min_frac=0.01,
                    color=None):
        ax.pdfs[pdf.name] = pdf
        ax_settings = self.settings['{0:d}.{1:d}'.format(ax.row, ax.col)]
        set_pdfs = ax_settings['pdfs']
        if pdf.name not in set_pdfs:
            set_pdfs[pdf.name] = {}
        parameter = pdf.get_chain_parameter(ax.parameters[0])

        # draw random samples from the chains with probability
        # proportional to multiplicity weight
        mult = pdf.chain.multiplicity
        indices = np.random.choice(len(mult), n_samples,
                                   p=mult/np.sum(mult))
        p_samples = parameter.values[indices]

        # estimate PDF with KDE
        kde = stats.gaussian_kde(p_samples)
        kde_bw = smoothing * kde.covariance_factor()
        kde.set_bandwidth(kde_bw)

        # evaluate the PDF on a regular grid
        border = 0.05*(np.max(p_samples) - np.min(p_samples))
        p_limits = [np.min(p_samples)-border, np.max(p_samples)+border]
        grid = np.linspace(*p_limits, num=grid_size)
        pdf_1d = kde(grid)
        
        # trim points at either end with prob./max(pdf_1d) < p_min_frac
        while pdf_1d[0] < p_min_frac*pdf_1d.max():
            pdf_1d = np.delete(pdf_1d, 0)
            grid = np.delete(grid, 0)
        while pdf_1d[-1] < p_min_frac*pdf_1d.max():
            pdf_1d = np.delete(pdf_1d, -1)
            grid = np.delete(grid, -1)

        if color is None:
            if pdf.settings['color'] is None:
                color = (0, 0, 0)
            else:
                color = pdf.settings['color']

        ax.plot(grid, pdf_1d, color=color, label=pdf.name)

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

        contour_data = pdf.load_contour_data(ax.parameters, n_samples, 
                                             grid_size, smoothing, 
                                             contour_pct)
        if contour_data is None:

            print 'Computing new contours...'
            
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
        
            pdf.save_contour_data(ax.parameters, n_samples, grid_size, 
                                  smoothing, contour_pct, contour_levels,
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
