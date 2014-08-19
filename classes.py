import os.path

# rename Session?
class Script(object):

    def __init__(self, name, path=None):
        self.name = name
        # if path is given, search it for the named script (error if not found)

        # if no path given, search multiple paths for named script

        # if existing script found, set up environment using old settings

        # if script not found, notify that new script is assumed

    def start_interactive(self, options=None):
        print 'Session name:', self.name
        menu = Menu(options=self.active_options(options))
        menu.get_choice()
        while menu.choice != menu.exit:
            if options[menu.choice]['condition']():
                options[menu.choice]['action']()
            menu.get_choice()
        self.save_and_exit()

    def active_options(self, all_options):
        options = []
        for key in all_options:
            if all_options[key]['condition']():
                options.append(key)
        return options

    def save_and_exit(self):
        pass

    def add_data(self):
        print 'Adding data'

    def set_up_plot(self):
        print 'Setting up plot'

    def plot_constraint(self):
        print 'Plotting constraint'

    def plot_exists(self):
        return False


class Settings(object):

    def __init__(self):
        pass


class Menu(object):

    def __init__(self, options=None):
        self.exit = 'Exit'
        self.choice = None
        self.prompt = '----\n> '
        self.options = [self.exit]
        if options:
            self.options += list(options)

    def get_choice(self):
        for i, opt in enumerate(self.options):
            print str(i) + ': ' + opt
        # get an integer
        try:
            i_choice = int(raw_input(self.prompt))
        except ValueError as e:
            i_choice = -1
        # check that the integer corresponds to a valid option
        if i_choice >=0 and i_choice < len(self.options):
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

    def __init__(self, name, model, chain_files=None):
        self.name = name
        self.model = model

        # use chain class instead????
        if chain_files:
            self.load_chain(chain_files)

    def load_chain(self, files):
        # check if files exist

        # check if model is consistent
        pass
