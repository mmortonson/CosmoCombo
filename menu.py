import sys
import textwrap


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
