#!/usr/bin/env python

import classes
import time
import argparse
from collections import OrderedDict

#current_date = time.strftime('%Y-%m-%d')
current_time = time.strftime('%Z.%H.%M.%S')

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--script', 
                    help='name of new or existing script')
parser.add_argument('-d', '--dir',
                    help='directory to search for existing script')
args = parser.parse_args()

if args.script:
    if args.dir:
        script = classes.Script(args.script, path=args.dir)
    else:
        script = classes.Script(args.script)
else:
    script = classes.Script('session.' + current_time + '.scr')

# - set up a new plot
# - define a new chain or likelihood function
# - get 1D stats for a parameter
# - add some combination of constraints to a subplot (if plot exists)
# - change plot appearance (labels, font size, etc.) (if plot exists)
# - save the current script and exit (verify name to use first)
options = (('Add data', {'action': script.add_data, 
                         'condition': lambda: True}),
           ('Set up plot', {'action': script.set_up_plot, 
                            'condition': lambda: True}),
           ('Plot constraint', {'action': script.plot_constraint, 
                               'condition': script.plot_exists}))

script.start_interactive(options=OrderedDict(options))


