#!/usr/bin/env python

import classes
import time
import argparse
from collections import OrderedDict

#current_date = time.strftime('%Y-%m-%d')
current_time = time.strftime('%Z.%H.%M.%S')

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--log', 
                    help='name of new or existing session log file')
parser.add_argument('-d', '--dir',
                    help='directory to search for existing log file')
parser.add_argument('--no_log', action='store_true', 
                    help='run session without recording a new log file')
args = parser.parse_args()

if args.log:
    if args.dir:
        session = classes.Session(args.log, path=args.dir, 
                                  save_log=not args.no_log)
    else:
        session = classes.Session(args.log, save_log=not args.no_log)
else:
    session = classes.Session('session.' + current_time + '.log',
                              save_log=not args.no_log)

# - set up a new plot
# - define a new chain or likelihood function
# - get 1D stats for a parameter
# - add some combination of constraints to a subplot (if plot exists)
# - change plot appearance (labels, font size, etc.) (if plot exists)
# - save the current log file and exit (verify name to use first)
options = (('Set up new joint constraint', 
            {'action': session.set_up_pdf, 
             'condition': lambda: True}),
           ('Rename a constraint',
            {'action': session.rename_pdf,
             'condition': session.pdf_exists}),
           ('Add MCMC chain', 
            {'action': session.add_chain, 
             'condition': session.pdf_without_chain_exists}),
           ('Compute marginalized 1D statistics',
            {'action': session.compute_1d_stats,
             'condition': session.pdf_with_data_exists}),
           ('Set up plot', 
            {'action': session.set_up_plot, 
             'condition': lambda: not session.plot_exists()}),
           ('Plot constraint', 
            {'action': session.plot_constraint, 
             'condition': session.plot_and_pdf_with_data_exist}))

session.start_interactive(options=OrderedDict(options))


