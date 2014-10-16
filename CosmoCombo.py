#!/usr/bin/env python

import classes
import time
import argparse
from collections import OrderedDict

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
                                  save=not args.no_log)
    else:
        session = classes.Session(args.log, save=not args.no_log)
else:
    session = classes.Session('session.' + current_time + '.log',
                              save=not args.no_log)

# - set up a new plot
# - define a new chain or likelihood function
# - get 1D stats for a parameter
# - add some combination of constraints to a subplot (if plot exists)
# - change plot appearance (labels, font size, etc.) (if plot exists)
# - save the current log file and exit (verify name to use first)
options = (('Set up new joint constraint', 
            {'action': session.set_up_pdf, 
             'condition': lambda: True}),
           ('Copy a constraint',
            {'action': session.copy_pdf,
             'condition': session.pdf_exists}),
           ('List constraint properties',
            {'action': session.print_pdf_settings,
             'condition': session.pdf_exists}),
           ('Rename a constraint',
            {'action': session.rename_pdf,
             'condition': session.pdf_exists}),
           ('Save a constraint',
            {'action': session.save_pdf,
             'condition': session.pdf_with_data_exists}),
           ('Add MCMC chain', 
            {'action': session.add_chain, 
             'condition': session.pdf_without_chain_exists}),
           ('Add a likelihood function',
            {'action': session.add_likelihood,
             'condition': session.pdf_exists}),
           ('Add a derived parameter',
            {'action': session.add_derived_parameter,
             'condition': session.pdf_with_data_exists}),
           ('Compute marginalized 1D statistics',
            {'action': session.compute_1d_stats,
             'condition': session.pdf_with_data_exists}),
           ('Set up plot', 
            {'action': session.set_up_plot, 
             'condition': lambda: not session.plot_exists()}),
           ('Plot constraint', 
            {'action': session.plot_constraint, 
             'condition': session.plot_and_pdf_with_data_exist}),
           ('Change plot appearance',
            {'action': session.change_plot,
             'condition': session.plot_exists}),
           ('Save plot',
            {'action': session.save_plot,
             'condition': session.plot_exists}),
           ('Delete session history',
            {'action': session.delete_history,
             'condition': lambda: True}))

session.start_interactive(options=OrderedDict(options))


