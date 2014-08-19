#!/usr/bin/env python

import classes
import argparse

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
    script = classes.Script('session' + local_time + '.scr')

script.start_interactive()

