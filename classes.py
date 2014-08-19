class Script(object):

    def __init__(self, name, path=None):
        name = self.name
        # if path is given, search it for the named script (error if not found)

        # if no path given, search multiple paths for named script

        # if existing script found, set up environment using old settings

        # if script not found, notify that new script is assumed

    def start_interactive(self):
        # give user a menu of options, e.g.
        # - set up a new plot
        # - define a new chain or likelihood function
        # - get 1D stats for a parameter
        # - add some combination of constraints to a subplot (if plot exists)
        # - change plot appearance (labels, font size, etc.) (if plot exists)
        # - save the current script and exit (verify name to use first)
        pass
