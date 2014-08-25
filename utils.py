import os, errno

def check_path(file):
    """ Check whether a path to a file exists, and if not, create it.
        http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    path = os.path.dirname(file)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

def open_with_path(file, flag):
    """ Open file for reading or writing (set by flag), 
        creating the path to the file if it doesn't exist. """
    try:
        open_file = open(file, flag)
    except IOError as e:
        if e.errno == errno.ENOENT:
            check_path(file)
            open_file = open_with_path(file, flag)
        else:
            raise

    return open_file
