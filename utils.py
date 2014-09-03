import sys
import os
import os.path
import errno
import numpy as np

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

def open_if_exists(file, flag):
    """
    Check if a file exists. If so open for reading or writing
    (set by flag). If not print error message and exit.
    """
    if os.path.isfile(file):
        return open(file, flag)
    else:
        sys.exit('File ' + str(file) + ' not found.')

def get_input_integer(prompt, num=1, 
                      error_text='Input must be integers.',
                      error_action='retry'):
    values = []
    success = True
    user_input = raw_input(prompt).split()
    if len(user_input) != num:
        print 'Expected ' + str(num) + ' numbers.'
        success = False
    else:
        for i in range(num):
            try:
                next_value = int(user_input[i])
                values.append(next_value)
            except ValueError as e:
                print error_text
                success = False
    if not success:
        if error_action == 'retry':
            values = get_input_integer(prompt, num=num, error_text=error_text,
                                       error_action=error_action)
        else:
            sys.exit()
    return values

def get_input_float(prompt, num=1, 
                      error_text='Input must be floats.',
                      error_action='retry'):
    values = []
    success = True
    user_input = raw_input(prompt).split()
    if len(user_input) != num:
        print 'Expected ' + str(num) + ' numbers.'
        success = False
    else:
        for i in range(num):
            try:
                next_value = float(user_input[i])
                values.append(next_value)
            except ValueError as e:
                print error_text
                success = False
    if not success:
        if error_action == 'retry':
            values = get_input_float(prompt, num=num, error_text=error_text,
                                       error_action=error_action)
        else:
            sys.exit()
    return values

def color_gradient(rgb_tuple, num):
    colors = [rgb_tuple]
    for i in range(1, num):
        colors.insert(0, tuple(np.array(rgb_tuple) + \
                                   i*(1.-np.array(rgb_tuple)) / float(num)))
    return colors
