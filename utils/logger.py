from contextlib import contextmanager
import sys, os
from time import perf_counter

"""
Module that defines functions for making logs during running programs and experiments.
"""


def print_message(message, kind='simple'):
    """
    Print current computation status in console
    """
    kinds = ['section', 'subsection', 'subsubsection', 'simple']
    # assert kind in kinds, "No such kind of status printing"
    # assert type(message) == type('a'), "Message is not string"

    l = len(message)
    l += 4
    if kind == kinds[0]:
        print()
        print(''.join(['#'] * l))
        print('# ' + message + ' #')
        print(''.join(['#'] * l))
        print()
    elif kind == kinds[1]:
        print()
        print(''.join(['.'] * l))
        print('. ' + message + ' .')
        print(''.join(['.'] * l))
        print()
    elif kind == kinds[2]:
        print()
        print('### ' + message + ' ###')
        print()
    elif kind == kinds[3]:
        print(message)
    else:
        print(message)


@contextmanager
def suppress_stdout():
    """ Suppress the output of some function """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def time_counter(func):
    def wrap(*args, **kwargs):
        start_time = perf_counter()
        res = func(*args, **kwargs)
        ex_time = perf_counter() - start_time
        print(f'Function was executed in {ex_time:.4f} sec = {ex_time / 60:.4f} mins = {ex_time / 3600:.4f} hours')
        return res

    return wrap
