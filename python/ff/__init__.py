"""
Decorators for feature function definitions.
To create a new feature function simply:
    1) import ff
    2) use @configure in order to load pre-trained models and parameters
    3) use @feature to define a new feature which returns a single value 
    4) use @features('f1', 'f2', ..., 'fn') to define a function which returns a list of n feature values
    the arguments of the decoration are the feature names

@author waziz
"""
import logging
import sys
import os
import itertools

_FEATURES_ = []
_CONFIGURE_ = []

def configure(func):
    _CONFIGURE_.append(func)
    return func

def feature(func):
    _FEATURES_.append((func, [func.__name__]))
    logging.info('Function/feature %s', func.__name__)
    return func

class features(object):

    def __init__(self, *args):
        self.fnames_ = list(args)

    def __call__(self, func):
        if not self.fnames_:
            self.fnames_ = [func.__name__]
        logging.info('Function %s features %s', func.__name__, str(self.fnames_))
        _FEATURES_.append((func, self.fnames_))
        return func

def load_features(features):
    for featdef in features:
        logging.info('Loading additional feature definitions from %s', featdef)
        prefix = os.path.dirname(featdef)
        sys.path.append(prefix)
        __import__(os.path.basename(featdef).replace('.py', ''))
        sys.path.remove(prefix)

def configure_features(config):
    [func(config) for func in _CONFIGURE_]

def compute_features(hypothesis):
    pairs = []
    for func, fnames in _FEATURES_:
        result = func(hypothesis)
        if len(fnames) == 1: # function is returning a number
            pairs.append((fnames[0], result))
        else: # function is returning a list/tuple
            assert len(fnames) == len(result), 'more or less features than expected'
            pairs.extend(itertools.izip(fnames, result))
    return pairs
