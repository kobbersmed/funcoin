"""
funcoin

A python package for doing Functiontal Connectivity Integrative Normative Modelling.
"""

from .funcoin import Funcoin

__author__ = 'Janus R. L. Kobbersmed'

# Setup the version
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("fcin")
except PackageNotFoundError:
    __version__ = "unknown"