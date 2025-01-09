"""
funcoin

A python package for doing Functiontal Connectivity Integrative Normative Modelling.
"""

from .funcoin import Funcoin

from pkg_resources import DistributionNotFound, get_distribution

__author__ = 'Janus R. L. Kobbersmed'

# Setup the version
try:
    __version__ = get_distribution("funcoin").version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
