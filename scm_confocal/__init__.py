__version__ = '1.1'

from .sp8 import sp8_series
from .visitech import visitech_series,visitech_faststack
from .util import util

#make available when using 'from scm_confocal import *'
__all__ = [
    'sp8_series',
    'visitech_series',
    'visitech_faststack',
    'util'
]
