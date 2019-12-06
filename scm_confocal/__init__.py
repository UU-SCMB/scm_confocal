__version__ = '1.0'

from .confocal import sp8_series,visitech_series,visitech_faststack,util

#make available when using 'from scm_confocal import *'
__all__ = [
    'sp8_series',
    'visitech_series',
    'visitech_faststack',
    'util'
]
