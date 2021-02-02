__version__ = '2.0.0'

<<<<<<< Updated upstream
from .sp8 import sp8_lif,sp8_series
=======
from .sp8 import sp8_series,sp8_lif
>>>>>>> Stashed changes
from .visitech import visitech_series,visitech_faststack
from .util import util

#make available when using 'from scm_confocal import *'
__all__ = [
    'sp8_lif',
    'sp8_series',
    'sp8_lif',
    'visitech_series',
    'visitech_faststack',
    'util'
]
