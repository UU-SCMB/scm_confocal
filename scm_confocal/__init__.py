__version__ = '2.0.0'

from .sp8 import sp8_lif,sp8_image,sp8_series
from .visitech import visitech_series,visitech_faststack
from .utility import util

#make available when using 'from scm_confocal import *'
__all__ = [
    'sp8_lif',
    'sp8_image',
    'sp8_series',
    'visitech_series',
    'visitech_faststack',
    'util'
]

#add submodules to pdoc ignore list
__pdoc__ = {
    'sp8' : False,
    'visitech' : False,
    'utility' : False,
}
