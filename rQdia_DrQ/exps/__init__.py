import sys
from pathlib import Path

from .executor import Executor

# from .logger import *
DATADIR = Path(__file__).parent.parent / 'data'
this = sys.modules[__name__]
