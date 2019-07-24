import os
import stat
import shutil
import logging
from numba import jit

# custom modules
import modules.constants


# set up logger for this module
LOGGER = logging.getLogger('filehandler'); LOGGER.setLevel(logging.INFO)


