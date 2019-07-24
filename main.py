#!/usr/bin/env python3


import logging
import signal
import sys
import time
import random
import numpy as np
import math
from numba import jit

# custom modules
import modules.constants as constants
import modules.filehandler as filehandler
from modules.fitnessfunction import Fitnessfunction
from modules.oa_neldermead import NelderMead


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s\t%(levelname)s\t[%(name)s: %(funcName)s]\t%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(constants.RUN_ARCHIVE_DIR + "log"), 
                    logging.StreamHandler()])

# setting of global minimum logging level
logging.disable(logging.NOTSET)

# set up logger for this module
LOGGER = logging.getLogger('main'); LOGGER.setLevel(logging.DEBUG)
FITNESSFUNCTION = Fitnessfunction()

def main():
    test=NelderMead(fitness, 5)
    test.run()
    pass


def fitness(data):
    a, b, c = FITNESSFUNCTION.run_normalized(data)
    return -a



# MAIN EXECUTION
# INFO:     run script as main, attach signal handling
# ARGS:     /
# RETURNS:  /
if __name__ == "__main__":
    
    main()





























