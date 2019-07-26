import logging
import signal
import sys
import time
import random
import numpy as np
import math
from numba import jit
import os
import stat
import shutil
import atexit

# custom modules
import modules.constants as constants
import modules.filehandler as filehandler
from modules.fitnessfunction import Fitnessfunction



class Fitnessfunction_Wrapper():

    textbuffer = "iter,eductF_n,fracA_n,eductT_n,heatexT_n,purgeR_n,eductF,fracA,eductT,heatexT,purgeR,capex,opex,profit,energy_used,roi,fitness_function\n"
    textbuffer_maximum = 512*8
    datafile_path = constants.RUN_ARCHIVE_DIR+"fitnessfunction.data"
    iteration = 0

    FITNESSFUNCTION = Fitnessfunction()
    _evaluator = None

    def __init__(self):

        # set-up of general logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s\t%(levelname)s\t[%(name)s: %(funcName)s]\t%(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.StreamHandler()])

        # setting of global minimum logging level
        logging.disable(logging.DEBUG)

        # set-up for logging of main. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'wrapper'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self._evaluator = self._evaluator_capexVSroi

        atexit.register(self.cleanup)



    def cleanup(self):

        self.writeToFile(force = True)


    def evaluate(self, paramset):

        self.iteration += 1

        a, b, c, d, e = self.FITNESSFUNCTION.run_normalized(paramset)
        paramset_nonnormalized = self.FITNESSFUNCTION.unnormalized_parameters(paramset)

        fitness = self._evaluator(a, b, c, d, e)

        results = (a, b, c, d, e, fitness)
        
        self.textbuffer += (("%1.0f," % self.iteration) + ','.join(['%1.8f']*(len(paramset)+len(paramset_nonnormalized)+len(results))) % (tuple(paramset) + tuple(paramset_nonnormalized) + results)) + "\n"
        self.writeToFile()

        return fitness


    def _evaluator_roi(self, capex, opex, profit, energy_used, roi):
        return roi

    def _evaluator_profit(self, capex, opex, profit, energy_used, roi):
        return profit

    def _evaluator_capexVSroi(self, capex, opex, profit, energy_used, roi):
        characteristic_iterations = 5000
        preoptimization_iterations = 0 # was 500
        limits = [0.0, 0.75]

        iters = max(self.iteration-preoptimization_iterations, 0)
        w1 = 0.5+0.5*math.cos(math.pi*iters/characteristic_iterations)
        w2 = 1 - w1
        print(w2)
        k2 = limits[0] + w2*(limits[1]-limits[0]); k1 = 1 - k2
        return k1*roi + k2*(-capex)

    def _evaluator_capexVSroi_fix(self, capex, opex, profit, energy_used, roi):
        k1 = 0.4
        k2 = 1-k1
        return k1*roi + k2*(-capex)


    def writeToFile(self, force = False):

        if len(self.textbuffer) > self.textbuffer_maximum or force is True:
            with open(self.datafile_path, "a+") as datafile:
                datafile.write(str(self.textbuffer))
                datafile.close()
            self.textbuffer = ""
        else:
            pass
