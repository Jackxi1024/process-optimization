import numpy as np
import logging

from process.lib_basics import Basics
from process.class_stream import Stream
from process.lib_solvers import Solvers
from process.lib_process import Reactor, HeatExchanger, Decanter, DistillationColumn, Splitter

DEBUG = True
T_GUESS = 350

class Flowsheet():

    def __init__(self, FEED_A, FEED_B, GUESSES, PURGERATIO, HEATEX_T):
        # set-up for logging of flowsheet. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'Flowsheet'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Flowsheet instance is initialised")

        self._FEED_A = Stream()
        self._FEED_A.F = FEED_A.F
        self._FEED_A.T = T_GUESS
        self._FEED_B = Stream()
        self._FEED_B.F = FEED_B.F
        self._FEED_B.T = T_GUESS
        self._RECYCLE = Stream()
        self._RECYCLE.F = GUESSES['RECYCLE']
        self._RECYCLE.T = T_GUESS
        self._REACOUT = Stream()
        self._REACOUT.F = GUESSES['REACOUT']
        self._REACOUT.T = T_GUESS
        self._HEATEXOUT = Stream()
        self._HEATEXOUT.F = GUESSES['HEATEXOUT']
        self._DECANTEROUT = Stream()
        self._DECANTEROUT.F = GUESSES['DECANTEROUT']
        self._WASTE = Stream()
        self._WASTE.F = GUESSES['WASTE']
        self._WASTE.T = T_GUESS
        self._BOTTOMS = Stream()
        self._BOTTOMS.F = GUESSES['BOTTOMS']
        self._PRODUCT = Stream()
        self._PRODUCT.F = GUESSES['PRODUCT']
        self._PURGE = Stream()
        self._PURGE.F = GUESSES['PURGE']
        self._PURGE.T = T_GUESS

        self._PURGERATIO = PURGERATIO
        self._HEATEX_T = HEATEX_T

        self._MW = np.array([1, 1, 2, 2, 1, 3])
        self._REACOUT_W = self._REACOUT.F/sum(self._REACOUT.F)



    def print_Solution(self):

        print("\n\n")
        
        print("#################################\nSOLUTION (streams in lb/hr)\n#################################\n")
        print(Basics().np_to_pd([
            np.concatenate((self._FEED_A.F, np.array([self._FEED_A.T]))),
            np.concatenate((self._FEED_B.F, np.array([self._FEED_B.T]))),
            np.concatenate((self._REACOUT.F, np.array([self._REACOUT.T]))),
            np.concatenate((self._HEATEXOUT.F, np.array([self._HEATEXOUT.T]))),
            np.concatenate((self._DECANTEROUT.F, np.array([self._DECANTEROUT.T]))),
            np.concatenate((self._WASTE.F, np.array([self._WASTE.T]))),
            np.concatenate((self._PRODUCT.F, np.array([self._PRODUCT.T]))),
            np.concatenate((self._BOTTOMS.F, np.array([self._BOTTOMS.T]))),
            np.concatenate((self._RECYCLE.F, np.array([self._RECYCLE.T]))),
            np.concatenate((self._PURGE.F, np.array([self._PURGE.T])))
        ]))

        print("")

        print("PURGE RATIO: %1.4f" % self._PURGERATIO)

        print("\n\n")

        print("TEMPERATURE AFTER REACTOR: %1.1f" % self._REACOUT.T)

        print("\n\n")

        iterations = Basics().getPerformanceAssessment()

        print("#################################\nSTATISTICS\n#################################\n")
        for category in iterations:
            if iterations[category] is not None: print("ITERATIONS %s:\t%1.0f" % (category, iterations[category]))
        print("\n")



    def SEQ_Setup(self, tearstream = 'Recycle'):

        self._TEARSTREAM = Stream()
        self._TEARSTREAM_ID = tearstream

        if self._TEARSTREAM_ID is 'Recycle':
            self._TEARSTREAM.F = self._RECYCLE.F
            self._TEARSTREAM.T = self._RECYCLE.T
        else:
            self.logger.error("Unknown tear stream "+str(self._TEARSTREAM_ID)+".")

        self.initialised = True
        


    def SEQ_Start(self):

        if not self.initialised: 
            self.logger.error("Flowsheet SEQ was not set up! Call SEQ_Setup.")
            return None
        
        solution, stats = Solvers().solve(system = self.SEQ_System, guess = self._TEARSTREAM.F, category = 'TearStream')
        self.logger.info("Converged tear stream in "+str(stats["Iterations"])+" iterations.")
        Basics().TrackPerformance(stats, 'TearStream')

        self._TEARSTREAM.F = solution


        if DEBUG: 
            Basics().TrackSuccess()
            self.print_Solution()

        return Basics().np_to_pd([
            np.concatenate((self._FEED_A.F, np.array([self._FEED_A.T]))),
            np.concatenate((self._FEED_B.F, np.array([self._FEED_B.T]))),
            np.concatenate((self._REACOUT.F, np.array([self._REACOUT.T]))),
            np.concatenate((self._HEATEXOUT.F, np.array([self._HEATEXOUT.T]))),
            np.concatenate((self._DECANTEROUT.F, np.array([self._DECANTEROUT.T]))),
            np.concatenate((self._WASTE.F, np.array([self._WASTE.T]))),
            np.concatenate((self._PRODUCT.F, np.array([self._PRODUCT.T]))),
            np.concatenate((self._BOTTOMS.F, np.array([self._BOTTOMS.T]))),
            np.concatenate((self._RECYCLE.F, np.array([self._RECYCLE.T]))),
            np.concatenate((self._PURGE.F, np.array([self._PURGE.T])))
        ])


    def SEQ_System(self, DATA):

        self._TEARSTREAM.F = DATA

        if self._TEARSTREAM_ID is 'Recycle':
            self._RECYCLE.F = self._TEARSTREAM.F
            start_unit = 'Reactor'
        else:
            self.logger.error("Unknown tear stream "+str(self._TEARSTREAM_ID)+".")

        next_unit = start_unit

        while True:

            if next_unit is 'Reactor':
                self._REACOUT = Reactor(self._FEED_A, self._FEED_B, self._RECYCLE).SEQ(self._REACOUT)
                next_unit = 'HeatExchanger'
                if next_unit is start_unit:
                    return self._REACOUT.F

            elif next_unit is 'HeatExchanger':
                self._HEATEXOUT = HeatExchanger(self._REACOUT).SEQ(self._HEATEX_T)
                next_unit = 'Decanter'
                if next_unit is start_unit:
                    return self._HEATEXOUT.F

            elif next_unit is 'Decanter':
                self._DECANTEROUT, self._WASTE = Decanter(self._HEATEXOUT).SEQ(self._DECANTEROUT, self._WASTE)
                next_unit = 'DistillationColumn'
                if next_unit is start_unit:
                    return self._DECANTEROUT.F

            elif next_unit is 'DistillationColumn':
                self._PRODUCT, self._BOTTOMS = DistillationColumn(self._DECANTEROUT).SEQ()
                next_unit = 'Splitter'
                if next_unit is start_unit:
                    return self._BOTTOMS.F

            elif next_unit is 'Splitter':
                self._RECYCLE, self._PURGE = Splitter(self._BOTTOMS, self._PURGERATIO).SEQ()
                next_unit = 'Reactor'
                if next_unit is start_unit:
                    return self._RECYCLE.F

            else:
                self.logger.error("Unknown unit "+str(next_unit)+".")
                return None
