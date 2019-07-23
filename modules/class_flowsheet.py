import numpy as np
import logging

from modules.lib_basics import Basics
from modules.class_stream import Stream
from modules.lib_solvers import Solvers
from modules.lib_process import Reactor, HeatExchanger, Decanter, DistillationColumn, Splitter

DEBUG = True
T_GUESS = 350

class Flowsheet():

    def __init__(self, FEED_A, FEED_B, GUESSES, PURGERATIO = None):
        # set-up for logging of flowsheet. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'Flowsheet'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Flowsheet instance is initialised")

        self._FEED_A = Stream()
        self._FEED_A.F = FEED_A.F
        self._FEED_A.T = 350
        self._FEED_B = Stream()
        self._FEED_B.F = FEED_B.F
        self._FEED_B.T = 350
        self._RECYCLE = Stream()
        self._RECYCLE.F = GUESSES['RECYCLE']
        self._RECYCLE.T = 350
        self._REACOUT = Stream()
        self._REACOUT.F = GUESSES['REACOUT']
        self._REACOUT.T = 350
        self._HEATEXOUT = Stream()
        self._HEATEXOUT.F = GUESSES['HEATEXOUT']
        self._HEATEXOUT.T = 350
        self._DECANTEROUT = Stream()
        self._DECANTEROUT.F = GUESSES['DECANTEROUT']
        self._DECANTEROUT.T = 350
        self._WASTE = Stream()
        self._WASTE.F = GUESSES['WASTE']
        self._WASTE.T = 350
        self._BOTTOMS = Stream()
        self._BOTTOMS.F = GUESSES['BOTTOMS']
        self._BOTTOMS.T = 350
        self._PRODUCT = Stream()
        self._PRODUCT.F = GUESSES['PRODUCT']
        self._PRODUCT.T = 350
        self._PURGE = Stream()
        self._PURGE.F = GUESSES['PURGE']
        self._PURGE.T = 350

        if PURGERATIO is None:
            self._PURGERATIO = GUESSES['PURGERATIO']
        else: 
            self._PURGERATIO = PURGERATIO

        self._MW = np.array([1, 1, 2, 2, 1, 3])
        self._REACOUT_W = self._REACOUT.F/sum(self._REACOUT.F)

        if DEBUG: self.print_Guess()


    def print_Solution(self):

        print("\n\n")
        
        print("#################################\nSOLUTION (streams in lb/hr)\n#################################\n")
        print(Basics().np_to_pd([
            self._FEED_A.F,
            self._FEED_B.F,
            self._REACOUT.F,
            self._HEATEXOUT.F,
            self._DECANTEROUT.F,
            self._WASTE.F,
            self._PRODUCT.F,
            self._BOTTOMS.F,
            self._RECYCLE.F,
            self._PURGE.F
        ]))

        print("")

        print("PURGE RATIO: %1.4f" % self._PURGERATIO)

        print("\n\n")

        iterations = Basics().getPerformanceAssessment()

        print("#################################\nSTATISTICS\n#################################\n")
        for category in iterations:
            if iterations[category] is not None: print("ITERATIONS %s:\t%1.0f" % (category, iterations[category]))
        print("\n")



    def print_Guess(self):

        print("\n\n")
        
        print("#################################\nGUESSES (streams in lb/hr)\n#################################\n")
        print(Basics().np_to_pd([
            self._FEED_A.F,
            self._FEED_B.F,
            self._REACOUT.F,
            self._HEATEXOUT.F,
            self._DECANTEROUT.F,
            self._WASTE.F,
            self._PRODUCT.F,
            self._BOTTOMS.F,
            self._RECYCLE.F,
            self._PURGE.F
        ]))

        print("")

        print("PURGE RATIO: %1.4f" % self._PURGERATIO)

        print("\n\n")


    def Start_Convergence(self, title = ""):
        print("################################# BEGIN CONVERGENCE "+str(title)+" #################################")


    def End_Convergence(self, title = ""):
        print("################################# END CONVERGENCE "+str(title)+" #################################\n")



    def SEQ_DesignSpec(self, designspec = 1500):

        guess = np.array([self._PURGERATIO])
        self._designspec_target = designspec

        if DEBUG: self.Start_Convergence(title = "DesignSpec")
        solution, stats = Solvers().solve(system = self.SEQ_DesignSpec_System, guess = guess, category = 'DesignSpec')
        Basics().TrackPerformance(stats, 'DesignSpec')
        self.logger.info("Converged design specification in "+str(stats["Iterations"])+" iterations.")
        if DEBUG: self.End_Convergence(title = "DesignSpec")

        self._PURGERATIO = solution[0]

        Basics().TrackSuccess()
        if DEBUG: self.print_Solution()



    def SEQ_DesignSpec_System(self, DATA):

        self._PURGERATIO = DATA[0]

        # after this call, all stream data in this class will be available
        self.SEQ_Start(print = False)

        production_rate = self._PRODUCT.F[4]

        return np.array([
            self._designspec_target - production_rate
        ])



    def SEQ_Setup(self, tearstream = 'Recycle'):

        self._TEARSTREAM = Stream()
        self._TEARSTREAM_ID = tearstream

        if self._TEARSTREAM_ID is 'Recycle':
            self._TEARSTREAM.F = self._RECYCLE.F
        elif self._TEARSTREAM_ID is 'ReactorOut':
            self._TEARSTREAM.F = self._REACOUT.F
        elif self._TEARSTREAM_ID is 'HeatExOut':
            self._TEARSTREAM.F = self._HEATEXOUT.F
        elif self._TEARSTREAM_ID is 'DecanterOut':
            self._TEARSTREAM.F = self._DECANTEROUT.F
        elif self._TEARSTREAM_ID is 'Bottoms':
            self._TEARSTREAM.F = self._BOTTOMS.F
        else:
            self.logger.error("Unknown tear stream "+str(self._TEARSTREAM_ID)+".")

        self.initialised = True
        


    def SEQ_Start(self, print = True):

        if not self.initialised: 
            self.logger.error("Flowsheet SEQ was not set up! Call SEQ_Setup.")
            return None

        if DEBUG: self.Start_Convergence(title = "TearStream")
        solution, stats = Solvers().solve(system = self.SEQ_System, guess = self._TEARSTREAM.F, category = 'TearStream')
        self.logger.info("Converged tear stream in "+str(stats["Iterations"])+" iterations.")
        Basics().TrackPerformance(stats, 'TearStream')
        if DEBUG: self.End_Convergence(title = "TearStream")

        self._TEARSTREAM.F = solution

        if print and DEBUG: 
            Basics().TrackSuccess()
            self.print_Solution()


    def SEQ_System(self, DATA):

        self._TEARSTREAM.F = DATA

        if self._TEARSTREAM_ID is 'Recycle':
            self._RECYCLE.F = self._TEARSTREAM.F
            start_unit = 'Reactor'
        elif self._TEARSTREAM_ID is 'ReactorOut':
            self._REACOUT.F = self._TEARSTREAM.F
            start_unit = 'HeatExchanger'
        elif self._TEARSTREAM_ID is 'HeatExOut':
            self._HEATEXOUT.F = self._TEARSTREAM.F
            start_unit = 'Decanter'
        elif self._TEARSTREAM_ID is 'DecanterOut':
            self._DECANTEROUT.F = self._TEARSTREAM.F
            start_unit = 'DistillationColumn'
        elif self._TEARSTREAM_ID is 'Bottoms':
            self._BOTTOMS.F = self._TEARSTREAM.F
            start_unit = 'Splitter'
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
                self._HEATEXOUT = HeatExchanger(self._REACOUT).SEQ()
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



    def EO_Setup(self):

        self._EO_guess = np.concatenate((
            self._REACOUT.F,
            self._REACOUT_W,
            self._HEATEXOUT.F,
            self._DECANTEROUT.F,
            self._WASTE.F,
            self._PRODUCT.F,
            self._BOTTOMS.F,
            self._RECYCLE.F,
            self._PURGE.F
        ))

        self.initialised = True



    def EO_Start(self):

        if not self.initialised: 
            self.logger.error("Flowsheet EO was not set up! Call EO_Setup.")
            return None

        if DEBUG: self.Start_Convergence(title = "EO")
        solution, stats = Solvers().solve(system = self.EO_System, jacobian = self. EO_Jacobian, guess = self._EO_guess, category = 'EO')
        self.logger.info("Converged EO system in "+str(stats["Iterations"])+" iterations.")
        Basics().TrackPerformance(stats, 'EO')
        if DEBUG: self.End_Convergence(title = "EO")

        self._REACOUT.F = solution[0:6]
        self._HEATEXOUT.F = solution[12:18]
        self._DECANTEROUT.F = solution[18:24]
        self._WASTE.F = solution[24:30]
        self._PRODUCT.F = solution[30:36]
        self._BOTTOMS.F = solution[36:42]
        self._RECYCLE.F = solution[42:48]
        self._PURGE.F = solution[48:54]

        Basics().mass_balance_check([self._FEED_A, self._FEED_B, self._RECYCLE], [self._REACOUT], by_species = False)
        Basics().mass_balance_check([self._REACOUT], [self._HEATEXOUT])
        Basics().mass_balance_check([self._HEATEXOUT], [self._DECANTEROUT, self._WASTE])
        Basics().mass_balance_check([self._DECANTEROUT], [self._PRODUCT, self._BOTTOMS])
        Basics().mass_balance_check([self._BOTTOMS], [self._PURGE, self._RECYCLE])

        Basics().TrackSuccess()
        if DEBUG: self.print_Solution()


    def EO_System(self, DATA):

        DATA_REACOUT = DATA[0:6]
        DATA_wREAC = DATA[6:12]
        DATA_HEATEXOUT = DATA[12:18]
        DATA_DECANTOUT = DATA[18:24]
        DATA_WASTE = DATA[24:30]
        DATA_HEAD = DATA[30:36]
        DATA_BOT = DATA[36:42]
        DATA_RECYCLE = DATA[42:48]
        DATA_PURGE = DATA[48:54]

        DATA_REACTOR = np.concatenate((DATA_REACOUT, DATA_wREAC))
        DATA_DECANTER = np.concatenate((DATA_DECANTOUT, DATA_WASTE))
        DATA_DIST = np.concatenate((DATA_HEAD, DATA_BOT))
        DATA_SPLITTER = np.concatenate((DATA_RECYCLE, DATA_PURGE))

        STREAM_REACOUT = Stream()
        STREAM_REACOUT.F = DATA_REACOUT

        STREAM_HEATEXOUT = Stream()
        STREAM_HEATEXOUT.F = DATA_HEATEXOUT

        STREAM_DECANTOUT = Stream()
        STREAM_DECANTOUT.F = DATA_DECANTOUT

        STREAM_BOT = Stream()
        STREAM_BOT.F = DATA_BOT

        STREAM_RECYCLE = Stream()
        STREAM_RECYCLE.F = DATA_RECYCLE

        System_Reactor = Reactor(self._FEED_A, self._FEED_B, STREAM_RECYCLE).System(DATA_REACTOR)
        System_HeatExchanger = HeatExchanger(STREAM_REACOUT).System(DATA_HEATEXOUT)
        System_Decanter = Decanter(STREAM_HEATEXOUT).System(DATA_DECANTER)
        System_Distillation = DistillationColumn(STREAM_DECANTOUT).System(DATA_DIST)
        System_Splitter = Splitter(STREAM_BOT, self._PURGERATIO).System(DATA_SPLITTER)



        return np.concatenate((
            System_Reactor,
            System_HeatExchanger,
            System_Decanter,
            System_Distillation,
            System_Splitter,
        ))




    def EO_Jacobian(self, DATA):

        DATA_REACOUT = DATA[0:6]
        DATA_wREAC = DATA[6:12]
        DATA_HEATEXOUT = DATA[12:18]
        DATA_DECANTOUT = DATA[18:24]
        DATA_WASTE = DATA[24:30]
        DATA_HEAD = DATA[30:36]
        DATA_BOT = DATA[36:42]
        DATA_RECYCLE = DATA[42:48]
        DATA_PURGE = DATA[48:54]

        DATA_REACTOR = np.concatenate((DATA_REACOUT, DATA_wREAC))
        DATA_DECANTER = np.concatenate((DATA_DECANTOUT, DATA_WASTE))
        DATA_DIST = np.concatenate((DATA_HEAD, DATA_BOT))
        DATA_SPLITTER = np.concatenate((DATA_RECYCLE, DATA_PURGE))

        STREAM_REACOUT = Stream()
        STREAM_REACOUT.F = DATA_REACOUT

        STREAM_HEATEXOUT = Stream()
        STREAM_HEATEXOUT.F = DATA_HEATEXOUT

        STREAM_DECANTOUT = Stream()
        STREAM_DECANTOUT.F = DATA_DECANTOUT

        STREAM_BOT = Stream()
        STREAM_BOT.F = DATA_BOT

        STREAM_RECYCLE = Stream()
        STREAM_RECYCLE.F = DATA_RECYCLE


        REACTOR = Reactor(self._FEED_A, self._FEED_B, STREAM_RECYCLE)
        Jacobian_Reactor = REACTOR.Jacobian(DATA_REACTOR)
        Jacobian_Reactor_Recycle = REACTOR.Jacobian_Feed()

        HEATEX = HeatExchanger(STREAM_REACOUT)
        Jacobian_HeatExchanger = HEATEX.Jacobian(DATA_HEATEXOUT)
        Jacobian_HeatExchanger_Feed = HEATEX.Jacobian_Feed()

        DECANTER = Decanter(STREAM_HEATEXOUT)
        Jacobian_Decanter = DECANTER.Jacobian(DATA_DECANTER)
        Jacobian_Decanter_Feed = DECANTER.Jacobian_Feed()

        DISTILLATION = DistillationColumn(STREAM_DECANTOUT)
        Jacobian_Distillation = DISTILLATION.Jacobian(DATA_DIST)
        Jacobian_Distillation_Feed = DISTILLATION.Jacobian_Feed()

        SPLITTER = Splitter(STREAM_BOT, self._PURGERATIO)
        Jacobian_Splitter = SPLITTER.Jacobian(DATA_SPLITTER)
        Jacobian_Splitter_Feed = SPLITTER.Jacobian_Feed()

        jacobian = np.zeros((54, 54))

        jacobian[0:12, 0:12] = Jacobian_Reactor
        jacobian[0:12, 42:48] = Jacobian_Reactor_Recycle

        jacobian[12:18, 12:18] = Jacobian_HeatExchanger
        jacobian[12:18, 0:6] = Jacobian_HeatExchanger_Feed

        jacobian[18:30, 18:30] = Jacobian_Decanter
        jacobian[18:30, 12:18] = Jacobian_Decanter_Feed

        jacobian[30:42, 30:42] = Jacobian_Distillation
        jacobian[30:42, 18:24] = Jacobian_Distillation_Feed

        jacobian[42:54, 42:54] = Jacobian_Splitter
        jacobian[42:54, 36:42] = Jacobian_Splitter_Feed

        return jacobian



    def EO_DesignSpec(self, designspec = 1500):

        if not self.initialised: 
            self.logger.error("Flowsheet EO was not set up! Call EO_Setup.")
            return None

        self._designspec_target = designspec

        self._EO_guess = np.concatenate((self._EO_guess, np.array([0.8])))

        if DEBUG: self.Start_Convergence(title = "EO + DesignSpec")
        solution, stats = Solvers().solve(system = self.EO_DesignSpec_System, jacobian = self.EO_DesignSpec_Jacobian, guess = self._EO_guess, category = 'EO')
        self.logger.info("Converged EO system with design specification in "+str(stats["Iterations"])+" iterations.")
        Basics().TrackPerformance(stats, 'EO')
        if DEBUG: self.End_Convergence(title = "EO + DesignSpec")

        self._REACOUT.F = solution[0:6]
        self._HEATEXOUT.F = solution[12:18]
        self._DECANTEROUT.F = solution[18:24]
        self._WASTE.F = solution[24:30]
        self._PRODUCT.F = solution[30:36]
        self._BOTTOMS.F = solution[36:42]
        self._RECYCLE.F = solution[42:48]
        self._PURGE.F = solution[48:54]
        self._PURGERATIO = solution[54]

        Basics().mass_balance_check([self._FEED_A, self._FEED_B, self._RECYCLE], [self._REACOUT], by_species = False)
        Basics().mass_balance_check([self._REACOUT], [self._HEATEXOUT])
        Basics().mass_balance_check([self._HEATEXOUT], [self._DECANTEROUT, self._WASTE])
        Basics().mass_balance_check([self._DECANTEROUT], [self._PRODUCT, self._BOTTOMS])
        Basics().mass_balance_check([self._BOTTOMS], [self._PURGE, self._RECYCLE])

        Basics().TrackSuccess()
        if DEBUG: self.print_Solution()



    def EO_DesignSpec_System(self, DATA):

        DATA_REACOUT = DATA[0:6]
        DATA_wREAC = DATA[6:12]
        DATA_HEATEXOUT = DATA[12:18]
        DATA_DECANTOUT = DATA[18:24]
        DATA_WASTE = DATA[24:30]
        DATA_HEAD = DATA[30:36]
        DATA_BOT = DATA[36:42]
        DATA_RECYCLE = DATA[42:48]
        DATA_PURGE = DATA[48:54]
        DATA_PURGERATIO = DATA[54]

        DATA_REACTOR = np.concatenate((DATA_REACOUT, DATA_wREAC))
        DATA_DECANTER = np.concatenate((DATA_DECANTOUT, DATA_WASTE))
        DATA_DIST = np.concatenate((DATA_HEAD, DATA_BOT))
        DATA_SPLITTER = np.concatenate((DATA_RECYCLE, DATA_PURGE))

        STREAM_REACOUT = Stream()
        STREAM_REACOUT.F = DATA_REACOUT

        STREAM_HEATEXOUT = Stream()
        STREAM_HEATEXOUT.F = DATA_HEATEXOUT

        STREAM_DECANTOUT = Stream()
        STREAM_DECANTOUT.F = DATA_DECANTOUT

        STREAM_BOT = Stream()
        STREAM_BOT.F = DATA_BOT

        STREAM_RECYCLE = Stream()
        STREAM_RECYCLE.F = DATA_RECYCLE

        System_Reactor = Reactor(self._FEED_A, self._FEED_B, STREAM_RECYCLE).System(DATA_REACTOR)
        System_HeatExchanger = HeatExchanger(STREAM_REACOUT).System(DATA_HEATEXOUT)
        System_Decanter = Decanter(STREAM_HEATEXOUT).System(DATA_DECANTER)
        System_Distillation = DistillationColumn(STREAM_DECANTOUT).System(DATA_DIST)
        System_Splitter = Splitter(STREAM_BOT, DATA_PURGERATIO).System(DATA_SPLITTER)


        PRODUCTFLOW = DATA[38]
        DESIGNSPEC = self._designspec_target


        return np.concatenate((
            System_Reactor,
            System_HeatExchanger,
            System_Decanter,
            System_Distillation,
            System_Splitter,
            np.array([PRODUCTFLOW - DESIGNSPEC])
        ))




    def EO_DesignSpec_Jacobian(self, DATA):

        DATA_REACOUT = DATA[0:6]
        DATA_wREAC = DATA[6:12]
        DATA_HEATEXOUT = DATA[12:18]
        DATA_DECANTOUT = DATA[18:24]
        DATA_WASTE = DATA[24:30]
        DATA_HEAD = DATA[30:36]
        DATA_BOT = DATA[36:42]
        DATA_RECYCLE = DATA[42:48]
        DATA_PURGE = DATA[48:54]
        DATA_PURGERATIO = DATA[54]

        DATA_REACTOR = np.concatenate((DATA_REACOUT, DATA_wREAC))
        DATA_DECANTER = np.concatenate((DATA_DECANTOUT, DATA_WASTE))
        DATA_DIST = np.concatenate((DATA_HEAD, DATA_BOT))
        DATA_SPLITTER = np.concatenate((DATA_RECYCLE, DATA_PURGE))

        STREAM_REACOUT = Stream()
        STREAM_REACOUT.F = DATA_REACOUT

        STREAM_HEATEXOUT = Stream()
        STREAM_HEATEXOUT.F = DATA_HEATEXOUT

        STREAM_DECANTOUT = Stream()
        STREAM_DECANTOUT.F = DATA_DECANTOUT

        STREAM_BOT = Stream()
        STREAM_BOT.F = DATA_BOT

        STREAM_RECYCLE = Stream()
        STREAM_RECYCLE.F = DATA_RECYCLE


        REACTOR = Reactor(self._FEED_A, self._FEED_B, STREAM_RECYCLE)
        Jacobian_Reactor = REACTOR.Jacobian(DATA_REACTOR)
        Jacobian_Reactor_Recycle = REACTOR.Jacobian_Feed()

        HEATEX = HeatExchanger(STREAM_REACOUT)
        Jacobian_HeatExchanger = HEATEX.Jacobian(DATA_HEATEXOUT)
        Jacobian_HeatExchanger_Feed = HEATEX.Jacobian_Feed()

        DECANTER = Decanter(STREAM_HEATEXOUT)
        Jacobian_Decanter = DECANTER.Jacobian(DATA_DECANTER)
        Jacobian_Decanter_Feed = DECANTER.Jacobian_Feed()

        DISTILLATION = DistillationColumn(STREAM_DECANTOUT)
        Jacobian_Distillation = DISTILLATION.Jacobian(DATA_DIST)
        Jacobian_Distillation_Feed = DISTILLATION.Jacobian_Feed()

        SPLITTER = Splitter(STREAM_BOT, DATA_PURGERATIO)
        Jacobian_Splitter = SPLITTER.Jacobian(DATA_SPLITTER)
        Jacobian_Splitter_Feed = SPLITTER.Jacobian_Feed()
        Jacobian_Splitter_Ratio = SPLITTER.Jacobian_Ratio()

        jacobian = np.zeros((55, 55))

        jacobian[0:12, 0:12] = Jacobian_Reactor
        jacobian[0:12, 42:48] = Jacobian_Reactor_Recycle

        jacobian[12:18, 12:18] = Jacobian_HeatExchanger
        jacobian[12:18, 0:6] = Jacobian_HeatExchanger_Feed

        jacobian[18:30, 18:30] = Jacobian_Decanter
        jacobian[18:30, 12:18] = Jacobian_Decanter_Feed

        jacobian[30:42, 30:42] = Jacobian_Distillation
        jacobian[30:42, 18:24] = Jacobian_Distillation_Feed

        jacobian[42:54, 42:54] = Jacobian_Splitter
        jacobian[42:54, 36:42] = Jacobian_Splitter_Feed

        jacobian[54, 38] = 1
        jacobian[42:54, 54] = Jacobian_Splitter_Ratio

        return jacobian