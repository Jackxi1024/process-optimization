#!/usr/bin/env python3

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from modules.lib_basics import Basics, exceptionreport
from modules.lib_solvers import Solvers
from modules.lib_process import Reactor, HeatExchanger, Decanter, DistillationColumn, Splitter
from modules.class_stream import Stream
from modules.class_flowsheet import Flowsheet



# 
# DEFINITION OF SPECIFICATIONS
# ORDER OF COMPONENTS: A B C E P G
# 

FEED_A = Stream()
FEED_A.F = np.array([6582, 0, 0, 0, 0, 0])

FEED_B = Stream()
FEED_B.F = np.array([0, 14996, 0, 0, 0, 0])

PRODUCT_DESIGNSPEC = 1500

PURGE_RATIO_BOUNDS = np.array([0.56, 0.9])
PURGE_RATIO_POINTS = 34



# 
# DEFINITION OF GUESSES.
# ORDER OF COMPONENTS: A B C E P G
# 

# recycle stream is guessed as mass flows in lb/hr
RECYCLE = np.array([0, 2000, 0, 2000, 0, 0])

PURGERATIO = 0.6

# reactor outlet is guessed as component mass fractions
REACOUT_W = np.array([0.02, 0.20, 0.01, 0.40, 0.10, 0.20])

# other unit streams are guessed as individual recovery on mass basis based on feed stream of that unit
HEATEXOUT_R = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
DECANTEROUT_R = np.array([1.00, 1.00, 1.00, 1.00, 0.95, 0.35])
WASTE_R = np.array([0.00, 0.00, 0.00, 0.00, 0.05, 0.65])
BOTTOMS_R = np.array([0.99, 0.99, 0.99, 0.99, 0.5, 0.99])
PRODUCT_R = np.array([0.01, 0.01, 0.01, 0.01, 0.5, 0.01])
PURGE_R = np.array([PURGERATIO, PURGERATIO, PURGERATIO, PURGERATIO, PURGERATIO, 1])



# 
# DEFINITION OF TEAR STREAM.
# AVAILABLE TEAR STREAMS: Recycle, ReactorOut, HeatExOut, DecanterOut, Bottoms
# 

# selected tear stream. Only Recycle was tested with the predefined guesses.
TEAR_STREAM = "Recycle"





class Main():

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
        self.logtitle = 'main'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Main instance is initialised")


        # total mass flow into the reactor unit
        TOTALMASSFLOW = sum(np.concatenate((FEED_A.F, FEED_B.F, RECYCLE)))

        # calculate mass flows of other units based on the guessed component recoveries
        REACOUT = REACOUT_W*TOTALMASSFLOW
        HEATEXOUT = HEATEXOUT_R*REACOUT
        DECANTEROUT = DECANTEROUT_R*HEATEXOUT
        WASTE = WASTE_R*HEATEXOUT
        BOTTOMS = BOTTOMS_R*DECANTEROUT
        PRODUCT = PRODUCT_R*DECANTEROUT
        PURGE = PURGE_R*BOTTOMS

        # compile all guesses in order to pass them to flowsheet
        self.GUESSES = {}
        self.GUESSES['RECYCLE'] = RECYCLE
        self.GUESSES['REACOUT'] = REACOUT
        self.GUESSES['HEATEXOUT'] = HEATEXOUT
        self.GUESSES['DECANTEROUT'] = DECANTEROUT
        self.GUESSES['WASTE'] = WASTE
        self.GUESSES['BOTTOMS'] = BOTTOMS
        self.GUESSES['PRODUCT'] = PRODUCT
        self.GUESSES['PURGE'] = PURGE
        self.GUESSES['PURGERATIO'] = PURGERATIO

        # manual specifications
        self.PURGERATIO = None
        self.DESIGNSPEC = None



    def Start(self):

        print("\n#################################\nAvailable operations\n#################################\n")
        print("1:\tPerform task 1 of exercise sheet")
        print("2:\tPerform task 2 of exercise sheet")
        print("3:\tPerform task 3 of exercise sheet")
        print("SEQ:\tPerform sequential simulation with specified purge ratio")
        print("EO:\tPerform equation-oriented simulation with specified purge ratio")
        print("SEQ-DS:\tPerform sequential design with specified product flow")
        print("EO-DS:\tPerform equation-oriented design with specified product flow")

        command = None

        while True:
            print("\n\nSelect program:")
            command = input()

            if command == "1":
                self.DESIGNSPEC = PRODUCT_DESIGNSPEC
                self.SEQ_DS()
                break
            
            elif command == "2":
                self.DESIGNSPEC = PRODUCT_DESIGNSPEC
                self.EO_DS()
                break
       
            elif command == "3":
                self.PerformanceAssessment()
                break
                        
            elif command == "SEQ":
                self.Query_PurgeRatio()
                self.SEQ()
                break
                        
            elif command == "EO":
                self.Query_PurgeRatio()
                self.EO()
                break
                        
            elif command == "SEQ-DS":
                self.Query_DesignSpec()
                self.SEQ_DS()
                break
                        
            elif command == "EO-DS":
                self.Query_DesignSpec()
                self.EO_DS()
                break

            else:
                print("Unknown command.")
                continue


    def Query_PurgeRatio(self):

        command = None

        while True:
            print("\n\nSpecify purge ratio:")

            try:
                command = float(input())
            except:
                print("Please input a number.")
                continue

            if command >= 0 and command <= 1:
                self.PURGERATIO = command
                break
            
            else:
                print("Please input a number between 0 and 1.")
                continue



    def Query_DesignSpec(self):

        command = None

        while True:
            print("\n\nSpecify product flow:")

            try:
                command = float(input())
            except:
                print("Please input a number.")
                continue

            if command >= 0 and command <= 1e5:
                self.DESIGNSPEC = command
                break
            
            else:
                print("Please input a number between 0 and 1e5.")
                continue



    @exceptionreport
    def SEQ(self):
        
        Process = Flowsheet(FEED_A, FEED_B, self.GUESSES, PURGERATIO = self.PURGERATIO)
        Process.SEQ_Setup(tearstream = TEAR_STREAM)
        Process.SEQ_Start()

        del Process


    @exceptionreport
    def EO(self):

        Process = Flowsheet(FEED_A, FEED_B, self.GUESSES, PURGERATIO = self.PURGERATIO)
        Process.EO_Setup()
        Process.EO_Start()

        del Process

    
    @exceptionreport
    def SEQ_DS(self):

        Process = Flowsheet(FEED_A, FEED_B, self.GUESSES)
        Process.SEQ_Setup(tearstream = TEAR_STREAM)
        Process.SEQ_DesignSpec(designspec = self.DESIGNSPEC)

        del Process


    @exceptionreport
    def EO_DS(self):

        Process = Flowsheet(FEED_A, FEED_B, self.GUESSES)
        Process.EO_Setup()
        Process.EO_DesignSpec(designspec = self.DESIGNSPEC)

        del Process
 
    @exceptionreport
    def PerformanceAssessment(self):

        purge_ratios = np.linspace(PURGE_RATIO_BOUNDS[0], PURGE_RATIO_BOUNDS[1], PURGE_RATIO_POINTS)
        results = {'SEQ': [], 'EO': []}

        for purge_ratio in purge_ratios:
            self.PURGERATIO = purge_ratio

            Basics().startPerformanceAssessment()
            self.SEQ()
            results['SEQ'].append(Basics().getPerformanceAssessment())

            Basics().startPerformanceAssessment()
            self.EO()
            results['EO'].append(Basics().getPerformanceAssessment())


        fig = plt.figure(figsize=(8, 6))

        seq_plot = fig.add_subplot(2, 1, 1)

        data = [[],[]]
        for item in results['SEQ']:
            data[0].append(item['Unit'])
            data[1].append(item['TearStream'])

        ind = np.arange(len(purge_ratios))

        p1 = plt.bar(ind, data[0])
        p2 = plt.bar(ind, data[1], bottom=data[0])
        
        plt.ylabel('Iterations')
        plt.xlabel('Purge Ratio')
        plt.title('SEQ Approach')
        plt.xticks(ind, ["%1.2f" % i for i in purge_ratios])
        plt.legend((p1[0], p2[0]), ('Unit', 'TearStream'))
        

        eo_plot = fig.add_subplot(2, 1, 2)

        data = []
        for item in results['EO']:
            data.append(item['EO'])

        ind = np.arange(len(purge_ratios))

        p1 = plt.bar(ind, data)
        
        plt.ylabel('Iterations')
        plt.xlabel('Purge Ratio')
        plt.title('EO Approach')
        plt.xticks(ind, ["%1.2f" % i for i in purge_ratios])

        plt.tight_layout()
        plt.show()
        fig.savefig('Task3.pdf')





if __name__ == "__main__":

    # invoke main class
    Program = Main()
    Program.Start()


# CONVERGED SOLUTION WITH DESIGN SPEC FOR REFERENCE:

    # REACOUT = np.array([353, 5349, 63, 11297, 2892, 5434])
    # HEATEXOUT = np.array([353, 5349, 63, 11297, 2892, 5434])
    # DECANTEROUT = np.array([353, 5349, 63, 11297, 2626, 1806])
    # WASTE = np.array([0, 0, 0, 0, 266, 3628])
    # BOTTOMS = np.array([353, 5346, 62.7, 11256, 1125, 1805])
    # PRODUCT = np.array([1, 2.8, 0.055, 40.6, 1500, 0.4])
    # PURGE = np.array([278, 4224, 49, 8893, 889, 1805])
    # RECYCLE = np.array([74, 1123, 13, 2364, 236, 0])
    # PURGERATIO = 0.7900