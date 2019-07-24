#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from process.lib_basics import Basics
from process.lib_solvers import Solvers
from process.lib_process import Reactor, HeatExchanger, Decanter, DistillationColumn, Splitter
from process.class_stream import Stream
from process.class_flowsheet import Flowsheet




class Fitnessfunction():

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


    def run(self, FEED, EDUCT_RATIO, EDUCT_T, HEATEX_T, PURGE_RATIO):

        # process feed specification
        FEED_A = Stream()
        FEED_A.F = np.array([FEED*EDUCT_RATIO, 0, 0, 0, 0, 0])
        FEED_A.T = EDUCT_T
        FEED_B = Stream()
        FEED_B.F = np.array([0, FEED*(1-EDUCT_RATIO), 0, 0, 0, 0])
        FEED_B.T = EDUCT_T

        # recycle stream is guessed as mass flows in lb/hr
        RECYCLE = np.array([0, 2000, 0, 2000, 0, 0])

        # reactor outlet is guessed as component mass fractions
        REACOUT_W = np.array([0.02, 0.20, 0.01, 0.40, 0.10, 0.20])

        # other unit streams are guessed as individual recovery on mass basis based on feed stream of that unit
        HEATEXOUT_R = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
        DECANTEROUT_R = np.array([1.00, 1.00, 1.00, 1.00, 0.95, 0.35])
        WASTE_R = np.array([0.00, 0.00, 0.00, 0.00, 0.05, 0.65])
        BOTTOMS_R = np.array([0.99, 0.99, 0.99, 0.99, 0.5, 0.99])
        PRODUCT_R = np.array([0.01, 0.01, 0.01, 0.01, 0.5, 0.01])
        PURGE_R = np.array([PURGE_RATIO, PURGE_RATIO, PURGE_RATIO, PURGE_RATIO, PURGE_RATIO, 1])

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
        GUESSES = {}
        GUESSES['RECYCLE'] = RECYCLE
        GUESSES['REACOUT'] = REACOUT
        GUESSES['HEATEXOUT'] = HEATEXOUT
        GUESSES['DECANTEROUT'] = DECANTEROUT
        GUESSES['WASTE'] = WASTE
        GUESSES['BOTTOMS'] = BOTTOMS
        GUESSES['PRODUCT'] = PRODUCT
        GUESSES['PURGE'] = PURGE
        GUESSES['PURGERATIO'] = PURGE_RATIO

        # selected tear stream. Only Recycle was tested with the predefined guesses.
        TEAR_STREAM = "Recycle"
        
        Process = Flowsheet(FEED_A, FEED_B, GUESSES, PURGE_RATIO, EDUCT_T, HEATEX_T)
        Process.SEQ_Setup(tearstream = TEAR_STREAM)
        data = Process.SEQ_Start()

        print(data)

        del Process

        CP_reactor = 2170 # in J/kgK
        CP_water = 4180 # in J/kgK

        species = ['A', 'B', 'C', 'E', 'P', 'G']

        capex = list()
        opex = list()
        energy_used = list()
        materials = list()
        
        # reactor
        flow = data.loc["REACOUT", species].sum()
        capex.append(60*flow)
        opex.append(10*flow + 0.002*(flow-30000)**2)
        energy_used.append(0)
        
        # heatex
        flow = data.loc["REACOUT", species].sum()
        DeltaT = data.loc["REACOUT", "T"] - data.loc["HEATEXOUT", "T"]
        energy = DeltaT*flow*CP_reactor
        energy_used.append(energy)
        capex.append(1e-6*energy)
        opex.append(1*energy/(20*CP_water))

        # decanter
        capex.append(1e5)
        opex.append(100*data.loc["WASTE", species].sum())
        energy_used.append(0.15*36000000*data.loc["WASTE", species].sum())

        # distillation column
        flow = data.loc["DECANTOUT", species].sum()
        DeltaT = data.loc["HEAD", "T"] - data.loc["BOT", "T"]
        energy = DeltaT*flow*CP_water
        capex.append(10*flow)
        opex.append(1e-5*energy+2.5*flow)
        energy_used.append(energy)

        # purge
        capex.append(0)
        opex.append(10*data.loc["PURGE", species].sum())
        energy_used.append(0.25*3600000*data.loc["PURGE", species].sum())

        # heatex
        flow = data.loc["RECYCLE", species].sum()
        DeltaT = data.loc["RECYCLE", "T"] - data.loc["FEED_A", "T"]
        energy = abs(DeltaT*flow*CP_reactor)
        energy_used.append(energy)
        capex.append(1e-6*energy)
        opex.append(1*energy/(20*CP_water))

        # materials
        materials.append(5000*0.007*data.loc["FEED_A", 'A'])
        materials.append(5000*0.005*data.loc["FEED_B", 'B'])
        materials.append(5000*-0.3*data.loc["HEAD", 'P']*(data.loc["HEAD", 'P']/data.loc["HEAD", species].sum())**2)
        capex.append(0); capex.append(0); capex.append(0);
        opex.append(materials[0]); opex.append(materials[1]); opex.append(materials[2]);
        energy_used.append(0); energy_used.append(0); energy_used.append(0);

        # overall
        capex.append(500000 + sum(capex))
        opex.append(sum(opex))
        energy_used.append(sum(energy_used))

        result = pd.DataFrame(np.transpose(np.array([capex, opex, energy_used])), index=["REAC", "HEATEX1", "DECANT", "DIST", "PURGE", "HEATEX2", "A", "B", "P", "SUM"], columns=["CAPEX", "OPEX", "ENERGY"])
        print(result)

        capex = sum(capex)
        opex = 500000 + sum(opex)
        energy_used = sum(energy_used)
        roi = -(sum(materials)+opex)/capex

        print(roi)

        return capex, roi, energy_used



if __name__ == "__main__":

    # invoke main class
    Program = Fitnessfunction()
    # Program.run(21500, 0.3, 550, 450, .2)
    # Program.run(21500, 0.3, 550, 450, .5)
    # Program.run(21500, 0.3, 550, 450, 1)
