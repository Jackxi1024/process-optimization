import numpy as np
import logging
import copy

from modules.process.lib_basics import Basics
from modules.process.class_stream import Stream
from modules.process.lib_solvers import Solvers

CP_reactor = 2170 # in J/kgK
CP_water = 4180 # in J/kgK

class Reactor():

    k1 = 43 # in 1/h
    k2 = 172 # in 1/h
    k3 = 258 # in 1/h
    V = 1e3 # in m^3
    p = 50 # in kg/m^3
    DrH = 6000*1000 # in J/kg

    def __init__(self, FEED_A, FEED_B, RECYCLE):
        # set-up for logging of reactor. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.WARNING
        self.logtitle = 'Reactor'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Reactor instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED_A = copy.copy(FEED_A); self._FEED_B = copy.copy(FEED_B); self._RECYCLE = copy.copy(RECYCLE);
        self._TEMPERATURE = (FEED_A.Ftotal*FEED_A.T + FEED_B.Ftotal*FEED_B.T + RECYCLE.Ftotal*RECYCLE.T)/(FEED_A.Ftotal + FEED_B.Ftotal + RECYCLE.Ftotal)

        self.k1 = self.k1*(1+0.3*np.exp(0.01*(self._TEMPERATURE-400)))
        self.k2 = self.k2*(1+0.2*np.exp(0.01*(self._TEMPERATURE-400)))
        self.k3 = self.k3*(1+1*np.exp(0.02*(self._TEMPERATURE-600)))

        # print(self.k1, self.k2, self.k3)

    def __del__(self):
        self.logger.debug("Reactor instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED_A, self._FEED_B, self._RECYCLE, self._TEMPERATURE


    # calls the sequential approach for the reactor
    def SEQ(self, GUESS):

        self.logger.debug("SEQ reactor called")

        GUESS = GUESS.F

        # generate remaining guesses based on guess
        X_guess = GUESS/sum(GUESS)
        DATA = np.concatenate((GUESS, X_guess))

        # calculate solution based on default solvers for units, then decompose solution
        solution, stats = Solvers().solve(system = self.System, jacobian = self.Jacobian, guess = DATA, category = 'Unit')
        Basics().TrackPerformance(stats, 'Unit')
        solution_flows = solution[0:6]

        self.logger.info("Converged unit in "+str(stats["Iterations"])+" iterations.")

        # construct solution stream leaving the reactor and return it
        FEFF = Stream()
        FEFF.F = solution_flows
        FEFF.T = self._TEMPERATURE + self.DrH/(FEFF.Ftotal*CP_reactor)*(self._FEED_A.F[0]+self._FEED_B.F[0]+self._RECYCLE.F[0]-FEFF.F[0])

        Basics().mass_balance_check([self._FEED_A, self._FEED_B, self._RECYCLE], [FEFF], by_species = False)

        return FEFF


    # returns the solution vector of the system of equations given the values in DATA
    def System(self, DATA):

        # reassign values to local vars to make the equations more readable
        F1A = self._FEED_A.F[0]; F2B = self._FEED_B.F[1]; FRA = self._RECYCLE.F[0]; FRB = self._RECYCLE.F[1]; FRC = self._RECYCLE.F[2]; FRE = self._RECYCLE.F[3]; FRP = self._RECYCLE.F[4]; FRG = self._RECYCLE.F[5]; 
        FeffA = DATA[0]; FeffB = DATA[1]; FeffC = DATA[2]; FeffE = DATA[3]; FeffP = DATA[4]; FeffG = DATA[5];
        xA = DATA[6]; xB = DATA[7]; xC = DATA[8]; xE = DATA[9]; xP = DATA[10]; xG = DATA[11];
        k1 = self.k1; k2 = self.k2; k3 = self.k3; Vp = self.V*self.p;

        # total flow
        Feff = FeffA + FeffB + FeffC + FeffE + FeffP + FeffG

        # system of equations
        return np.array([
            FeffA - F1A - FRA + k1 * xA * xB * Vp,
            FeffB - F2B - FRB + (k1 * xA + k2 * xC) * xB * Vp,
            FeffC - FRC - (2 * k1 * xA * xB - 2 * k2 * xB * xC - k3 * xP * xC) * Vp,
            FeffE - FRE - (2 * k2 * xB * xC) * Vp,
            FeffP - FRP - (k2 * xB * xC - 0.5 * k3 * xP * xC) * Vp,
            FeffG - FRG - (1.5 * k3 * xP * xC) * Vp,
            xA - FeffA / Feff,
            xB - FeffB / Feff,
            xC - FeffC / Feff,
            xE - FeffE / Feff,
            xP - FeffP / Feff,
            xG - FeffG / Feff
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian(self, DATA):

        # reassign values to local vars to make the equations more readable
        F1A = self._FEED_A.F[0]; F2B = self._FEED_B.F[1]; FRA = self._RECYCLE.F[0]; FRB = self._RECYCLE.F[1]; FRC = self._RECYCLE.F[2]; FRE = self._RECYCLE.F[3]; FRP = self._RECYCLE.F[4]; FRG = self._RECYCLE.F[5]; 
        FeffA = DATA[0]; FeffB = DATA[1]; FeffC = DATA[2]; FeffE = DATA[3]; FeffP = DATA[4]; FeffG = DATA[5];
        xA = DATA[6]; xB = DATA[7]; xC = DATA[8]; xE = DATA[9]; xP = DATA[10]; xG = DATA[11];
        k1 = self.k1; k2 = self.k2; k3 = self.k3; Vp = self.V*self.p;

        # total flow
        Feff = FeffA + FeffB + FeffC + FeffE + FeffP + FeffG

        # jacobian
        return np.array([
            [1, 0, 0, 0, 0, 0, k1*xB*Vp, k1*xA*Vp, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, k1*xB*Vp, (k1*xA+k2*xC)*Vp, k2*xB*Vp, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, -2*k1*xB*Vp, -(2*k1*xA-2*k2*xC)*Vp, -(-2*k2*xB-k3*xP)*Vp, 0, k3*xC*Vp, 0],
            [0, 0, 0, 1, 0, 0, 0, -2*k2*xC*Vp, -2*k2*xB*Vp, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, -k2*xC*Vp, -(k2*xB-0.5*k3*xP)*Vp, 0, 0.5*k3*xC*Vp, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, -1.5*k3*xP*Vp, 0 , -1.5*k3*xC*Vp, 0],
            [(FeffA-Feff)/(Feff**2), FeffA/(Feff**2), FeffA/(Feff**2), FeffA/(Feff**2), FeffA/(Feff**2), FeffA/(Feff**2), 1, 0, 0, 0, 0, 0],
            [FeffB/(Feff**2), (FeffB-Feff)/(Feff**2), FeffB/(Feff**2), FeffB/(Feff**2), FeffB/(Feff**2), FeffB/(Feff**2), 0, 1, 0, 0, 0, 0],
            [FeffC/(Feff**2), FeffC/(Feff**2), (FeffC-Feff)/(Feff**2), FeffC/(Feff**2), FeffC/(Feff**2), FeffC/(Feff**2), 0, 0, 1, 0, 0, 0],
            [FeffE/(Feff**2), FeffE/(Feff**2), FeffE/(Feff**2), (FeffE-Feff)/(Feff**2), FeffE/(Feff**2), FeffE/(Feff**2), 0, 0, 0, 1, 0, 0],
            [FeffP/(Feff**2), FeffP/(Feff**2), FeffP/(Feff**2), FeffP/(Feff**2), (FeffP-Feff)/(Feff**2), FeffP/(Feff**2), 0, 0, 0, 0, 1, 0],
            [FeffG/(Feff**2), FeffG/(Feff**2), FeffG/(Feff**2), FeffG/(Feff**2), FeffG/(Feff**2), (FeffG-Feff)/(Feff**2), 0, 0, 0, 0, 0, 1]
        ])




class HeatExchanger():

    def __init__(self, FEED):
        # set-up for logging of heat exchanger. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.WARNING
        self.logtitle = 'HeatEx'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("HeatEx instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED = copy.copy(FEED)
        self._TEMPERATURE = FEED.T


    def __del__(self):
        self.logger.debug("Heat Exchanger instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED, self._TEMPERATURE


   # calls the sequential approach for the heat exchanger
    def SEQ(self, Temperature):

        self.logger.debug("SEQ heat exchanger called")

        OUTFLOW = Stream()
        OUTFLOW = self._FEED
        OUTFLOW.T = Temperature

        Basics().mass_balance_check([self._FEED], [OUTFLOW])

        return OUTFLOW




class Decanter():

    def __init__(self, FEED):
        # set-up for logging of decanter. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.WARNING
        self.logtitle = 'Decanter'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Decanter instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED = copy.copy(FEED)
        self._TEMPERATURE = FEED.T


    def __del__(self):
        self.logger.debug("Decanter instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED, self._TEMPERATURE


   # calls the sequential approach for the decanter
    def SEQ(self, OUTFLOW_GUESS, WASTE_GUESS):

        self.logger.debug("SEQ decanter called")

        OUTFLOW_GUESS = OUTFLOW_GUESS.F
        WASTE_GUESS = WASTE_GUESS.F

        # compile guesses
        DATA = np.concatenate((OUTFLOW_GUESS, WASTE_GUESS))

        # calculate solution based on default solvers for units, then decompose solution
        solution, stats = Solvers().solve(system = self.System, jacobian = self.Jacobian, guess = DATA, category = 'Unit')
        Basics().TrackPerformance(stats, 'Unit')
        solution_waste_flows = solution[6:12]
        solution_outlet_flows = solution[0:6]

        self.logger.info("Converged unit in "+str(stats["Iterations"])+" iterations.")

        WASTE = Stream()
        OUTFLOW = Stream()
        WASTE.F = solution_waste_flows
        OUTFLOW.F = solution_outlet_flows
        WASTE.T = self._TEMPERATURE
        OUTFLOW.T = self._TEMPERATURE

        Basics().mass_balance_check([self._FEED], [WASTE, OUTFLOW])

        return OUTFLOW, WASTE


    # returns the solution vector of the system of equations given the values in DATA
    def System(self, DATA):

        # reassign values to local vars to make the equations more readable
        FoutA = DATA[0]; FoutB = DATA[1]; FoutC = DATA[2]; FoutE = DATA[3]; FoutP = DATA[4]; FoutG = DATA[5];
        FwasteA = DATA[6]; FwasteB = DATA[7]; FwasteC = DATA[8]; FwasteE = DATA[9]; FwasteP = DATA[10]; FwasteG = DATA[11];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];

        temp_factor_P = 0.3*(1-np.exp(-0.005*(self._TEMPERATURE-300)))
        temp_factor_G = 0.95*np.exp(-0.003*(self._TEMPERATURE-300))
        temp_factor_2 = max(1-np.exp(-0.005*(self._TEMPERATURE-400)), 0.2)

        # print(FwasteP, FwasteG, temp_factor_P, temp_factor_G)

        # system of equations
        return np.array([
            FoutA - FA,
            FoutB - FB,
            FoutC - FC,
            FoutE - FE,
            FwasteA,
            FwasteB,
            FwasteC,
            FwasteE,
            FoutP + FwasteP - FP,
            FoutG + FwasteG - FG,
            FwasteP - temp_factor_P*temp_factor_2*FP - 0.01*FwasteG + 0.015*temp_factor_P*(FA+FB),
            FwasteG - temp_factor_G*FG + 0.01*FoutE
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian(self, DATA):

        # reassign values to local vars to make the equations more readable
        FoutA = DATA[0]; FoutB = DATA[1]; FoutC = DATA[2]; FoutE = DATA[3]; FoutP = DATA[4]; FoutG = DATA[5];
        FwasteA = DATA[6]; FwasteB = DATA[7]; FwasteC = DATA[8]; FwasteE = DATA[9]; FwasteP = DATA[10]; FwasteG = DATA[11];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];
       
        # jacobian
        return np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -0.01],
            [0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 1]
        ])






class DistillationColumn():

    def __init__(self, FEED):
        # set-up for logging of distillation column. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.WARNING
        self.logtitle = 'Distillation'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Distillation column instance is initialised")

        # needed for system and jacobian functions to access data.
        FEED.F[FEED.F == 0.0] = 0.001
        self._FEED = FEED
        self._TEMPERATURE = FEED.T


    def __del__(self):
        self.logger.debug("Distillation column instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED, self._TEMPERATURE


   # calls the sequential approach for the distillation column
    def SEQ(self):

        self.logger.debug("SEQ distillation called")

        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];

        head_flows = np.array([
            1,
            1000/FA,
            10*FC/(FC+FE),
            50*FE/(FE+FP),
            max(FP-0.05*FE, 0),
            FG/(FP+FG)
        ])

        bottom_flows = self._FEED.F - head_flows

        HEAD = Stream()
        BOTTOM = Stream()
        HEAD.F = head_flows
        BOTTOM.F = bottom_flows
        HEAD.T = self._TEMPERATURE + 0.5*(800 - self._TEMPERATURE)
        BOTTOM.T = self._TEMPERATURE - 0.5*(self._TEMPERATURE - 500)

        Basics().mass_balance_check([self._FEED], [HEAD, BOTTOM])

        return HEAD, BOTTOM






class Splitter():

    def __init__(self, FEED, PURGERATIO):
        # set-up for logging of splitter. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.WARNING
        self.logtitle = 'Splitter'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Splitter instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED = FEED
        self._TEMPERATURE = FEED.T
        if PURGERATIO < 0.001: PURGERATIO = 0.001
        if PURGERATIO > 1: PURGERATIO = 1
        self._PURGERATIO = PURGERATIO


    def __del__(self):
        self.logger.debug("Distillation column instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED, self._PURGERATIO, self._TEMPERATURE



   # calls the sequential approach for the splitter
    def SEQ(self):

        self.logger.debug("SEQ splitter called")

        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];

        recycle_flows = np.array([
            (1-self._PURGERATIO)*FA,
            (1-self._PURGERATIO)*FB,
            (1-self._PURGERATIO)*FC,
            (1-self._PURGERATIO)*FE,
            (1-self._PURGERATIO)*FP,
            0
        ])

        purge_flows = self._FEED.F - recycle_flows

        RECYCLE = Stream()
        PURGE = Stream()
        RECYCLE.F = recycle_flows
        PURGE.F = purge_flows
        RECYCLE.T = self._TEMPERATURE
        PURGE.T = self._TEMPERATURE

        Basics().mass_balance_check([self._FEED], [RECYCLE, PURGE])

        return RECYCLE, PURGE