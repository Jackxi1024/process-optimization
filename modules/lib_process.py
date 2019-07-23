import numpy as np
import logging

from modules.lib_basics import Basics
from modules.class_stream import Stream
from modules.lib_solvers import Solvers

CP_reactor = 2170 # in J/kgK
CP_water = 4180 # in J/kgK

class Reactor():

    k1 = 43 # in 1/h
    k2 = 172 # in 1/h
    k3 = 258 # in 1/h
    V = 1e3 # in m^3
    p = 50 # in kg/m^3
    DrH = 100 # in J/kg

    def __init__(self, FEED_A, FEED_B, RECYCLE):
        # set-up for logging of reactor. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'Reactor'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Reactor instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED_A = FEED_A; self._FEED_B = FEED_B; self._RECYCLE = RECYCLE;
        self._TEMPERATURE = (FEED_A.Ftotal*FEED_A.T + FEED_B.Ftotal*FEED_B.T + RECYCLE.Ftotal*RECYCLE.T)/(FEED_A.Ftotal + FEED_B.Ftotal + RECYCLE.Ftotal)


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
        print(self._TEMPERATURE + self.DrH/CP_reactor*(self._FEED_A.F[0]+self._FEED_B.F[0]+self._RECYCLE.F[0]-FEFF.F[0]))
        FEFF.T = self._TEMPERATURE + self.DrH/CP_reactor*(self._FEED_A.F[0]+self._FEED_B.F[0]+self._RECYCLE.F[0]-FEFF.F[0])

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


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian_Feed(self):

        # jacobian
        return np.array([
            [-1, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])





class HeatExchanger():

    def __init__(self, FEED):
        # set-up for logging of heat exchanger. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'HeatEx'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("HeatEx instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED = FEED
        self._TEMPERATURE = FEED.T


    def __del__(self):
        self.logger.debug("Heat Exchanger instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED, self._TEMPERATURE


   # calls the sequential approach for the heat exchanger
    def SEQ(self):

        self.logger.debug("SEQ heat exchanger called")

        OUTFLOW = Stream()
        OUTFLOW = self._FEED
        OUTFLOW.T = self._TEMPERATURE

        Basics().mass_balance_check([self._FEED], [OUTFLOW])

        return OUTFLOW


    # returns the solution vector of the system of equations given the values in DATA
    def System(self, DATA):

        # reassign values to local vars to make the equations more readable
        FeffA = DATA[0]; FeffB = DATA[1]; FeffC = DATA[2]; FeffE = DATA[3]; FeffP = DATA[4]; FeffG = DATA[5];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];

        # system of equations
        return np.array([
            FeffA - FA,
            FeffB - FB,
            FeffC - FC,
            FeffE - FE,
            FeffP - FP,
            FeffG - FG
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian(self, DATA):

        # jacobian
        return np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian_Feed(self):

        # jacobian
        return np.array([
            [-1, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1]
        ])






class Decanter():

    def __init__(self, FEED):
        # set-up for logging of decanter. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'Decanter'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Decanter instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED = FEED
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

        temp_factor_P = 0.3*(1-np.exp(-0.0025*(self._TEMPERATURE-300)))
        temp_factor_G = 0.95*(1-np.exp(-0.0015*(self._TEMPERATURE-300)))

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
            FwasteP - temp_factor_P*FP - 0.05*FwasteG,
            FwasteG - temp_factor_G*FG + 0.05*FoutE
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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -0.05],
            [0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian_Feed(self):

        temp_factor_P = 0.3*(1-np.exp(-0.0025*(self._TEMPERATURE-300)))
        temp_factor_G = 0.95*(1-np.exp(-0.0015*(self._TEMPERATURE-300)))

        # jacobian
        return np.array([
            [-1, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
            [0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0,-temp_factor_P, 0],
            [0, 0, 0, 0, 0, temp_factor_G]
        ])











class DistillationColumn():

    def __init__(self, FEED):
        # set-up for logging of distillation column. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
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
            FP-0.1*(1-50/(FE+FP))*FE,
            FG/(FP+FG)
        ])

        bottom_flows = self._FEED.F - head_flows

        HEAD = Stream()
        BOTTOM = Stream()
        HEAD.F = head_flows
        BOTTOM.F = bottom_flows
        HEAD.T = self._TEMPERATURE + 0.5*(1000 - self._TEMPERATURE)
        BOTTOM.T = self._TEMPERATURE - 0.5*(self._TEMPERATURE - 300)

        Basics().mass_balance_check([self._FEED], [HEAD, BOTTOM])

        return HEAD, BOTTOM


    # returns the solution vector of the system of equations given the values in DATA
    def System(self, DATA):

        # reassign values to local vars to make the equations more readable
        FheadA = DATA[0]; FheadB = DATA[1]; FheadC = DATA[2]; FheadE = DATA[3]; FheadP = DATA[4]; FheadG = DATA[5];
        FbotA = DATA[6]; FbotB = DATA[7]; FbotC = DATA[8]; FbotE = DATA[9]; FbotP = DATA[10]; FbotG = DATA[11];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];

        # system of equations
        return np.array([
            FheadA - 1,
            FheadB - 1000/FA,
            FheadC - 10*FC/(FC+FE),
            FheadE - 50*FE/(FE+FP),
            FheadP - (FP - 0.1*(1-50/(FE+FP))*FE),
            FheadG - FG/(FP+FG),
            FbotA - FA + 1,
            FbotB - FB + 1000/FA,
            FbotC - FC + 10*FC/(FC+FE),
            FbotE - FE + 50*FE/(FE+FP),
            FbotP - 0.1*(1-50/(FE+FP))*FE,
            FbotG - FG + FG/(FP+FG)
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian(self, DATA):

        # reassign values to local vars to make the equations more readable
        FheadA = DATA[0]; FheadB = DATA[1]; FheadC = DATA[2]; FheadE = DATA[3]; FheadP = DATA[4]; FheadG = DATA[5];
        FbotA = DATA[6]; FbotB = DATA[7]; FbotC = DATA[8]; FbotE = DATA[9]; FbotP = DATA[10]; FbotG = DATA[11];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];

        # jacobian
        return np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian_Feed(self):

        # reassign values to local vars to make the equations more readable
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];


        # jacobian
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1000/(FC**2), 0, 0, 0],
            [0, 0, -10*FE/((FC+FE)**2), 10*FC/((FC+FE)**2), 0, 0],
            [0, 0, 0, -50*FP/((FE+FP)**2), 50*FE/((FE+FP)**2), 0],
            [0, 0, 0, 0.1-5/(FE+FP)+5*FE/((FE+FP)**2), 5*FE/((FP+FE)**2)-1, 0],
            [0, 0, 0, 0, FG/((FP+FG)**2), -FP/((FP+FG)**2)],
            [-1, 0, 0, 0, 0, 0],
            [-1000/(FA**2), -1, 0, 0, 0, 0],
            [0, 0, 10/(FC+FE)-10*FC/((FC+FE)**2)-1, -10*FC/((FE+FC)**2), 0, 0],
            [0, 0, 0, 50/(FE+FP)-50*FE/((FE+FP)**2)-1, -50*FE/((FE+FP)**2), 0],
            [0, 0, 0, 5/(FE+FP)-5*FE/((FE+FP)**2)-0.1, -5*FE/((FP+FE)**2), 0],
            [0, 0, 0, 0, -FG/((FP+FG)**2), 1/(FP+FG)-FG/((FP+FG)**2)-1]
        ])










class Splitter():

    def __init__(self, FEED, PURGERATIO):
        # set-up for logging of splitter. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'Splitter'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Splitter instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED = FEED
        self._TEMPERATURE = FEED.T
        if PURGERATIO < 0.001: PURGERATIO = 0.001
        if PURGERATIO > 0.999: PURGERATIO = 0.999
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


    # returns the solution vector of the system of equations given the values in DATA
    def System(self, DATA):

        # reassign values to local vars to make the equations more readable
        FrecA = DATA[0]; FrecB = DATA[1]; FrecC = DATA[2]; FrecE = DATA[3]; FrecP = DATA[4]; FrecG = DATA[5];
        FpurgeA = DATA[6]; FpurgeB = DATA[7]; FpurgeC = DATA[8]; FpurgeE = DATA[9]; FpurgeP = DATA[10]; FpurgeG = DATA[11];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];
        n = self._PURGERATIO

        # system of equations
        return np.array([
            FrecA - (1-n)*FA,
            FrecB - (1-n)*FB,
            FrecC - (1-n)*FC,
            FrecE - (1-n)*FE,
            FrecP - (1-n)*FP,
            FrecG - 0,
            FpurgeA - n*FA,
            FpurgeB - n*FB,
            FpurgeC - n*FC,
            FpurgeE - n*FE,
            FpurgeP - n*FP,
            FpurgeG - FG
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian(self, DATA):

        # reassign values to local vars to make the equations more readable
        FrecA = DATA[0]; FrecB = DATA[1]; FrecC = DATA[2]; FrecE = DATA[3]; FrecP = DATA[4]; FrecG = DATA[5];
        FpurgeA = DATA[6]; FpurgeB = DATA[7]; FpurgeC = DATA[8]; FpurgeE = DATA[9]; FpurgeP = DATA[10]; FpurgeG = DATA[11];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];
        n = self._PURGERATIO

        # jacobian
        return np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian_Ratio(self):

        # reassign values to local vars to make the equations more readable
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];

        # jacobian
        return np.array([
            FA,
            FB,
            FC,
            FE,
            FP,
            0,
            -FA,
            -FB,
            -FC,
            -FE,
            -FP,
            0
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian_Feed(self):

        n = self._PURGERATIO

        # jacobian
        return np.array([
            [-(1-n), 0, 0, 0, 0, 0],
            [0, -(1-n), 0, 0, 0, 0],
            [0, 0, -(1-n), 0, 0, 0],
            [0, 0, 0, -(1-n), 0, 0],
            [0, 0, 0, 0, -(1-n), 0],
            [0, 0, 0, 0, 0, 0],
            [-n, 0, 0, 0, 0, 0],
            [0, -n, 0, 0, 0, 0],
            [0, 0, -n, 0, 0, 0],
            [0, 0, 0, -n, 0, 0],
            [0, 0, 0, 0, -n, 0],
            [0, 0, 0, 0, 0, -1],
        ])

