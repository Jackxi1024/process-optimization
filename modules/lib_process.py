import numpy as np
import logging

from modules.lib_basics import Basics
from modules.class_stream import Stream
from modules.lib_solvers import Solvers



class Reactor():

    k1 = 43 # in 1/h
    k2 = 172 # in 1/h
    k3 = 258 # in 1/h
    V = 1e3 # in ft^3
    p = 50 # in lb/ft^3

    def __init__(self, FEED_A, FEED_B, RECYCLE):
        # set-up for logging of reactor. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'Reactor'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Reactor instance is initialised")

        # needed for system and jacobian functions to access data.
        self._FEED_A = FEED_A; self._FEED_B = FEED_B; self._RECYCLE = RECYCLE;


    def __del__(self):
        self.logger.debug("Reactor instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED_A, self._FEED_B, self._RECYCLE


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


    def __del__(self):
        self.logger.debug("Heat Exchanger instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED


   # calls the sequential approach for the heat exchanger
    def SEQ(self):

        self.logger.debug("SEQ heat exchanger called")

        OUTFLOW = Stream()
        OUTFLOW = self._FEED

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

        # molecular weight ratios of compounds
        self._MW = np.array([1, 1, 2, 2, 1, 3])

        # needed for system and jacobian functions to access data.
        self._FEED = FEED


    def __del__(self):
        self.logger.debug("Decanter instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED, self._MW


   # calls the sequential approach for the decanter
    def SEQ(self, OUTFLOW_GUESS, WASTE_GUESS):

        self.logger.debug("SEQ decanter called")

        OUTFLOW_GUESS = OUTFLOW_GUESS.F
        WASTE_GUESS = WASTE_GUESS.F

        # generate remaining guesses based on FEFF_guess
        X_OUTFLOW_guess = (OUTFLOW_GUESS[4:6]/self._MW[4:6])/sum(OUTFLOW_GUESS/self._MW)
        X_WASTE_guess = (WASTE_GUESS[4:6]/self._MW[4:6])/sum(WASTE_GUESS/self._MW)

        # compile guesses
        DATA = np.concatenate((OUTFLOW_GUESS, WASTE_GUESS, X_OUTFLOW_guess, X_WASTE_guess))

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

        Basics().mass_balance_check([self._FEED], [WASTE, OUTFLOW])

        return OUTFLOW, WASTE


    # returns the solution vector of the system of equations given the values in DATA
    def System(self, DATA):

        # reassign values to local vars to make the equations more readable
        FoutA = DATA[0]; FoutB = DATA[1]; FoutC = DATA[2]; FoutE = DATA[3]; FoutP = DATA[4]; FoutG = DATA[5];
        FwasteA = DATA[6]; FwasteB = DATA[7]; FwasteC = DATA[8]; FwasteE = DATA[9]; FwasteP = DATA[10]; FwasteG = DATA[11];
        xoutP = DATA[12]; xoutG = DATA[13];
        xwasteP = DATA[14]; xwasteG = DATA[15];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];
        MWA = self._MW[0]; MWB = self._MW[1]; MWC = self._MW[2]; MWE = self._MW[3]; MWP = self._MW[4]; MWG = self._MW[5]; 

        # total flows
        Mout = abs(FoutA/MWA) + abs(FoutB/MWB) + abs(FoutC/MWC) + abs(FoutE/MWE) + abs(FoutP/MWP) + abs(FoutG/MWG)
        Mwaste = abs(FwasteA/MWA) + abs(FwasteB/MWB) + abs(FwasteC/MWC) + abs(FwasteE/MWE) + abs(FwasteP/MWP) + abs(FwasteG/MWG)

        # activity coefficients
        if xoutP <= 0.0 or xoutG <= 0.0 or xoutP >= 1.0 or xoutG >= 1.0:
            gamma_outP = 1
            gamma_outG = 1     
        else:
            gamma_outP = np.exp(2.99/((1+(2.99*xoutP)/(9.34*xoutG))**2))
            gamma_outG = np.exp(9.34/((1+(9.34*xoutG)/(2.99*xoutP))**2)) 

        if xwasteP <= 0.0 or xwasteG <= 0.0 or xwasteP >= 1.0 or xwasteG >= 1.0:
            gamma_wasteP = 1
            gamma_wasteG = 1
        else:
            gamma_wasteP = np.exp(3.47/((1+(3.47*xwasteP)/(0.48*xwasteG))**2))
            gamma_wasteG = np.exp(0.48/((1+(0.48*xwasteG)/(3.47*xwasteP))**2))

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
            xoutP - (FoutP/MWP)/Mout,
            xoutG - (FoutG/MWG)/Mout,
            xwasteP - (FwasteP/MWP)/Mwaste,
            xwasteG - (FwasteG/MWG)/Mwaste,
            xoutP * gamma_outP - xwasteP * gamma_wasteP,
            xoutG * gamma_outG - xwasteG * gamma_wasteG
        ])


    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian(self, DATA):

        # reassign values to local vars to make the equations more readable
        FoutA = DATA[0]; FoutB = DATA[1]; FoutC = DATA[2]; FoutE = DATA[3]; FoutP = DATA[4]; FoutG = DATA[5];
        FwasteA = DATA[6]; FwasteB = DATA[7]; FwasteC = DATA[8]; FwasteE = DATA[9]; FwasteP = DATA[10]; FwasteG = DATA[11];
        xoutP = DATA[12]; xoutG = DATA[13];
        xwasteP = DATA[14]; xwasteG = DATA[15];
        FA = self._FEED.F[0]; FB = self._FEED.F[1]; FC = self._FEED.F[2]; FE = self._FEED.F[3]; FP = self._FEED.F[4]; FG = self._FEED.F[5];
        MWA = self._MW[0]; MWB = self._MW[1]; MWC = self._MW[2]; MWE = self._MW[3]; MWP = self._MW[4]; MWG = self._MW[5]; 

        # total flows
        Mout = abs(FoutA/MWA) + abs(FoutB/MWB) + abs(FoutC/MWC) + abs(FoutE/MWE) + abs(FoutP/MWP) + abs(FoutG/MWG)
        Mwaste = abs(FwasteA/MWA) + abs(FwasteB/MWB) + abs(FwasteC/MWC) + abs(FwasteE/MWE) + abs(FwasteP/MWP) + abs(FwasteG/MWG)

        # activity coefficients
        if xoutP <= 0.0 or xoutG <= 0.0 or xoutP >= 1.0 or xoutG >= 1.0:
            gamma_outP = 1
            gamma_outG = 1     
        else:
            gamma_outP = np.exp(2.99/((1+(2.99*xoutP)/(9.34*xoutG))**2))
            gamma_outG = np.exp(9.34/((1+(9.34*xoutG)/(2.99*xoutP))**2)) 

        if xwasteP <= 0.0 or xwasteG <= 0.0 or xwasteP >= 1.0 or xwasteG >= 1.0:
            gamma_wasteP = 1
            gamma_wasteG = 1
        else:
            gamma_wasteP = np.exp(3.47/((1+(3.47*xwasteP)/(0.48*xwasteG))**2))
            gamma_wasteG = np.exp(0.48/((1+(0.48*xwasteG)/(3.47*xwasteP))**2))

       
        # jacobian
        return np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [FoutP/(MWA*MWP*Mout**2), FoutP/(MWB*MWP*Mout**2), FoutP/(MWC*MWP*Mout**2), FoutP/(MWE*MWP*Mout**2), (FoutP-MWP*Mout)/((MWP*Mout)**2), FoutP/(MWG*MWP*Mout**2), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [FoutG/(MWA*MWG*Mout**2), FoutG/(MWB*MWG*Mout**2), FoutG/(MWC*MWG*Mout**2), FoutG/(MWE*MWG*Mout**2), FoutG/(MWP*MWG*Mout**2), (FoutG-MWG*Mout)/((MWG*Mout)**2), 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, FwasteP/(MWA*MWP*Mwaste**2), FwasteP/(MWB*MWP*Mwaste**2), FwasteP/(MWC*MWP*Mwaste**2), FwasteP/(MWE*MWP*Mwaste**2), (FwasteP-MWP*Mwaste)/((MWP*Mwaste)**2), FwasteP/(MWG*MWP*Mwaste**2), 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, FwasteG/(MWA*MWG*Mwaste**2), FwasteG/(MWB*MWG*Mwaste**2), FwasteG/(MWC*MWG*Mwaste**2), FwasteG/(MWE*MWG*Mwaste**2), FwasteG/(MWP*MWG*Mwaste**2), (FwasteG-MWG*Mwaste)/((MWG*Mwaste)**2), 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, gamma_outP+xoutP*(-(2*(2.99**2)*gamma_outP)/(9.34*xoutG*(1+(2.99*xoutP)/(9.34*xoutG))**3)), xoutP*((2*(2.99**2)*9.34**2*xoutP*xoutG*gamma_outP)/((9.34*xoutG+2.99*xoutP)**3)), -(gamma_wasteP+xwasteP*(-(2*3.47**2*gamma_wasteP)/(0.48*xwasteG*(1+(3.47*xwasteP)/(0.48*xwasteG))**3))), -xwasteP*((2*(3.47**2)*0.48**2*xwasteP*xwasteG*gamma_wasteP)/((0.48*xwasteG+3.47*xwasteP)**3))],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xoutG*( (2*(9.34**2)*2.99**2*xoutG*xoutP*gamma_outG)/((2.99*xoutP+9.34*xoutG)**3)), gamma_outG+xoutG*(-(2*(9.34**2)*gamma_outG)/(2.99*xoutP*(1+(9.34*xoutG)/(2.99*xoutP))**3)), -xwasteG*((2*(0.48**2)*3.47**2*xwasteG*xwasteP*gamma_wasteG)/((3.47*xwasteP+0.48*xwasteG)**3)), -(gamma_wasteG+xwasteG*(-(2*(0.48**2)*gamma_wasteG)/(3.47*xwasteP*(1+(0.48*xwasteG)/(3.47*xwasteP))**3)))] 
        ])

    # returns the values of the jacobian matrix for the system of equations given the values in DATA
    def Jacobian_Feed(self):

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
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
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


    def __del__(self):
        self.logger.debug("Distillation column instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED


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
        if PURGERATIO < 0.001: PURGERATIO = 0.001
        if PURGERATIO > 0.999: PURGERATIO = 0.999
        self._PURGERATIO = PURGERATIO


    def __del__(self):
        self.logger.debug("Distillation column instance is closed")

        # remove saved data just to be sure it cannot be reused on next iteration
        del self._FEED, self._PURGERATIO



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

