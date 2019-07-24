import numpy as np
import logging


from modules.process.lib_basics import Basics


#
# DEFINITION OF SELECTED SOLVERS.
# AVAILABLE: Netwon, ArmijoLineSearch, GradientDescent, DirectSubstitution, Wegstein, Broyden, Secant
#

# solver for units. Working: ArmijoLineSearch, Netwon. Can be used but do not converge: GradientDescent, Broyden
SOLVER_UNITS = "ArmijoLineSearch"

# solver for units. Working: Wegstein, DirectSubstitution.
SOLVER_TEARSTREAM = "Wegstein"

# solver for design specification in SEQ mode. Working: Broyden, Secant.
SOLVER_DESIGNSPEC = "Broyden"

# solver for EO mode. Working: ArmijoLineSearch, Netwon. Can be used but do not converge: GradientDescent, Broyden
SOLVER_EO = "ArmijoLineSearch"


class Solvers():

    max_iterations = 1000

    newton_e1 = 1e-4
    newton_e2 = 1e-9

    gradientdescent_e1 = 1e-2
    gradientdescent_e2 = 1e-6

    armijo_e1 = 1e-9
    armijo_e2 = 1e-12
    armijo_gamma = 0.1
    armijo_delta = 0.1
    armijo_amin = 1e-15

    direct_e = 1e-6

    wegstein_e = 1e-9
    wegstein_wait = 2
    wegstein_wmin = 0
    wegstein_wmax = 5

    broyden_e1 = 1e-9
    broyden_e2 = 1e-9

    secant_e1 = 1e-3
    secant_e2 = 1e-6


    def __init__(self):

        # set-up for logging of solvers. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.WARNING
        self.logtitle = 'Solvers'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)
        self.logger.debug("Solver instance is initialised")
      
        self.solver_default = 'Newton'
        self.solver_unit = SOLVER_UNITS
        self.solver_tearstream = SOLVER_TEARSTREAM
        self.solver_designspec = SOLVER_DESIGNSPEC
        self.solver_eo = SOLVER_EO


    def solve(self, system = None, jacobian = None, guess = None, category = None):

        if category is None:
            self.logger.warning("No category was set. Using default instead.")
            solver = self.solver_default
        elif category is 'Unit':
            solver = self.solver_unit
        elif category is 'TearStream':
            solver = self.solver_tearstream
        elif category is 'DesignSpec':
            solver = self.solver_designspec
        elif category is 'EO':
            solver = self.solver_eo
        else:
            self.logger.warning("Category "+str(category)+" unknown. Using default instead.")
            solver = self.solver_default

        if solver is 'Newton':
            return self.Newton(system, jacobian, guess)
        elif solver is 'GradientDescent':
            return self.GradientDescent(system, jacobian, guess)
        elif solver is 'ArmijoLineSearch':
            return self.ArmijoLineSearch(system, jacobian, guess)
        elif solver is 'DirectSubstitution':
            return self.DirectSubstitution(system, guess)
        elif solver is 'Wegstein':
            return self.Wegstein(system, guess)
        elif solver is 'Broyden':
            return self.Broyden(system, guess)
        elif solver is 'Secant':
            return self.Secant(system, guess)
        else:
            self.logger.error("Solver "+str(solver)+" unknown, aborting.")
            return None

      
        
    def Newton(self, system, jacobian, guess):

        if system is None or jacobian is None or guess is None:
            self.logger.error("Newton solver incorrectly initialised. It needs the system, the jacobian and a guess.")
            raise Exception("Solver incorrectly initialised.")

        iterations = 0

        while iterations < self.max_iterations:

            iterations = iterations + 1
            self.logger.debug("Performing Newton iteration #"+str(iterations))

            f = system(guess)
            j = jacobian(guess)

            j_inv = Basics().inverse(j)

            step = -np.dot(j_inv, f)
            guess = guess + step

            e1 = abs(np.dot(f, f))
            e2 = abs(np.dot(step, step))

            if e1 < self.newton_e1 and e2 < self.newton_e2:
                self.logger.debug("Passed convergence test. Returning solution.")
                break

        if iterations == self.max_iterations:
            self.logger.critical("No convergence achieved.")
            raise Exception("Solver did not converge in "+str(self.max_iterations)+" iterations.")

        stats = {'Iterations': iterations, 'ConvergenceError1': e1, 'ConvergenceError2': e2}
        self.logger.info("Converged in "+str(iterations)+" iterations.")

        return guess, stats



    def GradientDescent(self, system, jacobian, guess):

        if system is None or jacobian is None or guess is None:
            self.logger.error("Gradient descent solver incorrectly initialised. It needs the system, the jacobian and a guess.")
            raise Exception("Solver incorrectly initialised.")

        iterations = 0

        while iterations < self.max_iterations:

            iterations = iterations + 1
            self.logger.debug("Performing gradient descent iteration #"+str(iterations))

            f = system(guess)
            j = jacobian(guess)

            j_tra = np.transpose(j)

            step = -np.dot(j_tra, f)
            guess = guess + step

            e1 = abs(np.dot(f, f))
            e2 = abs(np.dot(step, step))

            if e1 < self.gradientdescent_e1 and e2 < self.gradientdescent_e2:
                self.logger.debug("Passed convergence test. Returning solution.")
                break

        if iterations == self.max_iterations:
            self.logger.critical("No convergence achieved.")
            raise Exception("Solver did not converge in "+str(self.max_iterations)+" iterations.")

        stats = {'Iterations': iterations, 'ConvergenceError1': e1, 'ConvergenceError2': e2}
        self.logger.info("Converged in "+str(iterations)+" iterations.")

        return guess, stats



    def ArmijoLineSearch(self, system, jacobian, guess):

        if system is None or jacobian is None or guess is None:
            self.logger.error("Armijo line search solver incorrectly initialised. It needs the system, the jacobian and a guess.")
            raise Exception("Solver incorrectly initialised.")

        iterations = 0

        h = lambda x: 0.5*np.dot(system(x), system(x))

        while iterations < self.max_iterations:

            iterations = iterations + 1
            self.logger.debug("Performing armijo line search iteration #"+str(iterations))

            f = system(guess)
            j = jacobian(guess)

            j_inv = Basics().inverse(j)

            step = -np.dot(j_inv, f)

            a = 1
            h_new = h(guess+a*step)
            h_old = h(guess)

            while h_new-h_old > -2*self.armijo_gamma*a*h_old and a > self.armijo_amin:
                a_q = a*h_old/((2*a-1)*h_old+h_new)
                factor = max(self.armijo_delta, a_q)
                a = factor*a
                h_new = h(guess+a*step)

            guess = guess + a*step

            e1 = abs(np.dot(f, f))
            e2 = abs(np.dot(a*step, a*step))

            if e1 < self.armijo_e1 and e2 < self.armijo_e2:
                self.logger.debug("Passed convergence test. Returning solution.")
                break

        if iterations == self.max_iterations:
            self.logger.critical("No convergence achieved.")
            raise Exception("Solver did not converge in "+str(self.max_iterations)+" iterations.")

        stats = {'Iterations': iterations, 'ConvergenceError1': e1, 'ConvergenceError2': e2}
        self.logger.info("Converged in "+str(iterations)+" iterations.")

        return guess, stats



    def DirectSubstitution(self, system, guess):

        if system is None or guess is None:
            self.logger.error("Direct substitution solver incorrectly initialised. It needs the system and a guess.")
            raise Exception("Solver incorrectly initialised.")

        iterations = 0

        while iterations < self.max_iterations:

            iterations = iterations + 1
            self.logger.debug("Performing direct substitution iteration #"+str(iterations))

            f = system(guess)

            step = f - guess

            guess = guess + step

            e = abs(np.dot(step, step))

            if e < self.direct_e:
                self.logger.debug("Passed convergence test. Returning solution.")
                break

        if iterations == self.max_iterations:
            self.logger.critical("No convergence achieved.")
            raise Exception("Solver did not converge in "+str(self.max_iterations)+" iterations.")

        stats = {'Iterations': iterations, 'ConvergenceError1': e}
        self.logger.info("Converged in "+str(iterations)+" iterations.")

        return guess, stats




    def Wegstein(self, system, guess):

        if system is None or guess is None:
            self.logger.error("Wegstein solver incorrectly initialised. It needs the system and a guess.")
            raise Exception("Solver incorrectly initialised.")

        iterations = 0
        guess_prev = 0
        guess_prevprev = 0
        f_prev = 0
        f_prevprev = 0
        w = np.ones(guess.size)

        while iterations < self.max_iterations:

            iterations = iterations + 1
            self.logger.debug("Performing Wegstein iteration #"+str(iterations))

            f = system(guess)

            guess_prevprev = guess_prev
            guess_prev = guess
            f_prevprev = f_prev
            f_prev = f

            if iterations > self.wegstein_wait and iterations >= 2:
                s = np.array([0 if guess_prev[i] == guess_prevprev[i] else (f_prev[i]-f_prevprev[i])/(guess_prev[i]-guess_prevprev[i]) for i in range(0,guess.size)])
                w = 1/(1-s)
                w = np.clip(w, self.wegstein_wmin, self.wegstein_wmax)                

            guess = w * f + (1-w) * guess_prev

            step = guess - guess_prev

            e = abs(np.dot(step, step))

            if e < self.wegstein_e:
                self.logger.debug("Passed convergence test. Returning solution.")
                break


        if iterations == self.max_iterations:
            self.logger.critical("No convergence achieved.")
            raise Exception("Solver did not converge in "+str(self.max_iterations)+" iterations.")

        stats = {'Iterations': iterations, 'ConvergenceError1': e}
        self.logger.info("Converged in "+str(iterations)+" iterations.")

        return guess, stats







    def Broyden(self, system, guess):

        if system is None or guess is None:
            self.logger.error("Broyden solver incorrectly initialised. It needs the system and a guess.")
            raise Exception("Solver incorrectly initialised.")

        iterations = 0

        B = 1e6*np.identity(guess.size)

        guess_prev = None
        f_prev = None
        B_prev = None

        while iterations < self.max_iterations:

            iterations = iterations + 1
            self.logger.debug("Performing broyden iteration #"+str(iterations))

            f = system(guess)

            if guess_prev is not None:
                delta_x = guess-guess_prev
                delta_f = f - f_prev
                B = B_prev + np.outer((delta_f - np.dot(B_prev, delta_x)), delta_x)/np.dot(delta_x, delta_x)

            guess_prev = guess
            f_prev = f
            B_prev = B

            B_inv = Basics().inverse(B)
            step = -np.dot(B_inv, f)
 
            guess = guess + step

            e1 = abs(np.dot(f, f))
            e2 = abs(np.dot(step, step))

            if e1 < self.broyden_e1 and e2 < self.broyden_e2:
                self.logger.debug("Passed convergence test. Returning solution.")
                break

        if iterations == self.max_iterations:
            self.logger.critical("No convergence achieved.")
            raise Exception("Solver did not converge in "+str(self.max_iterations)+" iterations.")

        stats = {'Iterations': iterations, 'ConvergenceError1': e1, 'ConvergenceError2': e2}
        self.logger.info("Converged in "+str(iterations)+" iterations.")

        return guess, stats



    def Secant(self, system, guess):

        if system is None or guess is None:
            self.logger.error("Secant solver incorrectly initialised. It needs the system and a guess.")
            raise Exception("Solver incorrectly initialised.")

        iterations = 0

        guess_prev = guess * 0.99
        f_prev = system(guess_prev)

        while iterations < self.max_iterations:

            iterations = iterations + 1
            self.logger.info("Performing secant iteration #"+str(iterations))

            f = system(guess)

            step = - f * (guess-guess_prev)/(f-f_prev)

            guess_prev = guess
            f_prev = f

            guess = guess + step

            e1 = abs(np.dot(f, f))
            e2 = abs(np.dot(step, step))

            if e1 < self.secant_e1 and e2 < self.secant_e2:
                self.logger.info("Passed convergence test. Returning solution.")
                break

        if iterations == self.max_iterations:
            self.logger.critical("No convergence achieved.")
            raise Exception("Solver did not converge in "+str(self.max_iterations)+" iterations.")

        stats = {'Iterations': iterations, 'ConvergenceError1': e1, 'ConvergenceError2': e2}
        self.logger.info("Converged in "+str(iterations)+" iterations.")

        return guess, stats