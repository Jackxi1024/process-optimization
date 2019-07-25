import logging
import numpy as np
import pandas as pd
from numba import jit
import random

# custom modules
import modules.constants as constants






class NelderMead():

    logger = None
    fitnessfunction = None
    dimensions = None

    # Settings
    alpha = 1
    beta = 2
    gamma = 0.5
    sigma = 0.5

    max_iterations = 1000
    parameter_threshold = 1e-6
    fitness_threshold = 1e-12
    repetition_threshold = 25
    random_restarts = 0


    def __init__(self, fitnessfunction, dimensions, settings = None):
        
        # set up logger for this module
        self.logger = logging.getLogger('neldermead'); self.logger.setLevel(logging.DEBUG)

        # set up parameters of optimization
        self.fitnessfunction = fitnessfunction
        self.dimension = dimensions

        if settings is not None:
            pass

    def run(self, vertices = None):

        if vertices is None:
            vertices = self.generateRandomVertices()
        
        data = np.hstack([vertices, np.zeros((self.dimension+1,1))])


        iterations = 0

        for i in range(self.dimension+1):
            data[i, self.dimension] = self.fitnessfunction(data[i, 0:self.dimension])
        data = data[data[:,self.dimension].argsort()[::-1]]
        best_result = data[0, 0:self.dimension+1]
        best_result_repetitions = 0


        while iterations <= self.max_iterations: 

            iterations += 1
            data = data[data[:,self.dimension].argsort()[::-1]]


            if np.all(data[0, 0:self.dimension] == best_result[0:self.dimension]):
                best_result_repetitions += 1
                if best_result_repetitions >= self.repetition_threshold:
                    self.logger.info("Stopped optimisation due to parameter repetition after %i iterations." % iterations)
                    self.logger.info(("Result: " + ', '.join(['%1.4f']*(self.dimension+1)) % tuple(data[0, 0:self.dimension+1])))
                    break
            elif np.all(abs(data[0, 0:self.dimension] - best_result[0:self.dimension]) < self.parameter_threshold):
                self.logger.info("Stopped optimisation due to parameter threshold after %i iterations." % iterations)
                self.logger.info(("Result: " + ', '.join(['%1.4f']*(self.dimension+1)) % tuple(data[0, 0:self.dimension+1])))
                best_result = data[0, 0:self.dimension+1]
                break
            elif np.all(abs(data[0, self.dimension] - best_result[self.dimension]) < self.fitness_threshold):
                self.logger.info("Stopped optimisation due to fitness threshold after %i iterations." % iterations)
                self.logger.info(("Result: " + ', '.join(['%1.4f']*(self.dimension+1)) % tuple(data[0, 0:self.dimension+1])))
                best_result = data[0, 0:self.dimension+1]
                break
            else:
                best_result_repetitions = 0
                best_result = data[0, 0:self.dimension+1]

            self.logger.info(("Iteration %1.0f with (" + ', '.join(['%1.3f']*(self.dimension+1)) + ").") % ((iterations,) + tuple(best_result)))

            p_bary = np.mean(data[0:self.dimension, 0:self.dimension], axis=0)
            p_worst = data[self.dimension, 0:self.dimension]
            p_1 = p_bary + self.alpha*(p_bary-p_worst)
            F_best = data[0, self.dimension]
            F_p1 = self.fitnessfunction(p_1)

            if F_p1 > F_best:
                p_2 = p_bary + self.beta*(p_bary-p_worst)
                F_p2 = self.fitnessfunction(p_2)

                if F_p2 > F_p1:
                    data[self.dimension, 0:self.dimension] = p_2
                    data[self.dimension, self.dimension] = F_p2
                else:
                    data[self.dimension, 0:self.dimension] = p_1
                    data[self.dimension, self.dimension] = F_p1                
                continue
    
            F_worst1 = data[self.dimension-1, self.dimension]

            if F_p1 > F_worst1:
                data[self.dimension, 0:self.dimension] = p_1
                data[self.dimension, self.dimension] = F_p1
                continue

            F_worst = data[self.dimension, self.dimension]

            if F_p1 > F_worst:
                p_better = p_1
            else:
                p_better = p_worst

            p_3 = p_better + self.gamma*(p_bary-p_better)
            F_p3 = self.fitnessfunction(p_3)

            if F_p3 > F_worst:
                data[self.dimension, 0:self.dimension] = p_3
                data[self.dimension, self.dimension] = F_p3
                continue
            
            p_best = data[0, 0:self.dimension]

            for i in range(1, self.dimension):
                data[i, 0:self.dimension] = data[i, 0:self.dimension]+self.sigma*(p_best-data[i, 0:self.dimension])
                data[i, self.dimension] = self.fitnessfunction(data[i, 0:self.dimension])
                continue


        pass 



            

    
    def generateRandomVertices(self):
        vertices = list()

        for _ in range(self.dimension+1):
            vertice = list()

            for _ in range(self.dimension):
                vertice.append(random.uniform(0.3, 0.7))
            
            vertices.append(vertice)

        return np.array(vertices)

