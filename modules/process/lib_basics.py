import numpy as np
import pandas as pd
import logging


def singleton(cls, *args, **kw):
    instances = {}
    def _singleton():
       if cls not in instances:
            instances[cls] = cls(*args, **kw)
       return instances[cls]
    return _singleton


def exceptionreport(func):
    def new_func(*args, **kwargs):
        # try:
        #     return func(*args, **kwargs)
        # except Exception as error:
        #     print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print("OPERATION FAILED.")
        #     print("REASON: ", error)
        #     print("STOPPING EXECUTION.")
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     exit
        return func(*args, **kwargs)
    return new_func


@singleton
class Basics():

    iterations = {'Unit': None, 'TearStream': None, 'DesignSpec': None, 'EO': None}
    success = False

    def __init__(self):

        # set-up for logging of solvers. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.WARNING
        self.logtitle = 'Basics'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Basics instance is initialised")


    def inverse(self, matrix):
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            self.logger.error("Encountered singular matrix. Cannot invert.")
            return None
        else:
            return inverse


    def mass_balance_check(self, inflows, outflows, by_species = True):
        total_inflows = np.zeros(6)
        try:
            for i in range(0,len(inflows)):
                total_inflows = total_inflows + inflows[i].F
        except:
            total_inflows = inflows.F

        total_outflows = np.zeros(6)
        try:
            for i in range(0,len(outflows)):
                total_outflows = total_outflows + outflows[i].F
        except:
            total_outflows = outflows.F

        if by_species:
            if not np.allclose(total_inflows, total_outflows): raise Exception("Unit not in mass balance")
        else:
            if not np.allclose(sum(total_inflows), sum(total_outflows)): raise Exception("Unit not in mass balance")



    def np_to_pd(self, streams):

        index = ['FEED_A', 'FEED_B', 'REACOUT', 'HEATEXOUT', 'DECANTOUT', 'WASTE', 'HEAD', 'BOT', 'RECYCLE', 'PURGE']
        columns = ['A', 'B', 'C', 'E', 'P', 'G', 'T']

        return pd.DataFrame(data = streams, index = index, columns = columns)




    def startPerformanceAssessment(self):

        self.iterations = {'Unit': None, 'TearStream': None, 'DesignSpec': None, 'EO': None}
        self.success = False


    def TrackPerformance(self, stats, category):

        iterations = stats['Iterations']
        self.iterations[category] = iterations if self.iterations[category] is None else self.iterations[category] + iterations


    def TrackSuccess(self):

        self.success = True


    def getPerformanceAssessment(self):

        if self.success is False: return {'Unit': None, 'TearStream': None, 'DesignSpec': None, 'EO': None}
        return self.iterations




