import numpy as np
import logging


class Stream():


    def __init__(self):
        # set-up for logging of streams. Level options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.loglevel = logging.INFO
        self.logtitle = 'Stream'
        self.logger = logging.getLogger(self.logtitle)
        self.logger.setLevel(self.loglevel)

        self.logger.debug("Streams instance is initialised")

        self._F = None
        self._T = None


    @property
    def F(self):
        if self._F is None: raise Exception("Called undefined stream")
        return self._F

    @F.setter
    def F(self, value):
        self._F = value
        if np.any(value < 0): raise Exception("Stream contains negative flows")
        if np.any(value > 1e6): raise Exception("Stream contains very large flows")

    @property
    def Ftotal(self):
        if self._F is None: raise Exception("Called undefined stream")
        return sum(self._F)

    @property
    def T(self):
        if self._T is None: raise Exception("Called undefined temperature")
        return self._T

    @T.setter
    def T(self, value):
        self._T = value
        if np.any(value < 300): raise Exception("Temperature too low")
        if np.any(value > 1000): raise Exception("Temperature too large")

