import math
import random
import numpy as np

class Loader:
    def __init__(self,
                 dimensions=2,
                 train_percent=85.0):
        self.__train_percent = train_percent
        self.__tr, self.__ts = self.__loadData(dimensions)

    def __loadData(self, dim):
        data = self.__get2DData() if dim == 2 else self.__get3DData()
        ln = len(data)
        lnts = int(ln * (1 - self.__train_percent / 100))
        lntr = ln - lnts

        random.shuffle(data)
        return sorted(data[:lntr]), sorted(data[lntr:])

    @staticmethod
    def __get2DData():
        return [
            [
                [i / 10],
                [math.cos(i / 10) + random.random() * 0.2 - 0.1]
            ]
            for i in range(-60, 61)
        ]

    @staticmethod
    def __get3DData():

        return [
            [
                [i / 10, j/10],
                [math.cos(i / 10 + j/10) + random.random() * 0.2 - 0.1]
            ]
            for i in range(-40, 41) for j in range(-40, 41)
        ]

    def getTrainInp(self):
        return np.array([i[0] for i in self.__tr])

    def getTrainOut(self):
        return np.array([i[1] for i in self.__tr])

    def getTestInp(self):
        return np.array([i[0] for i in self.__ts])

    def getTestOut(self):
        return np.array([i[1] for i in self.__ts])