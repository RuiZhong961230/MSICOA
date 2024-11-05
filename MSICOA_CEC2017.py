import os
from copy import deepcopy
from scipy.stats import levy
import numpy as np
from cec17_functions import cec17_test_func


PopSize = 50
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 20
MaxFEs = DimSize * 1000
MaxIter = int(MaxFEs / PopSize)

curIter = 1

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

Func_num = 0
SuiteName = "CEC2017"


def fitness(X):
    global DimSize, Func_num
    f = [0]
    cec17_test_func(X, f, DimSize, 1, Func_num)
    return f[0]


# initialize the Pop randomly
def Initialization():
    global Pop, FitPop, DimSize
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = fitness(Pop[i])



def MSICOA():
    global Pop, FitPop, PopSize, DimSize
    Exploration()
    Exploitation()


def Exploration():
    global Pop, FitPop, PopSize, DimSize, LB, UB, curIter, MaxIter

    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    half = int(PopSize / 2)
    for i in range(half):
        I = np.random.randint(1, 3)
        w = (np.exp(curIter / MaxIter) - 1) / (np.e - 1)
        for j in range(DimSize):
            Off[i][j] = w * Pop[i][j] + np.random.rand() * (Pop[np.argmin(FitPop)][j] - I * Pop[i][j])   # Eq. (4)
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = fitness(Off[i])

    E = 2 * np.random.uniform(-1, 1) * (1 - curIter / MaxIter)
    for i in range(half, PopSize):
        if np.random.rand() > 0.5:
            Off[i] = Pop[np.argmin(FitPop)] - E * np.abs(np.random.uniform(-1, 1, DimSize) * Pop[np.argmin(FitPop)] - Pop[i])
        else:
            Off[i] = Pop[np.argmin(FitPop)] - E * np.abs(np.random.uniform(-1, 1, DimSize) * Pop[np.argmin(FitPop)] - Pop[i]) + levy.rvs()
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = fitness(Off[i])

    for i in range(PopSize):  # Update
        if FitOff[i] < FitPop[i]:
            FitPop[i] = FitOff[i]
            Pop[i] = deepcopy(Off[i])


def Exploitation():
    global curIter, Pop, FitPop, PopSize, DimSize, LB, UB, curIter, MaxIter
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    for i in range(PopSize):
        if FitPop[i] > min(FitPop):
            Off[i] = Pop[np.argmin(FitPop)] + np.random.normal(0, 1, DimSize) * np.abs(Pop[i] - Pop[np.argmin(FitPop)])
        else:
            Off[i] = Pop[i] + np.random.uniform(-1, 1) * (np.abs(Pop[i] - Pop[np.argmax(FitPop)]) / (Pop[i] - Pop[np.argmax(FitPop)] + 0.0001))
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = fitness(Off[i])
        
    for i in range(PopSize):  # Update
        if FitOff[i] < FitPop[i]:
            FitPop[i] = FitOff[i]
            Pop[i] = deepcopy(Off[i])


def RunMSICOA():
    global curIter, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curIter = 1
        Initialization()
        Best_list.append(min(FitPop))
        np.random.seed(1 + 88 * i)
        while curIter < MaxIter:
            MSICOA()
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./MSICOA_Data/CEC2017/F" + str(Func_num) + "_" + str(DimSize) + "D.csv", All_Trial_Best, delimiter=",")


def main(dim):
    global Func_num, DimSize, Pop, MaxFEs, MaxIter, SuiteName, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize / 2)
    LB = [-100] * dim
    UB = [100] * dim

    for i in range(1, 31):
        if i == 2:
            continue
        Func_num = i
        RunMSICOA()


if __name__ == "__main__":
    if os.path.exists('./MSICOA_Data/CEC2017') == False:
        os.makedirs('./MSICOA_Data/CEC2017')
    Dims = [30, 50]
    for Dim in Dims:
        main(Dim)
