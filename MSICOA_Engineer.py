import os
from copy import deepcopy
from scipy.stats import levy
from enoppy.paper_based.pdo_2022 import *


PopSize = 50
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 20
MaxFEs = 10000
MaxIter = int(MaxFEs / PopSize)

curIter = 1

Pop = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)

Func_num = 0
SuiteName = "Engineer"


# initialize the Pop randomly
def Initialization(func):
    global Pop, FitPop, DimSize
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func.evaluate(Pop[i])



def MSICOA(func):
    global Pop, FitPop, PopSize, DimSize
    Exploration(func)
    Exploitation(func)


def Exploration(func):
    global Pop, FitPop, PopSize, DimSize, LB, UB, curIter, MaxIter

    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    half = int(PopSize / 2)
    for i in range(half):
        I = np.random.randint(1, 3)
        w = (np.exp(curIter / MaxIter) - 1) / (np.e - 1)
        for j in range(DimSize):
            Off[i][j] = w * Pop[i][j] + np.random.rand() * (Pop[np.argmin(FitPop)][j] - I * Pop[i][j])  # Eq. (4)
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = func.evaluate(Off[i])

    E = 2 * np.random.uniform(-1, 1) * (1 - curIter / MaxIter)
    for i in range(half, PopSize):
        if np.random.rand() > 0.5:
            Off[i] = Pop[np.argmin(FitPop)] - E * np.abs(
                np.random.uniform(-1, 1, DimSize) * Pop[np.argmin(FitPop)] - Pop[i])
        else:
            Off[i] = Pop[np.argmin(FitPop)] - E * np.abs(
                np.random.uniform(-1, 1, DimSize) * Pop[np.argmin(FitPop)] - Pop[i]) + levy.rvs()
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = func.evaluate(Off[i])

    for i in range(PopSize):  # Update
        if FitOff[i] < FitPop[i]:
            FitPop[i] = FitOff[i]
            Pop[i] = deepcopy(Off[i])


def Exploitation(func):
    global curIter, Pop, FitPop, PopSize, DimSize, LB, UB, curIter, MaxIter
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)

    for i in range(PopSize):
        if FitPop[i] > min(FitPop):
            Off[i] = Pop[np.argmin(FitPop)] + np.random.normal(0, 1, DimSize) * np.abs(Pop[i] - Pop[np.argmin(FitPop)])
        else:
            Off[i] = Pop[i] + np.random.uniform(-1, 1) * (
                        np.abs(Pop[i] - Pop[np.argmax(FitPop)]) / (Pop[i] - Pop[np.argmax(FitPop)] + 0.0001))
        Off[i] = np.clip(Off[i], LB, UB)
        FitOff[i] = func.evaluate(Off[i])

    for i in range(PopSize):  # Update
        if FitOff[i] < FitPop[i]:
            FitPop[i] = FitOff[i]
            Pop[i] = deepcopy(Off[i])

def RunMSICOA(func):
    global curIter, TrialRuns, Pop, FitPop, DimSize
    All_Trial_Best = []
    for i in range(TrialRuns):
        Best_list = []
        curIter = 1
        Initialization(func)
        Best_list.append(min(FitPop))
        np.random.seed(1 + 88 * i)
        while curIter < MaxIter:
            MSICOA(func)
            curIter += 1
            Best_list.append(min(FitPop))
        All_Trial_Best.append(Best_list)
    np.savetxt("./MSICOA_Data/Engineer/" + str(Func_num) + ".csv", All_Trial_Best, delimiter=",")


def main():
    global Func_num, DimSize, Pop, MaxFEs, MaxIter, SuiteName, LB, UB

    Probs = [CBD(), CBHD(), CSP(), GTD(), IBD(), PLD(), PVP(), RCB(), SRD(), TBTD(), TCD(), WBP()]
    Names = ["CBDP", "CBHDP", "CSDP", "GTDP", "IBDP", "PLDP", "PVDP", "RCBDP", "SRDP", "TBTDP", "TCDP", "WBDP"]
    for i in range(len(Probs)):
        DimSize = Probs[i].n_dims
        Pop = np.zeros((PopSize, DimSize))
        MaxFEs = 11000
        MaxIter = int(MaxFEs / PopSize / 2)
        LB = Probs[i].lb
        UB = Probs[i].ub
        Func_num = Names[i]
        RunMSICOA(Probs[i])


if __name__ == "__main__":
    if os.path.exists('./MSICOA_Data/Engineer') == False:
        os.makedirs('./MSICOA_Data/Engineer')
    main()
