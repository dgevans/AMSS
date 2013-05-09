__author__ = 'dgevans'
from parameters import parameters
import numpy as np
import bellman
import initialize
import LucasStockey as LS
import sys

def compareI(x,c_policy,Para):
    cLS,lLS,ILS = LS.solveLucasStockey_alt(x,Para)
    cSB = np.zeros(S)
    for s in range(0,S):
        cSB[s] = c_policy[(0,s)](x)
    lSB = (cSB+Para.g)/Para.theta
    ISB =  Para.I(cSB,lSB)
    return ILS-ISB

Para = parameters()
Para.g = [.1, .2]
Para.P = np.ones((2,2))/2.0
Para.beta = .95
Para.sigma = 1
Para.nx = 30
S = Para.P.shape[0]


##Setup grid and initialize value function
#Setup
Para = initialize.setupGrid(Para)
Para.bounds = [(0,10)]*S+[(Para.xmin,Para.xmax)]*S
Vf,c_policy,xprime_policy = initialize.initializeFunctions(Para)

#Iterate until convergence
coef_old = np.zeros((Para.nx,S))
for s in range(0,S):
    coef_old[:,s] = Vf[s].getCoeffs()

Nmax = 150

diff = []
for i in range(0,Nmax):
    Vf,c_policy,xprime_policy = bellman.iterateBellmanMPI(Vf,c_policy,xprime_policy,Para)
    diff.append(0)
    for s_ in range(0,S):
        diff[i] = max(diff[i],np.max(np.abs(coef_old[:,s_]-Vf[s_].getCoeffs())))
        coef_old[:,s_] = Vf[s_].getCoeffs()
    print diff[i]
    sys.stdout.flush()

#Now fit accurate Policy functions
nx = max(min(Para.nx*10,1000),1000)
xgrid = np.linspace(Para.xmin,Para.xmax,nx)
c_policy,xprime_policy = bellman.fitNewPolicies(xgrid,Vf,c_policy,xprime_policy,Para)

