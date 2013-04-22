__author__ = 'dgevans'
from parameters import parameters
import numpy as np
import bellman
import initialize
import cloud

Para = parameters()
Para.g = [.1, .13, .15, .18, .2]
Para.P = np.ones((5,5))/5.0
Para.beta = .95
Para.sigma = 1
Para.nx = 400
S = Para.P.shape[0]


##Setup grid and initialize value function
#Setup
Para = initialize.setupGrid(Para)
Vf,c_policy,xprime_policy = initialize.initializeFunctions(Para)

#Iterate until convergence
coef_old = np.zeros((Para.nx,S))
for s in range(0,S):
    coef_old[:,s] = Vf[s].getCoeffs()

Nmax = 100

def solveOnCloud(Vf,c_policy,xprime_policy):
    diff = [0]*Nmax
    for i in range(0,Nmax):
        Vf,c_policy,xprime_policy = bellman.iterateBellmanOnCloud(Vf,c_policy,xprime_policy,Para,nCloud=5)
        for s_ in range(0,S):
            diff[i] = max(diff[i],np.max(np.abs(coef_old[:,s_]-Vf[s_].getCoeffs())))
            coef_old[:,s_] = Vf[s_].getCoeffs()
    return Vf,c_policy,xprime_policy,diff



jid = cloud.call(solveOnCloud,Vf,c_policy,xprime_policy,_env="gspy_env",_type='m1')
Vf,c_policy,xprime_policy,diff = cloud.result(jid)

#Now fit accurate Policy functions
nx = max(min(Para.nx*10,1000),1000)
xgrid = np.linspace(Para.xmin,Para.xmax,nx)
#c_policy,xprime_policy = bellman.fitNewPolicies(xgrid,Vf,c_policy,xprime_policy,Para)
