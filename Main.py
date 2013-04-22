__author__ = 'dgevans'
from parameters import parameters
from parameters import DictWrap
from scipy.optimize import fmin_slsqp
import numpy as np
import bellman
import initialize
from Spline import Spline
import pickle
import timeit
import cProfile

def test():
    Para = parameters()
    Para.g = [.1, .13, .15, .18, .2]
    Para.P = np.ones((5,5))/5.0
    Para.beta = .95
    Para.sigma = 1
    Para.nx = 30
    S = Para.P.shape[0]


    ##Setup grid and initialize value function

    #Para.xgrid = np.hstack((np.linspace(Para.xmin,-1.5,Para.nx),np.linspace(-1.499,-1.25,Para.nx),np.linspace(-1.2499,Para.xmax,Para.nx)))
    #Para.xgrid = np.hstack((np.linspace(Para.xmin,-1.5,Para.nx),np.linspace(-1.49,Para.xmax,Para.nx)))
    Para.xgrid = np.linspace(Para.xmin,Para.xmax,Para.nx)
    Para.nx = Para.xgrid.shape[0]

    #Setup
    Para = initialize.setupGrid(Para)
    Vf,c_policy,xprime_policy = initialize.initializeFunctions(Para)

    #Iterate until convergence
    coef_old = np.zeros((Para.nx,S))
    for s in range(0,S):
        coef_old[:,s] = Vf[s].getCoeffs()

    Nmax = 50

    for i in range(1,Nmax):
        Vf,c_policy,xprime_policy = bellman.iterateBellman(Vf,c_policy,xprime_policy,Para)
        diff = 0
        for s_ in range(0,S):
            diff = max(diff,np.max(np.abs(coef_old[:,s_]-Vf[s_].getCoeffs())))
            coef_old[:,s_] = Vf[s_].getCoeffs()
        print diff



    #Now fit accurate Policy functions
    nx = max(min(Para.nx*10,1000),1000)
    xgrid = np.linspace(Para.xmin,Para.xmax,nx)
    #c_policy,xprime_policy = bellman.fitNewPolicies(xgrid,Vf,c_policy,xprime_policy,Para)


cProfile.run('test()','profile.stat')