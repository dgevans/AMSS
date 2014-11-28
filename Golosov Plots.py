# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:10:46 2014

@author: dgevans
"""

__author__ = 'dgevans'
from parameters import parameters
import numpy as np
import LucasStockey as LS
from parameters import UQL
from parameters import UCES
import policy_iteration as PI
import ErgodicDistribution
from copy import copy
import matplotlib.pyplot as plt
import linearization as  LI
from scipy.optimize import root
import pandas as pd

Para = parameters()
Para.g = np.array([0.15,0.17,0.19])
Para.theta = 1.
Para.P = np.ones((3,3))/3
S = len(Para.P)
Para.U = UQL
Para.beta = np.array([.95])
bbar,pbar = LI.getSteadyState(Para,-.2)
phat = np.array([0.98,1.,0.98])

muGrid = np.linspace(-0.4,0.,40)
PI.setupDomain(Para,muGrid)


FCM = lambda state : LS.CMPolicy(state,Para)
PF = PI.fitPolicyFunction(Para,FCM)

b,tau = [],[]

for port,line in zip([pbar,pbar+0.25*(phat-np.mean(phat)),phat],['-k','--k',':k']):
    Para.port = port
    PF = PI.solveInfiniteHorizon(Para,PF)
    muHist,xHist,sHist,bHist,tauHist = PI.simulate(-.1,500000,PF,Para)
    b.append(pd.Series(bHist[100000:]))
    tau.append(pd.Series(tauHist[100000:]))
    plt.subplot(211)
    b[-1].plot(kind='kde',style=line,grid=False)
    plt.xlabel('Government Debt')
    plt.title('Ergodic Distribution')
    plt.legend(['Perfectly Correlated','Partially Correlated','Uncorrelated'],loc='upper left')
    plt.ylim([0,4])
    plt.subplot(212)
    tau[-1].plot(kind='kde',style=line,grid=False)
    plt.xlabel('Taxes')
    plt.ylim([0,40])
    
Para.U = UCES
muGrid = np.linspace(-0.15,0.,40)
PI.setupDomain(Para,muGrid)
Para.port = np.ones(3)
FCM = lambda state : LS.CMPolicy(state,Para)
PF = PI.fitPolicyFunction(Para,FCM)
pbar = np.ones(S)
pbar2 = np.array([ 1.02668841,  1.00017934,  0.97449364])*.9

plt.figure()
for port,line in zip([pbar,pbar+0.2*(phat-np.mean(phat)),pbar2+(phat-np.mean(phat))],['-k','--k',':k']):
    Para.port = port
    PF = PI.solveInfiniteHorizon(Para,PF)
    muHist,xHist,sHist,bHist,tauHist = PI.simulate(-.1,500000,PF,Para)
    b.append(pd.Series(bHist[100000:]))
    tau.append(pd.Series(tauHist[100000:]))
    plt.subplot(211)
    b[-1].plot(kind='kde',style=line,grid=False)
    plt.xlabel('Government Debt')
    plt.title('Ergodic Distribution')
    plt.legend(['Perfectly Correlated','Partially Correlated','Uncorrelated'],loc='upper left')
    plt.ylim([0,4])
    plt.subplot(212)
    tau[-1].plot(kind='kde',style=line,grid=False)
    plt.xlabel('Taxes')
    plt.ylim([0,40])
'''
port = LS.getPortfolio(1.1/Para.beta,Para)
port = port / Para.P[0,:].dot(port)
Para.port = copy(port)
mu = LS.solveForLSmu(1./Para.beta,0,Para)

muGrid = np.sort(np.unique(np.hstack((mu,np.linspace(-0.4,0.,40)))))
PI.setupDomain(Para,muGrid)
muvec = np.linspace(-0.4,0.,200)

FCM = lambda state: LS.CMPolicy(state,Para)

PFs = []

for p in [0.005,0.]:
    phat = np.array([0,1,0])*p
    Para.port = port+phat
    PFs.append(PI.fitPolicyFunction(Para,FCM))
    PFs[-1] = PI.solveInfiniteHorizon(Para,PFs[-1])
lindist = []
mubars = []
for p in [0.005,0.]:
    phat = np.array([0,1,0])*p
    Para.port = port+phat
    def f(mu):
        bbar,pbar = LI.getSteadyState(Para,mu)
        SS = mu,bbar,pbar
        return LI.getErgodic(Para,SS,Para.port)[0]
    mubar = root(f,mu).x
    bbar,pbar = LI.getSteadyState(Para,mubar)
    SS = mubar,bbar,pbar
    lindist.append(LI.getErgodic(Para,SS,Para.port))
    mubars.append(mubar)

simHists = []
for PF in PFs:
    np.random.seed(125232)
    simHists.append(PI.simulate(mubar,50000,PF,Para))
'''
'''
dists = []
PFs = []
for p in np.linspace(port[0],port[2],10):
    Para.port[1] = p
    PF = PI.fitPolicyFunction(Para,FCM)
    PF = PI.solveInfiniteHorizon(Para,PF)
    PFs.append(PF)
    cf,lf,muprimef,xif,xf = PF
    dists.append(ErgodicDistribution.distribution(Para,[muvec]*3))
    dists[-1].findErgodic(muprimef)
    dists[-1].createNewGrid(2000)
    dists[-1].findErgodic(muprimef)

for d,PF in zip(dists,PFs):
    cf,lf,muprimef,xif,xf = PF
    bGrid,pdf = d.get_b_dist(xf)
    plt.plot(bGrid[0],pdf[0])
'''




