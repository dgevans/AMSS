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
import policy_iteration_2 as PI
import ErgodicDistribution
from copy import copy
import matplotlib.pyplot as plt
import linearization as  LI
from scipy.optimize import root

Para = parameters()
Para.g = np.array([0.15,0.17,0.19])
Para.theta = 1.
Para.sigma = 2.
Para.epsilon = 0.01
Para.alpha = 1.
#Para.P = np.ones((3,3))/3
Para.P = np.ones((3,3))/3.
p = np.array([0.95,1.0,0.95])

S = len(Para.P)
#Para.P = np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8]])
Para.U = UQL
#Para.P = np.array([[.6,.4],[.4,.6]])
Para.beta = np.array([.95])
mubar = 0.5*(0.99-0.99**2)
muGrid = np.linspace(-0.9,mubar,40)
PI.setupDomain(Para,muGrid)
FCM = lambda state: LS.CMPolicy(state,Para)
#solve global
Para.port = p
PF = PI.fitPolicyFunction(Para,FCM)
PF = PI.solveInfiniteHorizon(Para,PF)
cf,lf,muprimef,xif,bf = PF
mus = np.linspace(Para.domain[0][0],Para.domain[0][-1],200)
dist = ErgodicDistribution.distribution(Para,[mus]*S)
dist.findErgodic(muprimef)
dist.createNewGrid(2000)
dist.findErgodic(muprimef)
bgrid,pdf = dist.get_b_dist(bf)
plt.plot(bgrid[0],pdf[0])
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




