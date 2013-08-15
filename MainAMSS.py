__author__ = 'dgevans'
from parameters import parameters
from parameters import UCES_AMSS
import numpy as np
import bellman
import initialize
from scipy.stats import norm
import sys
from mpi4py import MPI
import cPickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print rank
Para = parameters()
#Calibrate to AMSS
gmin = -2.0/0.4
gmax = 2.0/0.4
S = 10
gGauss = np.linspace(gmin,gmax,S)
Para.U = UCES_AMSS
cdf = norm.cdf(np.linspace(-2.0,2.0,S))
cdf[S-1] = 1.0
PGauss = np.ones((S,S))
PGauss[:,0] = cdf[0]
for s in range(1,S):
    PGauss[:,s] = cdf[s]-cdf[s-1]

PWar = np.array([[.9,.1],[0.3,.7]])
gWar = np.array([30,42.5])
Para.P = np.kron(PWar,PGauss)
Para.S = Para.P.shape[0]
Para.g = np.array(list(gGauss+gWar[0])+list(gGauss+gWar[1]))
#Para.P = PWar
#Para.g = gWar
#Para.S = 2
Para.beta = np.array([.95])
Para.xmax = 25.
Para.sigma_1 = 0.5
Para.sigma_2 = 2.0
Para.sigma = Para.sigma_1
Para.theta = 100.0
Para.eta = 100.0**(1-Para.sigma_2)
Para.nx = 40
Para.xmin= -90.
Para.transfers = True


S = Para.P.shape[0]


##Setup grid and initialize value function
#Setup
if rank == 0:
    print "Initializing policies"
    sys.stdout.flush()
    Para = initialize.setupGrid(Para)
    
    Para.bounds = zip([0]*S,Para.theta-Para.g)+[(Para.xmin,Para.xmax)]*S
    Vf,c_policy,xprime_policy = initialize.initializeFunctions(Para)
    policies = Vf,c_policy,xprime_policy,Para
    print "sending policies"
    sys.stdout.flush()
else:
    import time
    time.sleep(20)
    policies = []


Vf,c_policy,xprime_policy,Para = comm.bcast(policies)    

#Iterate until convergence
coef_old = np.zeros((Para.nx,S))
for s in range(0,S):
    coef_old[:,s] = Vf[s].getCoeffs()
    
Nmax = 150

diff = []
print "iterating bellman"
sys.stdout.flush()
for i in range(0,Nmax):
    Vf,c_policy,xprime_policy = bellman.iterateBellmanMPI(Vf,c_policy,xprime_policy,Para)
    diff.append(0)
    for s_ in range(0,S):
        diff[i] = max(diff[i],np.max(np.abs(coef_old[:,s_]-Vf[s_].getCoeffs())))
        coef_old[:,s_] = Vf[s_].getCoeffs()
    if rank == 0:
        print diff[i]
        sys.stdout.flush()
        

#Now fit accurate Policy functions
if rank ==0:
    fout = open('AMSSpolicyCourse.dat','w')
    cPickle.dump((Vf,c_policy,xprime_policy,Para),fout)
    fout.close()
nx = max(min(Para.nx*10,1000),1000)
xgrid = np.linspace(Para.xmin,Para.xmax,nx)
c_policy,xprime_policy = bellman.fitNewPolicies(xgrid,Vf,c_policy,xprime_policy,Para)
if rank ==0:
    fout = open('AMSSpolicy.dat','w')
    cPickle.dump((Vf,c_policy,xprime_policy,Para,xgrid),fout)
    fout.close()
    
