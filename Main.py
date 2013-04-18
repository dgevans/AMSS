__author__ = 'dgevans'
from parameters import parameters
from parameters import DictWrap
from scipy.optimize import root
from scipy.optimize import fmin_slsqp
import numpy as np
from Spline import Spline
import bellman
Para = parameters()

Para.P = np.array([[.5,.5],[.5,.5]])
Para.nx = 25

S = Para.P.shape[0]

###FIND FIRST BEST
def firstBestRoot(c,Para):
    l = (c+Para.g)/Para.theta
    return Para.U.uc(c,l,Para)+Para.U.ul(c,l,Para)/Para.theta

cFB = root(firstBestRoot,0.5*np.ones(S),Para).x
lFB = (cFB+Para.g)/Para.theta
ucFB = Para.U.uc(cFB,lFB,Para)
ulFB = Para.U.ul(cFB,lFB,Para)

IFB = cFB*ucFB+lFB*ulFB

EucFB = Para.P.dot(ucFB)

xFB = np.kron(IFB,np.ones(S))/(np.kron(ucFB,1/(Para.beta*EucFB))-1)
Para.xmin = np.min(xFB)

##Setup grid and initialize value function

Para.xgrid = np.linspace(Para.xmin,Para.xmax,Para.nx)


#Initializing using deterministic stationary equilibrium
c = np.zeros((Para.nx,S))
xprime = np.zeros((Para.nx,S))
V = np.zeros((Para.nx,S))

for i in range(0,Para.nx):
    x = Para.xgrid[i]

    def stationaryRoot(c):
        l = (c+Para.g)/Para.theta
        return c*Para.U.uc(c,l,Para)+c*Para.U.ul(c,l,Para)+(1.0-1.0/Para.beta)*x

    c[i,:] = root(stationaryRoot,0.5*np.ones(S)).x
    l = (c[i,:]+Para.g)/Para.theta
    u = Para.U.u(c[i,:],l,Para)

    V[i,:] = np.linalg.solve((np.eye(S)-Para.beta*Para.P),u)
    xprime[i,:] = x*np.ones(S)

Vf = []
c_policy = {}
xprime_policy = {}
for s_ in range(0,S):
    Vf.append(Spline(Para.xgrid,V[:,s_]))
    for s in range(0,S):
        c_policy[(s_,s)] = Spline(Para.xgrid,c[:,s],[1])
        xprime_policy[(s_,s)] = Spline(Para.xgrid,xprime[:,s],[1])

#Iterate until convergence
coef_old = np.zeros((Para.nx,S))
for s in range(0,S):
    coef_old[:,s] = Vf[s].getCoeffs()

Nmax = 200
bounds = [(0,10),(0,10),(Para.xmin,Para.xmax),(Para.xmin,Para.xmax)]
for i in range(1,Nmax):
    V_new = np.zeros((Para.nx,S))
    c_new = np.zeros((Para.nx,S,S))
    xprime_new = np.zeros((Para.nx,S,S))
    for s_ in range(0,S):
        for ix in range(0,Para.nx):

            state = DictWrap({'x': Para.xgrid[ix],'s':s_})
            z0 = np.zeros(2*S)
            for s in range(0,S):
                z0[s] = c_policy[(s_,s)](state.x)
                z0[S+s] = xprime_policy[(s_,s)](state.x)
            (policy,minusv,_,imode,smode) = fmin_slsqp(bellman.objectiveFunction,z0,f_ieqcons=bellman.impCon,bounds=bounds,fprime=bellman.objectiveFunctionJac,fprime_ieqcons=bellman.impConJac,args=(Vf,Para,state),iprint=False,full_output=True,acc=1e-12,iter=1000)
            if imode != 0:
                print smode
                exit()
            c_new[ix,s_,:] = policy[0:S]
            xprime_new[ix,s_,:] = policy[S:2*S]
            V_new[ix,s_] = -minusv
    diff = 0
    for s_ in range(0,S):
        Vf[s_].fit(Para.xgrid,V_new[:,s_])
        for s in range(0,S):
            c_policy[(s_,s)].fit(Para.xgrid,c_new[:,s_,s],[1])
            xprime_policy[(s_,s)].fit(Para.xgrid,xprime_new[:,s_,s],[1])
        diff = max(diff,np.max(np.abs(coef_old[:,s_]-Vf[s_].getCoeffs())))
        coef_old[:,s_] = Vf[s_].getCoeffs()
    print diff


#Now fit accurate Policy functions
nx = Para.nx*10
xgrid = np.linspace(Para.xmin,Para.xmax,nx)
V_new = np.zeros((nx,S))
c_new = np.zeros((nx,S,S))
xprime_new = np.zeros((nx,S,S))
for s_ in range(0,S):
    for ix in range(0,nx):

        state = DictWrap({'x': xgrid[ix],'s':s_})
        z0 = np.zeros(2*S)
        for s in range(0,S):
            z0[s] = c_policy[(s_,s)](state.x)
            z0[S+s] = xprime_policy[(s_,s)](state.x)
        (policy,_,_,imode,smode) = fmin_slsqp(bellman.objectiveFunction,z0,f_ieqcons=bellman.impCon,bounds=bounds,fprime=bellman.objectiveFunctionJac,fprime_ieqcons=bellman.impConJac,args=(Vf,Para,state),iprint=False,full_output=True,acc=1e-12,iter=1000)

        c_new[ix,s_,:] = policy[0:S]
        xprime_new[ix,s_,:] = policy[S:2*S]

for s_ in range(0,S):
    for s in range(0,S):
        c_policy[(s_,s)].fit(xgrid,c_new[:,s_,s],[1])
        xprime_policy[(s_,s)].fit(xgrid,xprime_new[:,s_,s],[1])



def simulate(x0,T):
    xHist = np.zeros(T)
    sHist = np.zeros(T,dtype=np.int)
    xHist[0] = x0
    cumP = np.cumsum(Para.P,axis=1)
    for t in range(1,T):
        r = np.random.uniform()
        s_ = sHist[t-1]
        for s in range(0,S):
            if r < cumP[s_,s]:
                sHist[t] = s
                break
        xHist[t] = xprime_policy[(s_,s)](xHist[t-1])

    return xHist,sHist