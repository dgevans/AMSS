__author__ = 'dgevans'
from scipy.optimize import root
import numpy as np
from bellman import ValueFunctionSpline
from Spline import Spline

def computeFB(Para):
    ###FIND FIRST BEST
    S = Para.P.shape[0]
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
    return cFB,xFB


def setupGrid(Para):
    cFB,xFB = computeFB(Para)
    Para.xmin = min(xFB)
    Para.xgrid = np.linspace(Para.xmin,Para.xmax,Para.nx)
    return Para

def initializeFunctions(Para):
#Initializing using deterministic stationary equilibrium
    S = Para.P.shape[0]
    cFB,_ = computeFB(Para)
    lFB = (cFB+Para.g)/Para.theta
    ucFB = Para.U.uc(cFB,lFB,Para)
    EucFB = Para.P.dot(ucFB)

    Q = np.zeros((S*S,S*S))
    for s_ in range(0,S):
        for s in range(0,S):
            Q[s_,s_*S+s] = Para.P[s_,s]
            Q[S+s_,s_*S+s] = Para.P[s_,s]
    c = np.zeros((Para.nx,S,S))
    xprime = np.zeros((Para.nx,S,S))
    V = np.zeros((Para.nx,S))

    for i in range(0,Para.nx):
        u = np.zeros((S,S))
        for s_ in range(0,S):
            x = Para.xgrid[i]
            def stationaryRoot(c):
                l = (c+Para.g)/Para.theta
                return c*Para.U.uc(c,l,Para)+l*Para.U.ul(c,l,Para)+(1.0-ucFB/(Para.beta*EucFB[s_]))*x
            c[i,s_,:] = root(stationaryRoot,cFB).x
            xprime[i,:] = x*np.ones(S)
            for s in range(0,S):
                c[i,s_,s] = min(c[i,s_,s],cFB[s])
                l = (c[i,s_,s]+Para.g[s])/Para.theta
                u[s_,s] = Para.U.u(c[i,s_,s],l,Para)
                xprime[i,s_,s] = (c[i,s_,s]*Para.U.uc(c[i,s_,s],l,Para)+l*Para.U.ul(c[i,s_,s],l,Para)+x)*Para.beta*EucFB[s_]/ucFB[s]
        for s_ in range(0,S):
            v = np.linalg.solve(np.eye(S*S) - Para.beta*Q,u.reshape(S*S))
            V[i,s_] = Para.P[s_,:].dot(v[s_*S:(s_+1)*S])

    Vf = []
    c_policy = {}
    xprime_policy = {}
    for s_ in range(0,S):
        Vf.append(ValueFunctionSpline(Para.xgrid,V[:,s_],[2],Para.sigma,Para.beta))
        for s in range(0,S):
            c_policy[(s_,s)] = Spline(Para.xgrid,c[:,s_,s],[1])
            xprime_policy[(s_,s)] = Spline(Para.xgrid,xprime[:,s_,s],[1])

    return Vf,c_policy,xprime_policy