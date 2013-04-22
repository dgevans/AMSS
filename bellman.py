__author__ = 'dgevans'
import numpy as np
from Spline import Spline
from scipy.optimize import fmin_slsqp
from parameters import DictWrap
import cloud
from multiprocessing import Pool
from functools import partial

class ValueFunctionSpline:
    def __init__(self,X,y,k,sigma,beta):
        self.sigma = sigma
        self.beta = beta
        if sigma == 1.0:
            y = np.exp((1-beta)*y)
        else:
            y = ((1-beta)*(1.0-sigma)*y)**(1.0/(1.0-sigma))
        self.f = Spline(X,y,k)

    def fit(self,X,y,k):
        if self.sigma == 1.0:
            y = np.exp((1-self.beta)*y)
        else:
            y = ((1-self.beta)*(1.0-self.sigma)*y)**(1.0/(1.0-self.sigma))
        self.f.fit(X,y,k)

    def getCoeffs(self):
        return self.f.getCoeffs()

    def __call__(self,X,d = None):
        if d==None:
            if self.sigma == 1.0:
                return np.log(self.f(X,d))/(1-self.beta)
            else:
                return (self.f(X,d)**(1.0-self.sigma))/((1.0-self.sigma)*(1-self.beta))

        if d==1:
            return self.f(X)**(-self.sigma) * self.f(X,1)/(1-self.beta)
        raise Exception('Error: d must equal None or 1')


def iterateBellman(Vf,c_policy,xprime_policy,Para):

    p = Pool(2)
    S = Para.P.shape[0]

    xbounds = []
    cbounds = []
    for s in range(0,S):
        xbounds.append((Para.xmin,Para.xmax))
        cbounds.append((0,10))
    bounds = cbounds+xbounds



    if(Para.cloud):
        jids = cloud.map(iterateOn_s,range(0,S),_env='gspy_env')
        [c_new,xprime_new,V_new] = zip(*cloud.result(jids))
    else:
        iterateOn_s_partial = partial(iterateOn_s,Vf=Vf,c_policy=c_policy,xprime_policy=xprime_policy,bounds=bounds,Para=Para)
        [c_new,xprime_new,V_new] = zip(*map(iterateOn_s_partial,range(0,S)))
        #[c_new,xprime_new,V_new] = zip(*p.map(iterateOn_s,range(0,S)))
    for s_ in range(0,S):
        Vf[s_].fit(Para.xgrid,V_new[s_][:],[2])
        for s in range(0,S):
            c_policy[(s_,s)].fit(Para.xgrid,c_new[s_][:,s],[1])
            xprime_policy[(s_,s)].fit(Para.xgrid,xprime_new[s_][:,s],[1])

    return Vf,c_policy,xprime_policy

def iterateOn_s(s_,Vf,c_policy,xprime_policy,bounds,Para):
    nsplit = 1
    if Para.cloud and nsplit >1:
        jids = cloud.map(lambda xgrid: iterateOnGrid(xgrid,s_,Vf,c_policy,xprime_policy,bounds,Para)
            ,np.split(Para.xgrid,nsplit),_env='gspy_env')
        [c,xprime,V] = zip(*cloud.result(jids))
    else:
        iterateOnGrid_partial = partial(iterateOnGrid,s_=s_,Vf=Vf,c_policy=c_policy,xprime_policy=xprime_policy,bounds=bounds,Para=Para)
        [c,xprime,V] = zip(*map(iterateOnGrid_partial
            ,np.split(Para.xgrid,nsplit))) #unzip the map results in to c,xprime,V
    return (np.vstack(c),np.vstack(xprime),np.hstack(V))


def iterateOnGrid(xgrid,s_,Vf,c_policy,xprime_policy,bounds,Para):
    S = Para.P.shape[0]
    c_new = []
    xprime_new = []
    V_new = []
    for x in xgrid:
        state = DictWrap({'x': x,'s':s_})
        z0 = np.zeros(2*S)
        for s in range(0,S):
            z0[s] = c_policy[(s_,s)](x)
            z0[S+s] = xprime_policy[(s_,s)](x)
        (policy,minusv,_,imode,smode) = fmin_slsqp(objectiveFunction,z0,f_ieqcons=impCon,bounds=bounds,fprime=objectiveFunctionJac,fprime_ieqcons=impConJac,args=(Vf,Para,state),iprint=False,full_output=True,acc=1e-10,iter=10000)
        if imode != 0:
            print smode
            exit()
        c_new.append( policy[0:S] )
        xprime_new.append(policy[S:2*S])
        V_new.append(-minusv)
    return c_new,xprime_new,V_new

def objectiveFunction(z,V,Para,state):
    u = Para.U.u
    P = Para.P

    S = P.shape[0]

    c = z[0:S]
    l = (c+Para.g)/Para.theta
    xprime = z[S:2*S]
    Vprime = np.zeros(S)

    for s in range(0,S):
        Vprime[s] = V[s](xprime[s])

    return -np.dot(P[state.s,:], u(c,l,Para) + Para.beta*Vprime )

def objectiveFunctionJac(z,V,Para,state):
    P = Para.P

    S = P.shape[0]

    c = z[0:S]
    l = (c+Para.g)/Para.theta
    xprime = z[S:2*S]
    dVprime = np.zeros(S)
    uc = Para.U.uc(c,l,Para)
    ul = Para.U.ul(c,l,Para)

    for s in range(0,S):
        dVprime[s] = V[s](xprime[s],1)

    return np.hstack((-P[state.s,:]*( uc+ul/Para.theta ),
                       -P[state.s,:]*Para.beta*dVprime))

def impCon(z,V,Para,state):
    x = state.x
    s_ = state.s
    P = Para.P
    S = Para.P.shape[0]
    beta = Para.beta

    c = z[0:S]
    l = (c+Para.g)/Para.theta
    xprime =z[S:2*S]
    uc = Para.U.uc(c,l,Para)
    ul = Para.U.ul(c,l,Para)


    Euc = np.dot(P[s_,:],uc)

    return c*uc + l*ul + xprime - x*uc/(beta*Euc)

def impConJac(z,V,Para,state):
    x = state.x
    s_ = state.s
    P = Para.P
    S = Para.P.shape[0]
    beta = Para.beta
    theta = Para.theta

    c = z[0:S]
    l = (c+Para.g)/theta
    uc = Para.U.uc(c,l,Para)
    ul = Para.U.ul(c,l,Para)
    ucc = Para.U.ucc(c,l,Para)
    ull = Para.U.ull(c,l,Para)
    Euc = np.dot(P[s_,:],uc)

    JacI = np.diag( uc+ucc*c+(ul+ull*l)/theta )
    JacXprime = np.eye(S)
    JacXterm = np.diag(-x*ucc/(beta*Euc)) + x*np.kron(uc.reshape(S,1),P[s_,:]*ucc.reshape(1,S))/(beta*Euc**2)
    return np.hstack((JacI+JacXterm,JacXprime))


def simulate(x0,T,xprime_policy,Para):
    S = Para.P.shape[0]
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

def fitNewPolicies(xgrid,Vf,c_policy,xprime_policy,Para):

    S = Para.P.shape[0]
    nx = xgrid.shape[0]
    c_new = np.zeros((nx,S,S))
    xprime_new = np.zeros((nx,S,S))
    xbounds = []
    cbounds = []
    for s in range(0,S):
        xbounds.append((Para.xmin,Para.xmax))
        cbounds.append((0,10))
    bounds = cbounds+xbounds
    for s_ in range(0,S):
        for ix in range(0,nx):
            state = DictWrap({'x': xgrid[ix],'s':s_})
            z0 = np.zeros(2*S)
            for s in range(0,S):
                z0[s] = c_policy[(s_,s)](state.x)
                z0[S+s] = xprime_policy[(s_,s)](state.x)
            (policy,_,_,imode,smode) = fmin_slsqp(objectiveFunction,z0,f_ieqcons=impCon,bounds=bounds,fprime=objectiveFunctionJac,fprime_ieqcons=impConJac,args=(Vf,Para,state),iprint=False,full_output=True,acc=1e-10,iter=1000)

            c_new[ix,s_,:] = policy[0:S]
            xprime_new[ix,s_,:] = policy[S:2*S]

    for s_ in range(0,S):
        for s in range(0,S):
            c_policy[(s_,s)].fit(xgrid,c_new[:,s_,s],[1])
            xprime_policy[(s_,s)].fit(xgrid,xprime_new[:,s_,s],[1])

    return c_policy,xprime_policy


