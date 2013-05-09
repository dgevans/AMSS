__author__ = 'dgevans'
import numpy as np
from scipy.optimize import root

def findBondSteadyState(Para):
    def xBondResidual(mu):
        c,l = solveLSmu(mu,Para)
        uc = Para.U.uc(c,l,Para)
        I = Para.I(c,l)
        Euc = Para.P[0,:].dot(uc)
        return I[0]/(uc[0]/(Para.beta*Euc)-1)-I[1]/(uc[1]/(Para.beta*Euc)-1)
    muSS = root(xBondResidual,0).x
    cSS,lSS = solveLSmu(muSS,Para)
    return muSS,cSS,lSS


def LSResiduals(z,mu,Para):
    S = Para.P.shape[0]
    c = z[0:S]
    l = z[S:2*S]
    uc = Para.U.uc(c,l,Para)
    ucc = Para.U.ucc(c,l,Para)
    ul = Para.U.ul(c,l,Para)
    ull = Para.U.ull(c,l,Para)

    res = Para.theta*l-c-Para.g
    foc_c = uc -(uc+ucc*c)*mu
    foc_l = (ul-(ul+ull*l)*mu)/Para.theta

    return np.hstack((res,foc_c+foc_l))

def solveLSmu(mu,Para):
    S = Para.P.shape[0]
    f = lambda z: LSResiduals(z,mu,Para)

    z_mu = root(f,0.5*np.ones(2*S)).x

    return z_mu[0:S],z_mu[S:2*S]

def solveLucasStockey(x,Para):
    S = Para.P.shape[0]
    def x_mu(mu):
        c,l = solveLSmu(mu,Para)
        I = c*Para.U.uc(c,l,Para)+l*Para.U.ul(c,l,Para)
        return Para.beta*Para.P[0,:].dot(np.linalg.solve(np.eye(S)-Para.beta*Para.P,I))

    mu_SL = root(lambda mu: x_mu(mu)-x,0).x

    return solveLSmu(mu_SL,Para)

def solveLucasStockey_alt(x,Para):
    S = Para.P.shape[0]
    def LSres(z):
        beta = Para.beta
        c = z[0:S]
        mu = z[S:2*S]
        xi = z[2*S:3*S]
        l = (c+Para.g)/Para.theta
        xprime = np.zeros(S)
        for s in range(0,S):
            [cprime,lprime] = solveLSmu(mu[s],Para)
            Iprime = c*Para.U.uc(cprime,lprime,Para)+lprime*Para.U.ul(cprime,lprime,Para)
            xprime[s] = Para.beta *Para.P[0,:].dot( np.linalg.solve(np.eye(S)-Para.beta*Para.P,Iprime))
        uc = Para.U.uc(c,l,Para)
        ucc = Para.U.ucc(c,l,Para)
        ul = Para.U.ul(c,l,Para)
        ull = Para.U.ull(c,l,Para)
        res = np.zeros(3*S)
        Euc = Para.P[0,:].dot(uc)
        mu_ = Para.P[0,:].dot(uc*mu)/Euc

        res[0:S] = c*uc+l*ul+xprime-x*uc/(beta*Euc)
        res[S:2*S] = uc-mu*( c*ucc+uc )+x*ucc/(beta*Euc) * (mu-mu_)-xi
        res[2*S:3*S] = ul - mu*( l*ull+ul ) + Para.theta * xi
        return res
    z0 = [0.5]*S+[0]*2*S
    z_SL = root(LSres,z0).x

    cLS = z_SL[0:S]
    lLS = (cLS+Para.g)/Para.theta

    return cLS,lLS,Para.I(cLS,lLS)
    
def LSxmu(x,mu,Para):
    c,l = solveLSmu(mu,Para)
    I = Para.I(c,l)
    uc = Para.U.uc(c,l,Para)
    Euc= Para.P[0,:].dot(uc)
    return x*uc/(Para.beta*Euc)-I