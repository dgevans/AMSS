# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:56:45 2013

@author: dgevans
"""
from numpy import *
from scipy.optimize import root
from Spline import Spline
from scipy.interpolate import interp1d

k = 'linear'

class PolicyMap(object):
    '''
    A class that representing the map associated with policy iteration. Optimal
    policies will be a fixed point of this map
    '''
    def __init__(self,Para,mu_bounds):
        '''
        Initializes with Para class and bounds on mu
        '''
        self.Para = Para
        self.mubar_l,self.mubar_h = mu_bounds
        
    def __call__(self,PolicyFunctions):
        '''
        Creates new map given policy functions
        '''
        self.cf,self.lf,self.muprimef,self.xif,self.xf = PolicyFunctions
        return self.policies
    
    def policies(self,state,z0=None,method = 'hybr'):
        '''
        Computes the optimal policies given coninuation policies
        '''
        mu_tilde,s_ = state
        S = len(self.Para.P)
        #construct initial guess from policies
        if z0 == None:
            z0 = zeros(4*S+1)
            for s in range(S):
                z0[s:4*S:S] = [self.cf[s_,s](mu_tilde),self.lf[s_,s](mu_tilde),
                   self.muprimef[s_,s](mu_tilde),self.xif[s_,s](mu_tilde)]
            z0[4*S] = self.xf[s_](mu_tilde)
        #find solution to system of equations
        f = lambda z: self.policyResiduals(state,z)
        res = root(f,z0,method=method)
        if res.success:
            return res.x
        return None
    
    def policyResiduals(self,state,z):
        '''
        Residuals of the first order conditions
        '''
        mu_tilde,s_ = state
        S = len(self.Para.P)
        Para = self.Para
        p = Para.port
        P = Para.P[s_,:]
        beta= Para.beta
        theta = Para.theta
        g = Para.g
        
        
        c,l,muprime,xi,x = z[:S],z[S:2*S],z[2*S:3*S],z[3*S:4*S],z[4*S]
        Uc = Para.U.uc(c,l,Para)
        Ucc = Para.U.ucc(c,l,Para)
        Ul = Para.U.ul(c,l,Para)
        Ull = Para.U.ull(c,l,Para)
        EpUc = P.dot(p*Uc)
        EpUcMu = P.dot(p*Uc*muprime)
        
        #construct mu_tilde from mu
        mu_tilde_prime = zeros(S)
        xprime = zeros(S)
        for s in range(S):
            mu_tilde_prime[s] = min(max(muprime[s],self.mubar_l),self.mubar_h)
            xprime[s] = self.xf[s](mu_tilde_prime[s])
        
        #Now do constraints
        res = zeros(4*S+1)
        res[:S] = x*p*Uc/( beta*EpUc ) - Uc*c - Ul*l - xprime
        res[S:2*S] = theta*l - c - g
        res[2*S:3*S] = Uc - muprime*( Ucc*c + Uc ) + ( x*Ucc*p )/( beta*EpUc ) * (muprime-mu_tilde) - xi
        res[3*S:4*S] = Ul - muprime*( Ull*l + Ul ) + theta*xi
        res[4*S] = mu_tilde - EpUcMu/EpUc
        return res
        
        
class time0Problem(object):
    '''
    Solves the time 0 problem
    '''
    def __init__(self,Para,PF,mu_bounds):
        '''
        Stores Para and policy rules
        '''
        self.Para = Para
        self.PF = PF
        self.mubar_l,self.mubar_h = mu_bounds
        
    def __residuals(self,state,z):
        '''
        Residuals given the state and policies z
        '''
        cf,lf,muprimef,xif,xf = self.PF
        b0,s0 = state
        c,l,mu,xi = z
        Para = self.Para
        
        theta = Para.theta
        g = Para.g
        
        mu = min(max(mu,self.mubar_l),self.mubar_h)        
        
        x = xf[s0](mu)        
        
        Uc = Para.U.uc(c,l,Para)
        Ucc = Para.U.ucc(c,l,Para)
        Ul = Para.U.ul(c,l,Para)
        Ull = Para.U.ull(c,l,Para)
        
        res = zeros(4)
        res[0] = Uc*b0 - Uc*c - Ul*l - x
        res[1] = (Para.theta*l - c - g)[s0]
        res[2] = Uc - mu*(Ucc*(c-b0) + Uc) - xi
        res[3] = (Ul - mu*(Ull*l + Ul) + theta*xi + 0*g)[s0]
        return res
        
    def __call__(self,state):
        '''
        Finds the optimal policy rules at b0,s
        '''
        res = root(lambda z: self.__residuals(state,z),array([1.,0.5,-0.02,1.]))
        if not res.success:
            raise "could not find root"
        else:
            return res.x
        
def setupDomain(Para,mugrid):
    '''
    Creates the domain on which the functions are approximated
    '''
    S = len(Para.P)
    Para.domain = [mugrid]*S
    
def fitPolicyFunction(Para,PolicyF):
    '''
    Fits the policy functions given function PolicyF
    '''
    cf,lf,muprimef,xif = {},{},{},{}
    xf = []
    S = len(Para.P)
    for s_ in range(S):
        Policies = vstack(map(lambda mu:PolicyF((mu,s_)),Para.domain[s_]))
        mugrid = Para.domain[s_]
        for s in range(S):
            cf[s_,s] = Spline(mugrid,Policies[:,s],[2])
            lf[s_,s] = Spline(mugrid,Policies[:,s+S],[2])
            muprimef[s_,s] = Spline(mugrid,Policies[:,s+2*S],[2])
            xif[s_,s] = Spline(mugrid,Policies[:,s+3*S],[2])
        xf.append(Spline(mugrid,Policies[:,4*S],[1]))
    return cf,lf,muprimef,xif,xf
    
def fitFunction(F,Para):
    '''
    Fits the policy functions given function PolicyF
    '''
    Fhat = []
    S = len(Para.P)
    for s_ in range(S):
        mugrid = Para.domain[s_]
        Fs = hstack(map(lambda mu: F(mu,s_),mugrid))
        Fhat.append(Spline(mugrid,Fs,[1]))
    return Fhat
    
    
def solveInfiniteHorizon(Para,PolicyFunctions):
    '''
    Solves the infinite horizon problem
    '''
    S = len(Para.P)
    T = PolicyMap(Para,[min(Para.domain[0]),max(Para.domain[0])])
    diff  = 1.
    while diff > 1e-6:
        PolicyFunctionsNew = fitPolicyFunction(Para,T(PolicyFunctions))
        xf = PolicyFunctions[4]
        xfnew = PolicyFunctionsNew[4]
        diff = 0.
        for s_ in range(S):
            mugrid = Para.domain[s_]
            diff = max(diff,amax(abs(xf[s_](mugrid)-xfnew[s_](mugrid))))
        print diff
        PolicyFunctions = PolicyFunctionsNew
    return PolicyFunctions
    
def simulate(mu0,T,PF,Para):
    """
    Simulates starting from x0 given xprime_policy for T periods.  Returns sequence of xprimes and shocks
    """
    cf,lf,muprimef,xif,xf = PF
    S = Para.P.shape[0]
    muHist = zeros(T)
    xHist = zeros(T)
    bHist = zeros(T)
    tauHist = zeros(T)
    sHist = zeros(T,dtype=int)
    muHist[0] = mu0
    c = cf[0,0](mu0)
    l = lf[0,0](mu0)
    tauHist[0] = 1 + Para.U.ul(c,l,Para)/(Para.theta*Para.U.uc(c,l,Para))
    xHist[0] = xf[sHist[0]](muHist[0])
    bHist[0] = xHist[0]/Para.U.uc(c,l,Para)
    
    cumP = cumsum(Para.P,axis=1)
    mu_l = amin(Para.domain[0])
    mu_h = amax(Para.domain[0])
    for t in range(1,T):
        r = random.uniform()
        s_ = sHist[t-1]
        for s in range(0,S):
            if r < cumP[s_,s]:
                sHist[t] = s
                break
        muprime = muprimef[(s_,s)](muHist[t-1])
        c = cf[s_,s](muHist[t-1])
        l = lf[s_,s](muHist[t-1])
        tauHist[t] = 1 + Para.U.ul(c,l,Para)/(Para.theta*Para.U.uc(c,l,Para))
        muHist[t] = min(max(muprime,mu_l),mu_h)
        xHist[t] = xf[s](muHist[t])
        bHist[t] = xHist[t]/Para.U.uc(c,l,Para)
    return muHist,xHist,sHist,bHist,tauHist
    
def conditionalExpectations(muprimef,T,Para):
    '''
    Computes a list of conditional expectations T periods ahead
    '''
    P = Para.P
    S = len(P)
    Emu = [fitFunction(lambda mu,s:mu,Para)]
    mubar_l = Para.domain[0][0]
    mubar_h = Para.domain[0][-1]
    for t in range(1,T+1):
        Emu_ = Emu[-1]
        def Emu_new(mu_,s_):
            Emu_vec = zeros(S)
            for s in range(S):
                muprime = min(max(muprimef[s_,s](mu_),mubar_l),mubar_h)
                #muprime = muprimef[s_,s](mu_)
                Emu_vec[s] = Emu_[s](muprime)
            return P[s_,:].dot(Emu_vec)
        Emu.append(fitFunction(Emu_new,Para))
    return Emu
    
def conditionalExpectations_x(muprimef,xf,T,Para):
    '''
    Computes a list of conditional expectations T periods ahead
    '''
    P = Para.P
    S = len(P)
    Ex = [fitFunction(lambda mu,s:xf[s](mu),Para)]
    mubar_l = Para.domain[0][0]
    mubar_h = Para.domain[0][-1]
    for t in range(1,T+1):
        Ex_ = Ex[-1]
        def Ex_new(mu_,s_):
            Ex_vec = zeros(S)
            for s in range(S):
                muprime = min(max(muprimef[s_,s](mu_),mubar_l),mubar_h)
                Ex_vec[s] = Ex_[s](muprime)
            return P[s_,:].dot(Ex_vec)
        Ex.append(fitFunction(Ex_new,Para))
    return Ex
    
def computeErgodicDistribution(muprimef,Para, Ngrid = 1000):
    '''
    Compute the ergodic distributon for mu
    '''
    mubar_l = Para.domain[0][0]
    mubar_h = Para.domain[0][-1]
    muGrid = linspace(mubar_l,mubar_h,Ngrid)
    
    
            
            