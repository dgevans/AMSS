# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:49:48 2014

@author: dgevans
"""
from __future__ import division
import numpy as np
from scipy.optimize import root

def Tau(Para,mu):
    '''
    Tax rate as a function of mu
    '''
    gamma = Para.gamma
    return gamma*mu/((1.+gamma)*mu - 1.)
    
def dTau(Para,mu):
    '''
    Derivative of the tax rate
    '''
    gamma = Para.gamma
    return - gamma / ( (1.+gamma)*mu - 1. )**2
    
def d2Tau(Para,mu):
    '''
    2nd Derivative of the tax rate
    '''
    gamma = Para.gamma
    return 2*gamma*(1+gamma)*( (1+gamma)*mu - 1 )**(-3)
    
def I(Para,mu):
    '''
    Tax income as a function of the multiplier
    '''
    gamma = Para.gamma
    tau = Tau(Para,mu)
    return (1-tau)**(1/gamma) * tau
    
def dI(Para,mu):
    '''
    Derivative of tax income
    '''
    gamma = Para.gamma
    tau = Tau(Para,mu)
    return ((1-gamma)/gamma) * ( 1-(1+gamma)*tau/gamma ) *dTau(Para,mu)

def d2I(Para,mu):
    '''
    Second Derivative of tax income
    '''
    gamma = Para.gamma
    tau = Tau(Para,mu)
    return  ( 1-(1+gamma)*tau/gamma ) *d2Tau(Para,mu) \
        - dTau(Para,mu)**2 * ( (1+gamma)/gamma*(1-tau)**((1-gamma)/gamma)
        + (1-gamma)/gamma*(1-tau)**((1-2*gamma)/gamma)*( 1-(1+gamma)*tau/gamma ) ) 

def getSteadyState(Para,mubar):
    '''
    Computes the steady state for a given mubar
    '''
    g = Para.g
    P = Para.P
    Eg = P[0,:].dot(g)
    beta = Para.beta    
    
    bbar = beta/(1-beta) *(I(Para,mubar)-Eg)
    pbar = 1 - beta/bbar * (g-Eg)
    
    return bbar,pbar
    
def getSteadyStatePers(Para,mubar):
    '''
    Computes the steady state with persistance
    '''
    g = Para.g
    P = Para.P
    beta = Para.beta
    S = len(P)
    PVg = np.linalg.solve(np.eye(S)-beta*P,g)
    bbar = beta*I(Para,mubar)/(1-beta) - beta*P.dot(PVg)
    pbar = 1 - beta/bbar.reshape(-1,1) *(PVg-P.dot(PVg).reshape(-1,1))
    return bbar,pbar
    
def LinearizePersistant(Para,phat,SS):
    '''
    Linearize around SS with persistant process, with deviation phat
    '''
    mubar,bbar,pbar = SS
    bbarT = bbar.reshape(-1,1)
    P = Para.P
    S = len(P)
    beta = Para.beta
    #differentiate with respect to mu
    dI_dmu = dI(Para,mubar)
    f = lambda a: a * ( P * pbar**2 ).dot( 1/(beta*(1+a)) ) -1
    a = root(f,beta/(np.sum(P*pbar**2,1)-beta),tol=1e-15).x #guess from iid case
    db_dmu = dI_dmu *a
    db_dmuT = db_dmu.reshape(-1,1)
    
    dmuprime_dmu = a.reshape(-1,1)*pbar/(beta*(1+a))
    
    #linearize with respect to phat
    Ptild = P*dmuprime_dmu
    db_dp = np.linalg.solve(np.eye(S) - beta*Ptild, -bbar*(Ptild*phat).dot(np.ones(S)) )
    db_dpT = db_dp.reshape(-1,1)
    dmu_dp = (db_dpT + bbarT*phat/pbar - beta*db_dp/pbar) * dmuprime_dmu/db_dmuT
    
    return db_dmu,dmuprime_dmu,db_dp,dmu_dp
    
def getMeanPers(Para,phat,SS):
    '''
    Computes the mean
    '''
    P = Para.P
    i = np.argwhere( np.abs(np.linalg.eig(P)[0]-1) < 1e-15)[0] #allows for rounding
    pi = np.linalg.eig(P.T)[1][:,i].flatten()
    pi /= sum(pi)
    
    db_dmu,dmuprime_dmu,db_dp,dmu_dp = LinearizePersistant(Para,phat,SS)
    mubar = np.linalg.solve(np.diag(pi) - pi*(P*dmuprime_dmu).T,(P*dmu_dp).dot(pi))
    return pi.dot(mubar),pi.dot(db_dmu*mubar+db_dp)
    

def LinearizeMu(Para,SS):
    '''
    Linearize with respect to Mu around SS
    '''
    mubar,bbar,pbar = SS
    g = Para.g
    P = Para.P
    Eg = P[0,:].dot(g)
    beta = Para.beta   
    var_g = P[0,:].dot((g-Eg)**2)
    
    Epbar2 = 1 + beta**2 * var_g/bbar**2
    dI_dmu = dI(Para,mubar)
    db_dmu = dI_dmu / (Epbar2/beta - 1)
    dmuprime_dmu = ( db_dmu*pbar/beta ) / ( dI_dmu + db_dmu)
    return db_dmu,dmuprime_dmu
    
    
def Linearize(Para,SS):
    '''
    Linearizes with respect to mu and p
    '''
    mubar,bbar,pbar = SS
    pbarT = pbar.reshape(-1,1)
    g = Para.g
    P = Para.P
    Pi = P[:,0]
    S = len(P)
    Eg = P[0,:].dot(g)
    beta = Para.beta   
    var_g = P[0,:].dot((g-Eg)**2)
    Epbar2 = 1 + beta**2 * var_g/bbar**2
    
    dI_dmu = dI(Para,mubar)
    db_dmu,dmuprime_dmu = LinearizeMu(Para,SS)
    dmuprime_dmuT = dmuprime_dmu.reshape(-1,1)
    db_dp = P[0,:]*bbar*(1-pbar/Epbar2)
    
    dmuprime_dp = -( (bbar/beta)/(dI_dmu + db_dmu) ) * P[0,:]*pbarT*pbar/Epbar2
    dmuprime_dp += np.diag((bbar/beta)/(dI_dmu + db_dmu) * np.ones(S))
    
    #Now second order term
    d2I_dmu2 = d2I(Para,mubar)
    Epdmudmu = P[0,:].dot( pbarT*dmuprime_dmuT*dmuprime_dp )
    Epdmu2 = Pi.dot( pbar*dmuprime_dmu**2 )
    
    d2b_dmu2 = d2I_dmu2*Epdmu2/( Epbar2/beta - Epdmu2 )
    d2b_dmudp = beta/Epbar2 * ( Pi*(1-dmuprime_dmu)*(db_dmu+dI_dmu) + Epdmudmu * (d2I_dmu2+d2b_dmu2) +(Pi*Epbar2 - Pi*pbar) )
    d2muprime_dmudp = ( d2b_dmudp*pbar/beta + db_dmu*(np.eye(S)-pbarT*Pi)/beta - (d2I_dmu2+d2b_dmu2)*dmuprime_dmuT*dmuprime_dp)/(dI_dmu + db_dmu)
    
    return db_dmu,dmuprime_dmu,db_dp,dmuprime_dp,d2muprime_dmudp
    
def getErgodic(Para,SS,port):
    '''
    Computes the ergodic distribution
    '''
    db_dmu,dmuprime_dmu,db_dp,dmuprime_dp,d2muprime_dmudp = Linearize(Para,SS)
    mubar,bbar,pbar = SS
    phat = port - pbar
    Pi = Para.P[0,:]    
    
    B = dmuprime_dmu#+d2muprime_dmudp.dot(phat)
    C = dmuprime_dp.dot(phat)
    
    Bbar = Pi.dot(B)
    Cbar = Pi.dot(C)
    var_B = Pi.dot((B-Bbar)**2)
    var_C = Pi.dot((C-Cbar)**2)
    cov_BC = Pi.dot((B-Bbar)*(C-Cbar))
    
    zbar = Cbar/(1-Bbar)
    var_z = (zbar**2*var_B + 2*cov_BC*zbar+var_C)/(1-Bbar**2+var_B)
    return zbar,var_z
    
