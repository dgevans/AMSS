# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 12:41:02 2014

@author: dgevans
"""
import numpy as np
from scipy import sparse
from scipy.optimize import root
import matplotlib.pyplot as plt

class distribution(object):
    '''
    Object holding the distribution
    '''
    def __init__(self,Para,muGrid):
        '''
        Constructs pdf
        '''
        self.mubar = [Para.domain[0][0],Para.domain[0][-1]]
        self.P = Para.P
        assert (len(self.P)  == len(muGrid))
        S = len(self.P)
        self.Ns = np.zeros(S,int)
        self.muGrid = []
        for s in range(S):
            self.muGrid.append(np.sort(muGrid[s]))
            self.Ns[s] = len(self.muGrid[s])-1
        self.N = np.sum(self.Ns)
        
        #find ergodic distribution for s
        lam,V = np.linalg.eig(self.P.T)
        i = np.argmin(np.abs(lam-1.))
        Pi = V[:,i]/np.sum(V[:,i])
        self.mupdf = []
        for s in range(S):
            self.mupdf.append(Pi[s]*np.ones(self.Ns[s])/self.Ns[s])
                
    def getCoveredGrid(self,s,muInt):
        '''
        Gets the grids covered by the interval muInt
        '''
        imin = int(np.argwhere(self.muGrid[s]<=muInt[0])[-1])
        imax = int(np.argwhere(self.muGrid[s] >= muInt[1])[0])
        return [imin,imax]
        
    def getTransitionMatrix(self,muprimef):
        '''
        Computes the transition matrix
        '''
        T = sparse.dok_matrix((self.N,self.N))
        S = len(self.P)
        ioffset = 0
        for s_ in range(S):
            for i in range(self.Ns[s_]):
                joffset = 0
                for s in range(S):
                    muInt = muprimef[s_,s]([self.muGrid[s_][i],self.muGrid[s_][i+1]])
                    muInt[0] = min(max(muInt[0],self.mubar[0]),self.mubar[1])
                    muInt[1] = min(max(muInt[1],self.mubar[0]),self.mubar[1])
                    
                    jInt = self.getCoveredGrid(s,muInt)
                    dist = muInt[1]-muInt[0]
                    if jInt[0] == jInt[1]:
                        if jInt[0] == self.Ns[s]:
                            T[ioffset+i,joffset+jInt[0]-1] = self.P[s_,s] 
                        else:
                            T[ioffset+i,joffset+jInt[0]] = self.P[s_,s] 
                    for j in range(jInt[0],jInt[1]):
                        distprime = self.muGrid[s][j+1] - self.muGrid[s][j]
                        if j == jInt[0]:
                            distprime -= muInt[0] - self.muGrid[s][j]
                        if j == jInt[1]-1:
                            distprime -= self.muGrid[s][j+1] - muInt[1]
                        if dist == 0.:
                            T[ioffset+i,joffset+j] = self.P[s_,s]
                        else:
                            T[ioffset+i,joffset+j] = self.P[s_,s]*distprime/dist
                    joffset += self.Ns[s]
            ioffset += self.Ns[s_]
        return T.tocsc()
        
    def findErgodic(self,muprimef):
        '''
        Finds the ergodic distribution
        '''
        mu = np.hstack(self.mupdf)
        G = self.getTransitionMatrix(muprimef).transpose()
        diff = 1
        i = 0
        while diff > 1e-13:
            muprime = G.dot(mu)
            diff = np.amax(abs(muprime-mu))
            mu = muprime
            i += 1
            if np.mod(i,100) == 0:
                print diff
        S = len(self.P)
        offset = 0
        self.mupdf = []
        for s in range(S):
            self.mupdf.append(mu[offset:offset+self.Ns[s]])
            offset += self.Ns[s]
        
    def createNewGrid(self,N):
        '''
        Creates a new grid based on the accuracy of the old grid
        '''
        muGrid = []
        S = len(self.P)
        self.N = 0
        for s in range(S):
            mu = self.mupdf[s]
            Ndist = np.array(N*mu/sum(mu),np.int)
            muGrid = [np.array([self.mubar[0]])]
            for i in range(len(Ndist)):
                n = max(Ndist[i],2)
                muGrid.append(np.linspace(self.muGrid[s][i],self.muGrid[s][i+1],n)[1:])
            self.muGrid[s] = np.hstack(muGrid)
            self.Ns[s] = len(self.muGrid[s])-1
            self.N += self.Ns[s]
        #find ergodic distribution for s
        lam,V = np.linalg.eig(self.P.T)
        i = np.argmin(np.abs(lam-1.))
        Pi = V[:,i]/np.sum(V[:,i])
        self.mupdf = []
        for s in range(S):
            self.mupdf.append(Pi[s]*np.ones(self.Ns[s])/self.Ns[s])
    
    def get_b_dist(self,bf):
        '''
        Computes the b distribution
        '''
        bGrid = []
        pdf = []
        S = len(self.P)
        for s in range(S):
            mugrid = (self.muGrid[s][:-1]+self.muGrid[s][1:])/2
            mupdf = self.mupdf[s]/np.abs((self.muGrid[s][1:]-self.muGrid[s][:-1]))
            bGrid.append(bf[s](mugrid))
            pdf.append(mupdf/np.abs(bf[s](mugrid)))
        return bGrid,pdf
                
        
def CompareToLinearization(Para,p):
    '''
    Compares the Linearized solution to the ergodic solution
    '''
    p = p/Para.P[:,0].dot(p)
    S = len(Para.P)
    import policy_iteration as PI
    import linearization as LI
    import LucasStockey as LS
    
    
    def f(mu):
        bbar,pbar = LI.getSteadyState(Para,mu)
        SS = mu,bbar,pbar
        return LI.getErgodic(Para,SS,Para.port)[0]
    mubar = root(f,-1.).x
    bbar,pbar = LI.getSteadyState(Para,mubar)
    SS = mubar,bbar,pbar
    phat = p-pbar
    db_dmu,dmuprime_dmu,db_dp,dmuprime_dp,d2muprime_dmudp = LI.Linearize(Para,SS)
    _,var_muhat = LI.getErgodic(Para,SS,p)
    sigma_muhat = np.sqrt(var_muhat)
    FCM = lambda state: LS.CMPolicy(state,Para)
    #solve global
    Para.port = p
    PF = PI.fitPolicyFunction(Para,FCM)
    PF = PI.solveInfiniteHorizon(Para,PF)
    cf,lf,muprimef,xif,bf = PF
    muprimefhat = {}
    for s_ in range(S):
        for s in range(S):
            muprimefhat[s_,s] = lambda muvec,shat = s :mubar+(dmuprime_dmu[shat])*(muvec-mubar)+dmuprime_dp[shat,:].dot(phat)
    #solve for ergodic
    mus = np.linspace(Para.domain[0][0],Para.domain[0][-1],200)
    dist = distribution(Para,[mus]*S)
    disthat = distribution(Para,[mus]*S)
    dist.findErgodic(muprimef)
    disthat.findErgodic(muprimefhat)
    dist.createNewGrid(2000)
    disthat.createNewGrid(2000)
    dist.findErgodic(muprimef)
    disthat.findErgodic(muprimefhat)    
    
    #make plots
    plt.figure(1)
    muvec = np.linspace(mubar-2*sigma_muhat,mubar+2*sigma_muhat)
    for s in range(S):
         plt.plot(muvec,muprimef[0,s](muvec)-muvec,'b')
         plt.plot(muvec,mubar+(dmuprime_dmu[s])*(muvec-mubar)+dmuprime_dp[s,:].dot(phat)-muvec,'--r')
    plt.figure(2)
    def bhat(muvec,d=0):
        if d== 0:
            return bbar + db_dp.dot(phat)+db_dmu*(muvec-mubar)
        elif d==1:
            return db_dmu
    bgrid,pdf = dist.get_b_dist(bf)
    bgridhat,pdfhat = disthat.get_b_dist([bhat]*S)
    plt.plot(bgrid[0],pdf[0])
    plt.plot(bgridhat[0],pdfhat[0])
    #plt.plot((dist.muGrid[0][1:]+dist.muGrid[0][:-1])/2,dist.mupdf[0]/abs(dist.muGrid[0][1:]-dist.muGrid[0][:-1]))
    #plt.plot((disthat.muGrid[0][1:]+disthat.muGrid[0][:-1])/2,disthat.mupdf[0]/abs(disthat.muGrid[0][1:]-disthat.muGrid[0][:-1]))
    

def CompareToLinearizationPers(Para,p):
    '''
    Compares the Linearized solution to the ergodic solution
    '''
    Para.port = p
    p = p/(Para.P.dot(p)).reshape(-1,1)
    S = len(Para.P)
    import policy_iteration as PI
    import linearization as LI
    import LucasStockey as LS
    
    
    def f(mubar):
         bbar,pbar = LI.getSteadyStatePers(Para,mubar)
         SS =mubar,bbar,pbar
         phat = p-pbar
         return LI.getMeanPers(Para,phat,SS)[0]
         
    mubar = root(f,-0.2).x
    bbar,pbar = LI.getSteadyStatePers(Para,mubar)
    SS = mubar,bbar,pbar
    phat = p-pbar
    db_dmu,dmuprime_dmu,db_dp,dmu_dp = LI.LinearizePersistant(Para,phat,SS)
    
    FCM = lambda state: LS.CMPolicy(state,Para)
    #solve global
    PF = PI.fitPolicyFunction(Para,FCM)
    PF = PI.solveInfiniteHorizon(Para,PF)
    cf,lf,muprimef,xif,bf = PF
    muprimefhat = {}
    for s_ in range(S):
        for s in range(S):
            muprimefhat[s_,s] = lambda muvec,shat = s,shat_=s_ :mubar+(dmuprime_dmu[shat_,shat])*(muvec-mubar)+dmu_dp[shat_,shat]
    #solve for ergodic
    mus = np.linspace(Para.domain[0][0],Para.domain[0][-1],200)
    dist = distribution(Para,[mus]*S)
    disthat = distribution(Para,[mus]*S)
    dist.findErgodic(muprimef)
    disthat.findErgodic(muprimefhat)
    dist.createNewGrid(2000)
    disthat.createNewGrid(2000)
    dist.findErgodic(muprimef)
    disthat.findErgodic(muprimefhat)    
    
    #make plots
    plt.figure(2)
    bhat = []
    for s_ in range(S):
        def bhat_temp(muvec,d=0,s_hat = s_):
            if d== 0:
                return bbar[s_hat] + db_dp[s_hat]+db_dmu[s_hat]*(muvec-mubar)
            elif d==1:
                return db_dmu[s_hat]
        bhat.append(bhat_temp)
    bgrid,pdf = dist.get_b_dist(bf)
    bgridhat,pdfhat = disthat.get_b_dist(bhat)
    plt.plot(bgrid[0],pdf[0])
    plt.plot(bgridhat[0],pdfhat[0])
    #plt.plot((dist.muGrid[0][1:]+dist.muGrid[0][:-1])/2,dist.mupdf[0]/abs(dist.muGrid[0][1:]-dist.muGrid[0][:-1]))
    #plt.plot((disthat.muGrid[0][1:]+disthat.muGrid[0][:-1])/2,disthat.mupdf[0]/abs(disthat.muGrid[0][1:]-disthat.muGrid[0][:-1]))
    
    

                        
        