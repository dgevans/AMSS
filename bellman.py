__author__ = 'dgevans'
import numpy as np

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



