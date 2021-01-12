#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cupy as cp
import numpy as np
from cupy import random
import scipy as sc
from scipy import linalg
import matplotlib as plt
import numpy as np
from numpy import genfromtxt
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import invgamma
import matplotlib.pyplot as plt
from sympy import Matrix
import seaborn as sns
from datetime import datetime
import math
import time
ordqz = sc.linalg.ordqz
svd = sc.linalg.svd
J = 1
y_raw = genfromtxt('C:\Research\Y_data.csv', delimiter=',')
y_data = cp.array(np.array(cp.reshape(cp.repeat(y_raw,J),[260,3,J])))
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import invgamma


# In[2]:


def Generate_S1(J,T):
    S = cp.random.normal(0,.01,J*5);
    S  = S.reshape(5,1,J);
    EPS = cp.random.normal(0,.01,3*J*(T+1));
    EPS = EPS.reshape(3,T+1,J);
    Lx = cp.vstack((cp.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),cp.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),cp.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])))
    Li =  np.vstack((cp.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),cp.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),cp.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])))
    Lpi = cp.vstack((cp.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),cp.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),cp.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])))
    R =   cp.reshape(np.vstack((cp.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),cp.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),cp.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]))),[3,18])
    U =   cp.reshape(np.vstack((cp.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),cp.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),cp.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))),[3,18]) 
    I =   cp.reshape(cp.vstack((cp.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),cp.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),cp.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))),[3,18]) 
    I =   cp.repeat(I[:, :, cp.newaxis], J, axis=2)
    return EPS,S,Lx,Lpi,Li,I,R,U


# In[3]:


def param_checks(para,RC):
    det = RC[0]
    #det = 1
    check =0
    if det==1:
             #p_r         Beta         g              psi            h               gamma         rho           p_u               Theta           alpha            Phi_pi       Phi_x      sig_e          sig_r             sig_u     
        if para[0]<1 and para[4]<1 and para[7]<1 and para[10]>1 and para[11]<1 and para[12]<1 and para[13]<1 and para[14]<1 and para[6]<.8 and .20<para[9]<.3  and para[2]<4 and para[3]<2: # and para[15]<5 and para[8]<5 and para[16]<5: 
            if para[0]>0 and  para[1] >0 and para[2] >0 and para[3] >0 and para[4] >0 and para[5] >0 and para[6] >0 and para[7] >0 and para[8] >0 and para[9] >0 and para[10] >0 and para[11] >0 and para[12] >0 and para[13] >0 and para[14] >0 and para[15] >0 and para[16] >0: 
                check =1
    if RC[1] == 0:
            check =0
    if RC[0] == -2:   
            check =0
    if RC[1] == -2:
            check =0
    return check


# In[4]:


def NK_priors(para):
    #([0.8,0.8,1.6000,.8000,.99,1.4,0.016,0.0183,1.6700,0.3,1.01,.117,.032,.9,.85,1.15,.993,1.17])
    para = cp.asnumpy(para)
    shape = np.zeros(11)
    scale = np.zeros(11)
    c = np.zeros(2)
    d = np.zeros(2)
    mu = np.array([.125,.015,0.031,.2,.3,1.2,.99,.5,1.6,.2,.15])
    sig = np.array([.09,.011,.022,.1,.2,.15,.001,.2,.15,.15,.1])
    for i in range(11):
        shape[i] = (mu[i]/sig[i])**2 
        scale[i] = sig[i]**2/mu[i]
        
    shape[3] = (mu[3]/sig[3])**2 + 2
    scale[3] = (shape[3]-1)*mu[3]
    

    c[0] = (mu[6]**2)*((1-mu[6])/((sig[6])**2)-(1/mu[6]))
    d[0] = c[0]*((1/mu[6])-1)
    c[1] = (mu[7]**2)*((1-mu[7])/((sig[7])**2)-(1/mu[7]))
    d[1] = c[1]*((1/mu[7])-1)
    
    P1 = gamma.logpdf((1/para[1]),a=shape[0],scale = scale[0])
    P6 = gamma.logpdf(para[6],a=shape[1],scale = scale[1])
    P5 = gamma.logpdf(para[5],a=shape[4],scale = scale[4])
    P10 = gamma.logpdf(para[10],a=shape[5],scale = scale[5])
    P2 =   gamma.logpdf(para[2],a=shape[8],scale = scale[8])
    P3 =  gamma.logpdf(para[3],a=shape[9],scale = scale[9])

    P7 =  gamma.logpdf(para[7],a=shape[2],scale = scale[2])

    P9 = norm.logpdf(para[9],.3,.05**2)
    
   # P4 = gamma.logpdf(para[4],a=shape[6],scale=.001)
    
    
   # muU = .99
   # sigU = .01
    
   # alphaaz = ((1-muU)/sigU** - 1/muU)*muU**2
    
   # Betaaz = alphaaz*(1/muU-1)
    
   # P4 = beta.logpdf(para[4],alphaaz,Betaaz)
    
    
    P13 = gamma.logpdf(para[13],a=shape[7],scale=scale[7])

    
    shape[3] = (mu[3]/sig[3])**2 -2
    scale[3] = (shape[3]-1)*(mu[3])
    
    shape[10] = (mu[10]/sig[10])**2 -2
    scale[10] = (shape[10]-1)*(mu[10])
    
    
    P8 = invgamma.logpdf(para[8],a=shape[3],scale=scale[3])
    P15= invgamma.logpdf(para[15],a=shape[3],scale=scale[3])
    P16= invgamma.logpdf(para[16],a=shape[3],scale=scale[3])
    #np.log(np.exp(np.log(2) - np.log(math.gamma(.5*b)) + .5*b*np.log(b*(a**2)*.5) - ((b+1)*.5)*np.log(x**2) - (b*a**2)*(x**-2)*.5))

    prioc = P1+P6+P2+P3+P9+P5+P10+P8+P15+P16+P13+P7 #+P4
    return prioc


# In[5]:


def gensys(G0, G1, PSI, PI, DIV=1 + 1e-8,
           REALSMALL=1e-6,
           return_everything=False):
    """
    Solves a Linear Rational Expectations model via GENSYS.

    Γ₀xₜ = Γ₁xₜ₋₁ + Ψεₜ + Πηₜ

    Returns
    -------

    RC : 2d int array
         [ 1,  1] = existence and uniqueness
         [ 1,  0] = existence, not uniqueness
         [-2, -2] = coincicdent zeros

    Notes
    -----
    The solution method is detailed in ...

    """
    n, pin = G0.shape[0], PI.shape[1]

    with np.errstate(invalid='ignore', divide='ignore'):
        AA, BB, alpha, beta, Q, Z = ordqz(G0, G1, sort='ouc', output='complex')
        zxz = ((np.abs(beta) < REALSMALL) * (np.abs(alpha) < REALSMALL)).any()

        x = alpha / beta
        nunstab = (x * x.conjugate() < 1.0).sum()

    if zxz:
        RC = [-2, -2]
        #print("Coincident zeros")
        return

    nstab = n - nunstab

    Q = Q.T.conjugate()
    Qstab, Qunstab = Q[:nstab, :], Q[nstab:, :]

    etawt = Qunstab.dot(PI)
    ueta, deta, veta = np.linalg.svd(etawt, full_matrices=False)

    bigev = deta > REALSMALL
    deta = deta[bigev]
    ueta = ueta[:, bigev]
    veta = veta[bigev, :].conjugate().T

    RC = np.array([0, 0])
    RC[0] = len(bigev) >= nunstab

    if RC[0] == 0:
            RC[1] = 0
        #warnings.warn(
         #   f"{nunstab} unstable roots, but only {len(bigev)} "
         #   " RE errors! No solution.")

    if nunstab == n:
        raise NotImplementedError("case nunstab == n, not implemented")
    else:
        etawt1 = Qstab.dot(PI)
        ueta1, deta1, veta1 = svd(etawt1, full_matrices=False)
        bigev = deta1 > REALSMALL
        deta1 = deta1[bigev]
        ueta1 = ueta1[:, bigev]
        veta1 = veta1[bigev, :].conjugate().T

    if veta1.size == 0:
        unique = 1
    else:
        loose = veta1 - veta.dot(veta.conjugate().T).dot(veta1)
        ul, dl, vl = np.linalg.svd(loose)
        unique = (dl < REALSMALL).all()

        # existence for general epsilon[t]
        AA22 = AA[-nunstab:, :][:, -nunstab:]
        BB22 = BB[-nunstab:, :][:, -nunstab:]
        M = np.linalg.inv(BB22).dot(AA22)

    if unique:
        RC[1] = 1
    else:
        pass
        # print("Indeterminancy")

    deta = np.diag(1.0 / deta)
    deta1 = np.diag(deta1)

    etawt_inverseT = ueta.dot((veta.dot(deta)).conjugate().T)
    etatw1_T = veta1.dot(deta1).dot(ueta1.conjugate().T)
    tmat = np.c_[np.eye(nstab), -(etawt_inverseT.dot(etatw1_T)).conjugate().T]

    G0 = np.r_[tmat.dot(AA), np.c_[np.zeros(
        (nunstab, nstab)), np.eye(nunstab)]]
    G1 = np.r_[tmat.dot(BB), np.zeros((nunstab, n))]

    G0i = np.linalg.inv(G0)
    G1 = G0i.dot(G1)

    impact = G0i.dot(np.r_[tmat.dot(Q).dot(
        PSI), np.zeros((nunstab, PSI.shape[1]))])

    G1 = np.real(Z.dot(G1).dot(Z.conjugate().T))
    impact = np.real(Z.dot(impact))

    if return_everything:
        GZ = -np.linalg.inv(BB22).dot(Qunstab).dot(PSI)
        GY = Z.dot(G0i[:, -nunstab:])

        return G1, impact, M, GZ, GY, RC

    else:
        return G1, impact, RC


# In[6]:


def REE_gen1(para):
    para = cp.asnumpy(para)
    p_r= float(para[0])
    sigma=float(para[1])
    phi_pi = float(para[2])
    phi_x=float(para[3])
    Beta=.99
    nu=float(para[5])
    theta=float(para[6])
    g=float(para[7])
    sig_r = float(para[8])
    alpha =float(para[9])
    psi = float(para[10])
    h = float(para[11])
    gamma = float(para[12])
    rho = float(para[13])
    p_u = float(para[14])
    sig_e = float(para[15])
    sig_u = float(para[16])
    alpha_p = (alpha/(1-alpha));
    k = ((1-Beta*theta)*(1-theta))/((1+alpha_p*psi)*theta*(1+gamma*Beta*theta));
    c1 = (sigma/((1-h)*(1-h*Beta)));
    c2 = (nu/(alpha + nu));
    w1 = (1+(h**2)*Beta + h*Beta)*((1+h+(h**2)*Beta)**-1)
    w2 =((1-h)*(1-h*Beta))*((sigma*(1+h+h**2*(Beta)))**-1)
    w3 = (-h*Beta/(1+h+(h**2)*Beta));
    w4 =h*((1+h+(h**2)*Beta)**-1)
    n1 = k*c2*c1+k*(h**2)*Beta*c1*c2-k*alpha_p
    n2 = -k*(c2)*(c1)*(h);
    n3 = -k*h*Beta*c1*c2;
    n4 =  Beta*((1+gamma*Beta*theta)**-1);
    n5 = gamma*((1+gamma*Beta*theta)**-1) + (-gamma*psi*alpha_p*k);  
    x_t   = 0;
    pi_t   = 1;
    i_t   = 2;
    r_t  = 3;
    u_t   = 4;
    Ex_t = 5;
    Epi_t = 6;
    Ei_t = 7;
    Ex_t2 = 8;
    Epi_t2 = 9;
    Ei_t2 = 10;
    
    ex_sh  = 0;
    epi_sh  =1;
    ei_sh  = 2;
    ex2_sh  =3 ;
    epi2_sh  =4 ;
    ei2_sh  =5 ;
    
    r_sh = 0;
    pi_sh = 1;
    i_sh = 2;
    
    neq  = 11;
    neta = 6;
    neps = 3;
    GAM0 = np.zeros([neq,neq]);
    GAM1 = np.zeros([neq,neq]);
    C = np.zeros([neq,1]);        
    PSI = np.zeros([neq,neps]);
    PPI = np.zeros([neq,neta]);
    eq_1 = 0
    eq_2    = 1;  
    eq_3    = 2;  
    eq_4    = 3;  
    eq_5   = 4;  
    eq_6    = 5;  
    eq_7    = 6; 
    eq_8    = 7;
    eq_9    = 8;
    eq_10    = 9;
    eq_11    = 10;
#x_t
    GAM0[eq_1,x_t]   =  1;
    GAM0[eq_1,Ex_t]   =  -w1;
    GAM0[eq_1,Epi_t]   =  -w2;
    GAM0[eq_1,i_t]   =  w2;
    GAM0[eq_1,r_t]   =  w2;
    GAM0[eq_1,Ex_t2]   =  -w3;
    GAM1[eq_1,x_t] = w4;
#pi_t
    GAM0[eq_2,pi_t]   =  1;
    GAM0[eq_2,x_t]   = -n1;
    GAM1[eq_2,x_t]   = n2;
    GAM0[eq_2,Ex_t]   = -n3;
    GAM0[eq_2,Epi_t]   = -n4;
    GAM1[eq_2,pi_t]   =  n5;
    GAM1[eq_2,u_t]  =  1;
#i_t
    GAM0[eq_3,x_t]   = -(1-rho)*phi_x;
    GAM0[eq_3,pi_t]  = -(1-rho)*phi_pi;
    GAM0[eq_3,i_t]  =1;
    GAM1[eq_3,i_t]  = rho;
    PSI[eq_3,i_sh] = 1;

#r_t
    GAM0[eq_4,r_t]   = 1;
    GAM1[eq_4,r_t] = p_r;
    PSI[eq_4,r_sh] = 1;
#u_t
    GAM0[eq_5,u_t]   = 1;
    GAM1[eq_5,u_t] = p_u;
    PSI[eq_5,pi_sh] = 1;
#Epi_t

    GAM0[eq_6,pi_t]   = 1;
    GAM1[eq_6, Epi_t] =1;
    PPI[eq_6, epi_sh] = 1;
#Ex_t

    GAM0[eq_7,x_t]   = 1;
    GAM1[eq_7, Ex_t] =1;
    PPI[eq_7, ex_sh] = 1;
#Ex_t2

    GAM0[eq_8,Ex_t]   = 1;
    GAM1[eq_8, Ex_t2] =1;
    PPI[eq_8, ex2_sh] = 1;
#Ei_t

    GAM0[eq_9,i_t]   = 1;
    GAM1[eq_9, Ei_t] =1;
    PPI[eq_9, ei_sh] = 1;   
    
#Ei_t2

    GAM0[eq_10,Ei_t]   = 1;
    GAM1[eq_10, Ei_t2] =1;
    PPI[eq_10, ei2_sh] = 1;    
#Epi_t2

    GAM0[eq_11,Epi_t]   = 1;
    GAM1[eq_11, Epi_t2] =1;
    PPI[eq_11, epi2_sh] = 1;    
    
    
    
    G1,impact,RC  = gensys(GAM0, GAM1, PSI, PPI, DIV=1 + 1e-8,REALSMALL=1e-6,return_everything=False)
    
    # GAM0*x(t) = A*x(t-1) + B*Ex(t+1) + C*Ex(t+2) + E*u(t) + D*eps(t)

    GAM0 = cp.array([[1,0,w2],[-n1,1,0],[-(1-rho)*phi_x,-(1-rho)*phi_pi,1]])
    GAM0inv = cp.linalg.inv(GAM0)
    A  = cp.matmul(GAM0inv,cp.array([[w4,0,0],[n2,n5,0],[0,0,rho]]))
    B = cp.matmul(GAM0inv,cp.array([[w1,w2,0],[n3,n4,0],[0,0,0]]))
    C = cp.matmul(GAM0inv,cp.array([[w3,0,0],[0,0,0],[0,0,0]]))
    D = cp.matmul(GAM0inv,cp.array(([0,0,0],[0,0,0],[0,0,1])))  
    E = cp.matmul(GAM0inv,cp.array([[-w2,0],[0,1],[0,0]]))
    R = cp.array([[para[0],0],[0,para[14]]]) 
    # x(t) = A*x(t-1) + B*Ex(t+1) + C*Ex(t+2) + E*u(t) + DD*eps(t)
    V_s = cp.array([[para[8]**2,0,0],[0,para[16]**2,0],[0,0,para[15]**2]])
    return G1,impact,RC,A,B,C,E,D,R,V_s


# In[7]:


def check_eigenvalues(PHI):
    a,b,c,d,e,f,g,h,i = cp.reshape(PHI[3:12,:],[9,1,J])
    a0 = -a*e*i +a*f*h+b*d*i-b*f*g-c*d*h+c*e*g
    a1 =a*e-b*d+a*i-c*g+e*i-f*h
    a2 = -a-e-i
    #1
    bb1 =(abs(a2+a0)<=1+a1).astype(int)
    #2
    bb2 =(abs(a2-3*a0)<=(3-a1)).astype(int)
    #3 
    bb3 = ((a0**2+a1-a0*a2)<=1).astype(int)
    return cp.reshape(((bb1+bb2+bb3 !=3).astype(int)),[J])


# In[8]:


def Generate_S(J,T):
    S = cp.random.standard_normal([5,J]);
    EPS = cp.random.normal(0,.01,3*J*(T+1));
    EPS = EPS.reshape(3,T+1,J);
    S = cp.array(S,cp.float32)
    EPS = cp.array(EPS,cp.float32)
    return EPS,S


# In[9]:


def Generate_EPS(J,T):
    EPS = cp.random.normal(0,1,3*J*(T+1))
    EPS = cp.array(cp.reshape(EPS,[3,T+1,J]),dtype=cp.float32)
    return EPS


# In[10]:


def Generate_PHI(J,T3):
    PHI = cp.reshape(cp.repeat(T3,[J]),[18,J]) 
    PHI = cp.array(PHI,dtype = cp.float32)
    return PHI


# In[11]:


def PFLIKE(para,EPS,S,RR,randphi):   
    #K = cp.zeros([5,T])
    G1,impact,RC,A,B,C,E,D,R,V_s = REE_gen1(para)
    
    PP = cp.array(cp.zeros([18,18,J]),cp.float32)
    
    T3 = cp.vstack((cp.vstack(((cp.zeros([3,1]),cp.reshape(G1[0:3,0:3],[9,1])))),cp.reshape(G1[0:3,3:5],[6,1])))
    T3 = cp.array(T3,dtype = cp.float32)
    T3 = cp.vstack((cp.vstack(((cp.zeros([3,1]),cp.reshape(G1[0:3,0:3],[9,1])))),cp.reshape(G1[0:3,3:5],[6,1])))
    PHI = .1*randphi
    PHI = T3+PHI
    PHI = cp.array(PHI,dtype=cp.float32)
    Xb = S[0]
    error = cp.array(cp.zeros(T),cp.float32)
    LL = cp.array(cp.zeros([T]))
    EX = cp.array(cp.zeros([3,J]),cp.float32)
    EX1 = EX
    EX2 =EX
    PP = cp.zeros([18,18,J])
    M = cp.zeros([T])
    #X = cp.zeros([J,T])
    PHI_old = PHI
    for t in range(T):   
        S[3] = para[0]*S[3]+(para[8])*EPS[0,t,:]
        S[4] = para[14]*S[4]+(para[16])*EPS[1,t,:]
        
        EX[0] = PHI[0]+PHI[3]*S[0]+PHI[4]*S[1]+PHI[5]*S[2]+PHI[12]*S[3]+PHI[13]*S[4]
        EX[1] = PHI[1]+PHI[6]*S[0]+PHI[7]*S[1]+PHI[8]*S[2]+PHI[14]*S[3]+PHI[15]*S[4]
        EX[2] = PHI[2]+PHI[9]*S[0]+PHI[10]*S[1]+PHI[11]*S[2]+PHI[16]*S[3]+PHI[17]*S[4]
        

        
        
        EX1[0] = PHI[0]+PHI[3]*EX[0]+PHI[4]*EX[1]+PHI[5]*EX[2]+PHI[12]*para[0]*S[3]+PHI[13]*para[14]*S[4]
        EX1[1] = PHI[1]+PHI[6]*EX[0]+PHI[7]*EX[1]+PHI[8]*EX[2]+PHI[14]*para[0]*S[3]+PHI[15]*para[14]*S[4]
        EX1[2] = PHI[2]+PHI[9]*EX[0]+PHI[10]*EX[1]+PHI[11]*EX[2]+PHI[16]*para[0]*S[3]+PHI[17]*para[14]*S[4]
        

        
        EX2[0] = PHI[0]+PHI[3]*EX1[0]+PHI[4]*EX1[1]+PHI[5]*EX1[2]+PHI[12]*(para[0]**2)*S[3]+PHI[13]*(para[14]**2)*S[4]
        EX2[1] = PHI[1]+PHI[6]*EX1[0]+PHI[7]*EX1[1]+PHI[8]*EX1[2]+PHI[14]*(para[0]**2)*S[3]+PHI[15]*(para[14]**2)*S[4]
        EX2[2] = PHI[2]+PHI[9]*EX1[0]+PHI[10]*EX1[1]+PHI[11]*EX1[2]+PHI[16]*(para[0]**2)*S[3]+PHI[17]*(para[14]**2)*S[4]
        
        
        
        S[0] = A[0,0]*S[0]+A[0,1]*S[1]+A[0,2]*S[2]+B[0,0]*EX1[0]+B[0,1]*EX1[1]+B[0,2]*EX1[2]+C[0,0]*EX1[0]+C[0,1]*EX1[1]+C[0,2]*EX1[2]+D[0,2]*(para[15])*EPS[2,t,:]
        S[1] = A[1,0]*S[0]+A[1,1]*S[1]+A[1,2]*S[2]+B[1,0]*EX1[0]+B[1,1]*EX1[1]+B[1,2]*EX1[2]+C[1,0]*EX2[0]+C[1,1]*EX2[1]+C[1,2]*EX2[2]+D[1,2]*(para[15])*EPS[2,t,:]
        S[2] = A[2,0]*S[0]+A[2,1]*S[1]+A[2,2]*S[2]+B[2,0]*EX1[0]+B[2,1]*EX1[1]+B[2,2]*EX1[2]+C[2,0]*EX2[0]+C[2,1]*EX2[1]+C[2,2]*EX2[2]+D[2,2]*(para[15])*EPS[2,t,:]
        
        
        err = (y_data[t,0,0] - (1/100)*(S[0,:] -Xb-S[3,:]))**2 + (y_data[t,1,0] - (1/400)*S[1,:])**2 + (y_data[t,2,0] - (1/400)*S[2,:])**2
        w = (((2*math.pi)**(-3/2)))*cp.exp((-1/2)*(err))  
        w= w/cp.sum(w)
        #X[:,t] = w
        error[t] = cp.sum(w*err)
        w = cp.reshape(w,[J])
        #Resampling Procedure [0.1, 0.1, 0.1, 0.1, 0.1, 0.1,.1,.1,.1,.1]
        idx =np.array([]) 
        try:
            idx = np.random.choice(np.arange(J), J, replace=True,p=cp.asnumpy(w))
            S = S[:,idx]
            PHI = PHI[:,idx]
            RR = RR[:,:,idx]
        except:
            pass
        
        err = (y_data[t,0,0] - (1/100)*(S[0,:] -Xb-S[3]))**2 + (y_data[t,1,0] - (1/400)*S[1])**2 + (y_data[t,2,0] - (1/400)*S[2])**2
        LL[t] = (((2*math.pi)**(-3/2))) * (1/J)*cp.sum(cp.exp((-1/2)*(w*err)))
        ERR = EX - S[0:3]

        PHI_old = PHI

        
        #K[:,t] = cp.sum(cp.reshape(w,[1,J])*S,1)

        PP[0,0,:] =1;PP[0,3,:] =S[0];PP[0,4,:] =S[1];PP[0,5,:] =S[2];PP[0,12,:] =S[4];PP[0,13,:] =S[3];
        #row 1
        PP[1,1,:] =1;PP[1,6,:] =S[0];PP[1,7,:] =S[1];PP[1,8,:] =S[2];PP[1,14,:] =S[4];PP[1,15,:] =S[3];
        #row 2
        PP[2,2,:] =1;PP[2,9,:] =S[0];PP[2,10,:] =S[1];PP[2,11,:] =S[2];PP[2,16,:] =S[4];PP[2,17,:] =S[3];
        #row 3
        PP[3,0,:] = S[0];PP[3,3,:] = S[0]**2;PP[3,4,:] = S[0]*S[1];PP[3,5,:] = S[0]*S[2];PP[3,12,:] = S[0]*S[4];PP[3,13,:] = S[0]*S[3]; 
        #row 4
        PP[4,0,:] = S[1];PP[4,3,:] = S[1]*S[0];PP[4,4,:] =S[1]**2;PP[4,5,:] = S[1]*S[2];PP[4,12,:] = S[1]*S[4];PP[4,13,:] = S[1]*S[3]; 
        #row 5
        PP[5,0,:] = S[2];PP[5,3,:] = S[2]*S[0];PP[5,4,:] =S[1]*S[2];PP[5,5,:] = S[2]*S[2];PP[5,12,:] = S[2]*S[4];PP[5,13,:] = S[2]*S[3]; 
        #row 6
        PP[6,1,:] = S[0];PP[6,6,:] = S[0]*S[0];PP[6,7,:] = S[0]*S[1];PP[6,8,:] = S[0]*S[2];PP[6,14,:] = S[0]*S[4];PP[6,15,:] = S[0]*S[3];
        #row 7
        PP[7,1,:] = S[1];PP[7,6,:] = S[1]*S[0];PP[7,7,:] = S[1]*S[1];PP[7,8,:] = S[1]*S[2];PP[7,14,:] = S[1]*S[4];PP[7,15,:] = S[1]*S[3];
        #row 8
        PP[8,1,:] = S[2];PP[8,6,:] = S[2]*S[0];PP[8,7,:] = S[2]*S[1];PP[8,8,:] = S[2]*S[2];PP[8,14,:] = S[2]*S[4];PP[8,15,:] = S[2]*S[3];
        #row 9
        PP[9,2,:] = S[0];PP[9,9,:] = S[0]*S[0];PP[9,10,:] = S[0]*S[1];PP[9,11:] = S[0]*S[2];PP[9,16,:] = S[0]*S[4];PP[9,17,:] = S[0]*S[3];
        #row 10
        PP[10,2,:] = S[1];PP[10,9,:] = S[1]*S[0];PP[10,10,:] = S[1]*S[1];PP[10,11,:] = S[1]*S[2];PP[10,16,:] = S[1]*S[4];PP[10,17,:] = S[1]*S[3];
        #row 11
        PP[11,2,:] = S[2];PP[11,9,:] = S[2]*S[0];PP[11,10,:] = S[2]*S[1];PP[11,11,:] = S[2]*S[2];PP[11,16,:] = S[2]*S[4];PP[11,17,:] = S[2]*S[3];
        #row 12
        PP[12,0,:] = S[4];PP[12,3,:] = S[0]*S[4];PP[12,4,:] = S[1]*S[4];PP[12,5,:] = S[2]*S[4];PP[12,12,:] = S[4]*S[4];PP[12,13,:] = S[3]*S[4];
        #row 13
        PP[13,0,:] = S[3];PP[13,3,:] = S[0]*S[3];PP[13,4,:] = S[1]*S[3];PP[13,5,:] = S[2]*S[3];PP[13,12,:] = S[3]*S[4];PP[13,13,:] = S[3]*S[3];
        #row 14
        PP[14,1,:] = S[4];PP[14,6,:] = S[0]*S[4];PP[14,7,:] = S[1]*S[4];PP[14,8,:] = S[2]*S[4];PP[14,14,:] = S[4]*S[4];PP[14,15,:] = S[3]*S[4];
        #row 15
        PP[15,1,:] = S[3];PP[15,6,:] = S[0]*S[3];PP[15,7,:] = S[1]*S[3];PP[15,8,:] = S[2]*S[3];PP[15,14,:] = S[4]*S[3];PP[15,15,:] = S[3]*S[3];
        #row 16
        PP[16,2,:] = S[4];PP[16,9,:] = S[0]*S[4];PP[16,10,:] = S[1]*S[4];PP[16,11,:] = S[2]*S[4];PP[16,16,:] = S[4]*S[4];PP[16,17,:] = S[4]*S[3];
        #row 17
        PP[17,2,:] = S[3];PP[17,9,:] = S[0]*S[3];PP[17,10,:] = S[1]*S[3];PP[17,11,:] = S[2]*S[3];PP[17,16,:] = S[4]*S[3];PP[17,17,:] = S[3]*S[3];    
        
        PP[9,12:16,0] =0
        RR = (1-para[7])*(RR)+para[7]*PP
        Rinv  = cp.linalg.inv(RR.T).T
        epsilon = cp.vstack([ERR[0],ERR[1],ERR[2],
                             S[0]*ERR[0],S[1]*ERR[0],S[2]*ERR[0],
                             S[0]*ERR[1],S[1]*ERR[1],S[2]*ERR[1],
                             S[0]*ERR[2],S[1]*ERR[2],S[2]*ERR[2],
                             S[3]*ERR[0],S[4]*ERR[0],
                             S[3]*ERR[1],S[4]*ERR[1],
                             S[3]*ERR[2],S[4]*ERR[2]])
        PHI = PHI+para[7]*cp.sum(cp.reshape(cp.array(list(epsilon)*18),[18,18,J])*Rinv,1)
        dropout = cp.sum(M)
        vv = check_eigenvalues(PHI) 
        vv = cp.reshape(vv,[1,J])
        #vv = cp.reshape(cp.logical_not(cp.all(abs(cp.linalg.eigvalsh(cp.reshape(PHI[3:12,:].T,[J,3,3]))) < 1,1)),[1,J]).astype(int)
        M[t] = cp.sum(w*vv)
        #x = cp.reshape(cp.repeat(cp.array(T3),[J]),[18,J]) + .01*cp.random.normal(0,1,[18,J])-PHI
        x = PHI_old-PHI
        PHI = PHI+vv*x
        

    Xb = S[0]
    dropout = cp.sum(M)
    liki = cp.log(cp.sum(LL))  
    SSE = cp.sum(error)

    return liki,dropout
#,SSE


# In[12]:


def REE_gen(para):
    p_r= float(para[0])
    sigma=float(para[1])
    phi_pi = float(para[2])
    phi_x=float(para[3])
    Beta=.99
    nu=float(para[5])
    theta=float(para[6])
    g=float(para[7])
    sig_r = float(para[8])
    alpha =float(para[9])
    psi = float(para[10])
    h = float(para[11])
    gamma = float(para[12])
    rho = float(para[13])
    p_u = float(para[14])
    sig_e = float(para[15])
    sig_u = float(para[16])
    alpha_p = (alpha/(1-alpha));
    k = ((1-Beta*theta)*(1-theta))/((1+alpha_p*psi)*theta*(1+gamma*Beta*theta));
    c1 = (sigma/((1-h)*(1-h*Beta)));
    c2 = (nu/(alpha + nu));
    w1 = (1+(h**2)*Beta + h*Beta)*((1+h+(h**2)*Beta)**-1)
    w2 =((1-h)*(1-h*Beta))*((sigma*(1+h+h**2*(Beta)))**-1)
    w3 = (-h*Beta/(1+h+(h**2)*Beta));
    w4 =h*((1+h+(h**2)*Beta)**-1)
    n1 = k*c2*c1+k*(h**2)*Beta*c1*c2-k*alpha_p
    n2 = -k*(c2)*(c1)*(h);
    n3 = -k*h*Beta*c1*c2;
    n4 =  Beta*((1+gamma*Beta*theta)**-1);
    n5 = gamma*((1+gamma*Beta*theta)**-1) + (-gamma*psi*alpha_p*k);  
    x_t   = 0;
    pi_t   = 1;
    i_t   = 2;
    r_t  = 3;
    u_t   = 4;
    Ex_t = 5;
    Epi_t = 6;
    Ei_t = 7;
    Ex_t2 = 8;
    Epi_t2 = 9;
    Ei_t2 = 10;
    
    ex_sh  = 0;
    epi_sh  =1;
    ei_sh  = 2;
    ex2_sh  =3 ;
    epi2_sh  =4 ;
    ei2_sh  =5 ;
    
    r_sh = 0;
    pi_sh = 1;
    i_sh = 2;
    
    neq  = 11;
    neta = 6;
    neps = 3;
    GAM0 = np.zeros([neq,neq]);
    GAM1 = np.zeros([neq,neq]);
    C = np.zeros([neq,1]);        
    PSI = np.zeros([neq,neps]);
    PPI = np.zeros([neq,neta]);
    eq_1 = 0
    eq_2    = 1;  
    eq_3    = 2;  
    eq_4    = 3;  
    eq_5   = 4;  
    eq_6    = 5;  
    eq_7    = 6; 
    eq_8    = 7;
    eq_9    = 8;
    eq_10    = 9;
    eq_11    = 10;
#x_t
    GAM0[eq_1,x_t]   =  1;
    GAM0[eq_1,Ex_t]   =  -w1;
    GAM0[eq_1,Epi_t]   =  -w2;
    GAM0[eq_1,i_t]   =  w2;
    GAM0[eq_1,r_t]   =  w2;
    GAM0[eq_1,Ex_t2]   =  -w3;
    GAM1[eq_1,x_t] = w4;
#pi_t
    GAM0[eq_2,pi_t]   =  1;
    GAM0[eq_2,x_t]   = -n1;
    GAM1[eq_2,x_t]   = n2;
    GAM0[eq_2,Ex_t]   = -n3;
    GAM0[eq_2,Epi_t]   = -n4;
    GAM1[eq_2,pi_t]   =  n5;
    GAM1[eq_2,u_t]  =  1;
#i_t
    GAM0[eq_3,x_t]   = -(1-rho)*phi_x;
    GAM0[eq_3,pi_t]  = -(1-rho)*phi_pi;
    GAM0[eq_3,i_t]  =1;
    GAM1[eq_3,i_t]  = rho;
    PSI[eq_3,i_sh] = 1;

#r_t
    GAM0[eq_4,r_t]   = 1;
    GAM1[eq_4,r_t] = p_r;
    PSI[eq_4,r_sh] = 1;
#u_t
    GAM0[eq_5,u_t]   = 1;
    GAM1[eq_5,u_t] = p_u;
    PSI[eq_5,pi_sh] = 1;
#Epi_t

    GAM0[eq_6,pi_t]   = 1;
    GAM1[eq_6, Epi_t] =1;
    PPI[eq_6, epi_sh] = 1;
#Ex_t

    GAM0[eq_7,x_t]   = 1;
    GAM1[eq_7, Ex_t] =1;
    PPI[eq_7, ex_sh] = 1;
#Ex_t2

    GAM0[eq_8,Ex_t]   = 1;
    GAM1[eq_8, Ex_t2] =1;
    PPI[eq_8, ex2_sh] = 1;
#Ei_t

    GAM0[eq_9,i_t]   = 1;
    GAM1[eq_9, Ei_t] =1;
    PPI[eq_9, ei_sh] = 1;   
    
#Ei_t2

    GAM0[eq_10,Ei_t]   = 1;
    GAM1[eq_10, Ei_t2] =1;
    PPI[eq_10, ei2_sh] = 1;    
#Epi_t2

    GAM0[eq_11,Epi_t]   = 1;
    GAM1[eq_11, Epi_t2] =1;
    PPI[eq_11, epi2_sh] = 1;    
    
    
    
    G1,impact,RC  = gensys(GAM0, GAM1, PSI, PPI, DIV=1 + 1e-8,REALSMALL=1e-6,return_everything=False)
    
    # GAM0*x(t) = A*x(t-1) + B*Ex(t+1) + C*Ex(t+2) + E*u(t-1) + DD*eps(t)

    GAM0 = cp.array([[1,0,w2],[-n1,1,0],[-(1-rho)*phi_x,-(1-rho)*phi_pi,1]])
    GAM0inv = cp.linalg.inv(GAM0)
    A  = cp.matmul(GAM0inv,cp.array([[w4,0,0],[n2,n5,0],[0,0,rho]]))
    B = cp.matmul(GAM0inv,cp.array([[w1,w2,0],[n3,n4,0],[0,0,0]]))
    C = cp.matmul(GAM0inv,cp.array([[w3,0,0],[0,0,0],[0,0,0]]))
    DD = cp.matmul(GAM0inv,cp.array(([0,0,0],[0,0,0],[0,0,1])))  
    E = cp.matmul(GAM0inv,cp.array([[-w2,0],[0,1],[0,1]]))
    R = cp.array([[cp.float(para[0]),0],[0,cp.float(para[14])]]) 
    # x(t) = A*x(t-1) + B*Ex(t+1) + C*Ex(t+2) + E*u(t-1) + DD*eps(t)
    V_s = cp.array([[cp.float(para[8])**2,0,0],[0,cp.float(para[16])**2,0],[0,0,cp.float(para[15])**2]])
    M1 = cp.array([[1/100,0,0,-1/100,0],[0,1/400,0,0,0],[0,0,1/400,0,0]])
    M2 = cp.array([[-1/100,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
    V_m = cp.identity(3)
    return G1,impact,RC,A,B,C,E,DD,R,V_s,M1,M2,V_m


# In[13]:


def KFlike1(para):
    x =.01* cp.zeros([10,1])
    P = .1*cp.zeros([10,10])
    x0 =  .1*cp.random.rand(10,1)
    P0 = 1*cp.random.rand(10,10)
    RRR =  10*cp.reshape(cp.random.rand(18,18),[18,18])
    RR =  cp.reshape(RRR,[18,18,1])
    xb = x0
    M = 0
    EZ = cp.zeros([3,1])
    T = 260
    LL = cp.zeros([260,1])
    error = cp.zeros([T])
    shocks = cp.array([[1,0],[0,1]])
    G1,impact,RC,A,B,C,E,DD,R,V_s,M1,M2,V_m = REE_gen(para)
    MM = cp.hstack((M1,M2)) 
    T3 = cp.vstack((cp.vstack(((cp.zeros([3,1]),cp.reshape(G1[0:3,0:3],[9,1])))),cp.reshape(G1[0:3,3:5],[6,1])))
    PHI = T3 +0*.1*cp.random.standard_normal([18,1])
    for t in range(T):
    # GAM0*x(t) = A*x(t-1) + B*Ex(t+1) + C*Ex(t+2) + E*u(t) + DD*eps(t)

        PHI0 = cp.asarray(cp.reshape(PHI[0:3],[3,1]))
        PHI1 = cp.asarray(cp.reshape(PHI[3:12],[3,3]))
        PHI2 = cp.asarray(cp.reshape(PHI[12:18],[3,2]))
        PHI1_sq = cp.matmul(PHI1,PHI1)
        PHI_12 = cp.matmul(PHI1,PHI2)
        PHI_10 = cp.matmul(PHI1,PHI0)
        B0 = cp.matmul(B,(PHI0+PHI_10)) + cp.matmul(C,(PHI0+PHI_10+cp.matmul(PHI1_sq,PHI0)))
        B1 = cp.matmul(B,PHI1_sq) +  cp.matmul(C,cp.matmul(PHI1,PHI1_sq))
        ZZ = cp.matmul(PHI1_sq,PHI2) + cp.matmul(PHI_12,R) + cp.matmul(PHI2,R**2)
        B2 = cp.matmul(B,PHI_12) + cp.matmul(B,cp.matmul(PHI2,R)) + cp.matmul(C,ZZ)
        V0 = cp.vstack((B0,cp.zeros([2,1])))
        L1 = cp.vstack((B1+A,cp.zeros([2,3])))  # correct
        ZZ = cp.matmul(B2,R)+E
        L2 = cp.vstack((ZZ,R))  # correct
        zz = cp.hstack((B2,cp.zeros([3,1])))+DD
        V3 = cp.vstack((zz,cp.zeros([2,3])))
        V3[3:5,0:2] = shocks
        V1 = cp.hstack((L1,L2))  # correct
        AA0 = cp.vstack((V0,cp.zeros([5,1])))
        AA1 = cp.hstack((cp.vstack((V1,cp.identity(5))),cp.zeros([10,5]))) # correct
        V_S_bar = cp.vstack((V3,cp.zeros([5,3])))
        x = AA0 + cp.matmul(AA1,x0)
        y = cp.matmul(MM,x)
    
        P = cp.matmul(cp.matmul(AA1,P0),AA1.T) + cp.matmul(cp.matmul(V_S_bar,V_s),V_S_bar.T)
        D = cp.matmul(cp.matmul(MM,P0),MM.T) + V_m
        L = cp.matmul(P,MM.T)
        xb = x0    
        x0 = x + cp.matmul(cp.matmul(L,cp.linalg.inv(D)),(y_data[t,:,:]-y))
            
        P0 = P - cp.matmul(cp.matmul(L,cp.linalg.inv(D)),L.T)
        
        # calculate LL
        
        err = (y_data[t,:,:]- cp.matmul(M1,x0[0:5])-cp.matmul(M2,x0[5:]))**2
        error[t] = cp.sum(err)
        LL[t] = (((2*math.pi)**(-3/2))) * cp.exp((-1/2)*(cp.sum(err)))
        
        #RLS
        
        Xt = (x0[0]*Lx)+(x0[1]*Lpi)+(x0[2]*Li)+(x0[3]*U) +(x0[4]*Rs) +cp.array(cp.reshape(I,[3,18]))

        RR = (1-para[7])*RR + para[7]*cp.matmul(Xt.T,Xt)
        
        
        EX = PHI0 +cp.matmul(PHI1,xb[0:3])+ cp.matmul(PHI2,x0[3:5])
        
        ERR = cp.reshape(EX - x0[0:3],[3,1])
        
        PHI_old = PHI
        PHI = PHI + para[7]*cp.matmul(cp.matmul(cp.linalg.inv(RR.T).T[:,:,0],Xt.T),ERR)      
        
        eigval,eigvect = cp.linalg.eigh(PHI1)
        
        if (abs(eigval)>1).any():
            #PHI = T3 + .01*cp.random.standard_normal([18,1])
            M = M+1
            PHI = PHI_old
    SSE = cp.sum(error)
    
    return cp.log(cp.sum(LL)), M
#, SSE


# In[14]:


#EPS1,S1,Lx,Lpi,Li,I,Rs,U = Generate_S1(J=1,T = 260)
#KFlike1(para)


# In[ ]:


#PFLIKE(para,EPS,S,RR,randphi)
para = cp.zeros([17])
para[0] = .5
para[1] = .125
para[2] = 1.6
para[3] = .2
para[4] = .99
para[5] = .3
para[6] = .015
para[7] = .031
para[8] = .2
para[9] = .3
para[10] = 1.2
para[11] = .1
para[12] = .1
para[13] = .5
para[14] = .5
para[15] = .2
para[16] = .2


# In[41]:


J = 10000
T = 260
EPS,S = Generate_S(J,T)
RR = cp.array(cp.random.standard_normal([18,18,J]),cp.float32)
EPS = cp.array(EPS,cp.float32)
S = .1*cp.array(S,cp.float32)
RR = 10*cp.array(RR,cp.float32)
randphi = .1*cp.array(cp.random.standard_normal([J]),cp.float32)


# In[ ]:


# Begin MH
i = 0
Nsim = 100500
m = 1000
Thetasim = cp.zeros([Nsim,17])
logpost = cp.zeros([Nsim])
LIK = cp.zeros([Nsim])
AA = cp.zeros([Nsim])
DROP = cp.zeros([Nsim])
Thetasim[i,:] = para
c = .02
P3 = cp.eye(17)
POST = cp.load(r"C:\Research\KF Results\no proj\PostDIST.npy")
para = cp.mean(POST[10000:90000,:],0)
#P3 = cp.cov(POST[0:90000],rowvar= False)
Thetasim[i,:] = para
accept = 0
likij,dropoutj =  PFLIKE(Thetasim[i,:],EPS,S,RR,randphi)
obj = cp.float(likij)+cp.float(NK_priors(cp.asnumpy(Thetasim[i,:])))
DROP[i] = dropoutj
LIK[i] =cp.float(likij)
logpost[i] = obj
print('likelihood:', likij)
print('logposterior:', obj)


# In[44]:


#Continue MH
Nsim = 100500
m = 1000
i = cp.load(r'C:\Research\PF Results\Results\No Proj Fac\iter.npy')
Thetasim = cp.load(r'C:\Research\PF Results\Results\No Proj Fac\PostDIST.npy')
logpost = cp.load(r'C:\Research\PF Results\Results\No Proj Fac\logpost.npy')
AA = cp.load(r'C:\Research\PF Results\Results\No Proj Fac\Acceptance.npy')
DROP = cp.load(r'C:\Research\PF Results\Results\No Proj Fac\DROPpost.npy') 
LIK = cp.load(r'C:\Research\PF Results\Results\No Proj Fac\likelyhood.npy')
P3 = cp.load(r'C:\Research\PF Results\Results\No Proj Fac\covmat.npy')
c =cp.load(r'C:\Research\PF Results\Results\No Proj Fac\covmult.npy')
accept = int(AA[i]/100*i)
likij,dropoutj =  PFLIKE(Thetasim[i,:],EPS,S,RR,randphi)
obj = cp.float(likij)+cp.float(NK_priors(cp.asnumpy(Thetasim[i,:])))
LIK[i] =cp.float(likij)
logpost[i] = obj



# In[ ]:





# In[ ]:





# In[ ]:


go_on = 0
v = 0
AR = 0
while i<Nsim:
    #print(i)
    if i ==5000:
        P3 = cp.cov(Thetasim[0:i],rowvar= False)
    while go_on == 0:
            Thetac = cp.random.multivariate_normal(Thetasim[i,:],(c)*P3)
            try:
                G1,impact,RC,A,B,C,E,DD,R,V_s = REE_gen1(cp.asnumpy(Thetac))
            except:
                RC = cp.zeros([2,1])
            go_on = param_checks(Thetac,RC) 
    prioc =NK_priors(cp.asnumpy(Thetac))
    likic,dropoutc= PFLIKE(Thetac,EPS,S,RR,randphi)
    objc = cp.float(NK_priors(cp.asnumpy(Thetac))) + cp.float(likic)
    #print(objc)
    if cp.isfinite(objc) == False:
        alpha = -1
        u = 0
    else:
        u = cp.log(cp.random.uniform(0,1,1))
        alpha = objc-obj
    if  alpha-u>=0:
        Thetasim[i+1,:] = Thetac
        accept            = accept+1;
        DROP[i+1] = dropoutc
        logpost[i+1] = objc
        LIK[i+1] = likic
        obj = objc
        likij = likic
        dropoutj = dropoutc
        #print('accepted')
    else:
        Thetasim[i+1,:] = Thetasim[i,:]
        logpost[i+1] = obj
        LIK[i+1] = likij
        DROP[i+1] = dropoutj

    AR= (accept/(i+1))*100
    if v == m:
                    
                    v = 0
                    if  AR>90:
                                        c = c*2
                    if 90>AR and AR>75:
                                        c = c*1.6
                    if 75>AR and AR>50:
                                        c = c*1.5
                    if 50>AR and AR>45:
                                        c = c*1.3
                    if 45>AR and AR>40:
                                        c = c*1.26
                    if 40>AR and AR>35:
                                        c = c*1.25
                    if 35>AR and AR>33:
                                        c = c*1.05
                    if 33>AR and AR >30:
                                        c = c*1
                    if 20>AR and AR >15:
                                        c = c*.45
                    if 15>AR and AR>10:
                                        c = c*.03
                    if 10>AR and AR >5:
                                        c = c*.01
                    if 5>AR:
                            c = c*.001
    AA[i] = AR;
    if (i % 500) == 1:
        print("Iteration: ", i)
        print("Acceptance Rate: ",AR)
        print('Avg Log Posterior:', cp.mean(logpost[0:i]))
        print('Avg Likelihood:', cp.mean(LIK[0:i]))
        print("cov mult: ",c)
        print("p_r:",cp.mean(Thetasim[0:i,:],0)[0])
        print("sigma:",((cp.mean(Thetasim[0:i,:],0)[1])**-1))
        print("phi_pi:",cp.mean(Thetasim[0:i,:],0)[2])
        print("phi_x:",cp.mean(Thetasim[0:i,:],0)[3])
        print("Beta:",.99)
        print("nu:",cp.mean(Thetasim[0:i,:],0)[5])
        print("theta:",cp.mean(Thetasim[0:i,:],0)[6])
        print("g:",cp.mean(Thetasim[0:i,:],0)[7])
        print("sig_r:",cp.mean(Thetasim[0:i,:],0)[8])
        print("alpha:",cp.mean(Thetasim[0:i,:],0)[9])
        print("psi:",cp.mean(Thetasim[0:i,:],0)[10])
        print("h:",cp.mean(Thetasim[0:i,:],0)[11])
        print("gamma:",cp.mean(Thetasim[0:i,:],0)[12])
        print("rho:",cp.mean(Thetasim[0:i,:],0)[13])
        print("p_u:",cp.mean(Thetasim[0:i,:],0)[14])
        print("sig_e:",cp.mean(Thetasim[0:i,:],0)[15])
        print("sig_u:",cp.mean(Thetasim[0:i,:],0)[16])
        cp.save(r'C:\Research\PF Results\Results\No Proj Fac\PostDIST', Thetasim)
        cp.save(r'C:\Research\PF Results\Results\No Proj FacAcceptance', AA)
        cp.save(r'C:\Research\PF Results\Results\No Proj Fac\logpost', logpost)
        cp.save(r'C:\Research\PF Results\Results\No Proj Fac\likelyhood', LIK)
        cp.save(r'C:\Research\PF Results\Results\No Proj Fac\DROPpost', DROP)
        cp.save(r'C:\Research\PF Results\Results\No Proj Fac\covmat', P3)
        cp.save(r'C:\Research\PF Results\Results\No Proj Fac\covmult', c)
        cp.save(r'C:\Research\PF Results\Results\No Proj Fac\iter', i)
        print("Current Time =",datetime.now().strftime("%H:%M"))
    go_on = 0
    i = i+1
    v = v+1


# In[ ]:




