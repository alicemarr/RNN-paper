import numpy as np
import scipy as sp
import sympy as sym
from IPython.display import display, Math, Latex
from matplotlib import pyplot as plt
from rnn_paper.utils import GenerateSymbols

# Given coefficients and exponents of the polynomial expansion, computes the polynomial expansion
def evaluate_polynomial(C, I, P, symbolic = False ):
        
       # C is the coefficient (dim(SSM) \times m_order)-matrix 
       # I is the exponent (m_order \times dim(SSM))-matrix 
       # P is the d-dimensional vector to expand in monomials 

        if len(np.shape(P))!= 1:
                PHI = np.ones((np.shape(I)[0],*np.shape(P)[1:]), dtype = complex)
        else: 
                PHI = np.ones((np.shape(I)[0]),dtype =complex)
        if symbolic:
                PHI =  sym.MutableDenseNDimArray(PHI)
                C = np.round(C, 3)
        for i, exp in enumerate(I):
                
                for j in range(np.shape(P)[0]):
                        PHI[i] =  PHI[i]*np.power(P[j], exp[j])
        if len(np.shape(P))!= 1:   
            return  (np.tensordot(C, PHI, axes = 1))
        else: 
            return np.dot(C, PHI)


# Given coefficients and exponents of the (autonomous) SSM, computes the SSM expansion      
def construct_SSM(y, mfldCoeffs, mfldExps, symbolic = False):

    if symbolic == True:
           return evaluate_polynomial(mfldCoeffs, mfldExps, y, symbolic = symbolic)
    return evaluate_polynomial(mfldCoeffs, mfldExps,y)


# Given coefficients and exponents of the (autonomous) ROM, computes the ROM expansion
def ROM0(eta, rdCoeffs, rdExps, symbolic = False):

    if symbolic == True:
           return evaluate_polynomial(rdCoeffs,rdExps, eta, symbolic = symbolic)
    return np.real(evaluate_polynomial(rdCoeffs,rdExps, eta, symbolic = symbolic))
# To integrate with scipy
def ROM(t,eta, rdCoeffs, rdExps):

    return evaluate_polynomial(rdCoeffs,rdExps, eta)



def ROM1D(t, eta, rdCoeffs, rdExps, symbolic = False):
     rhs = 0
     for i in range(rdCoeffs.shape[1]):
          rhs+= eta**rdExps[i,0]*rdCoeffs[0,i]
  
     return rhs

def ROM1D0(eta, rdCoeffs, rdExps, symbolic = False):
     rhs = 0
     for i in range(rdCoeffs.shape[1]):
          rhs+= eta**rdExps[i]*rdCoeffs[0,i]
     return rhs

     


# Print the ROM
def printROM(d,rdCoeffs, rdExps):

    etas, detas = GenerateSymbols(d)
    # Construct ROM symbolic equations
    if d ==1:
        rhs = ROM1D0(etas,rdCoeffs, rdExps,symbolic=True)
    else:
        rhs = ROM0(etas,rdCoeffs, rdExps,symbolic=True)
         
    for i in range(d):
        display(Math(sym.latex(sym.Eq(detas[i], rhs[i]))))


# Calculate the Normal Mean Trajectory Error
def calculate_NMTE(nTest, DataTestTrunc, RomTraj):

    NMTE = 0
    for i in range(nTest):
        NMTE += np.mean(np.linalg.norm(DataTestTrunc[i,:,:]-RomTraj[i][:], axis = 0), axis = -1)/np.max(DataTestTrunc[i])

    return NMTE/nTest


# Calculate the Manifold Fitting Error
def calculate_MFE(nTest, DataTestTrunc, E, mfldCoeffs, mfldExps):

    MFE = 0
    for i in range(nTest):
        # Lift trajectories projected to the spectral subspace E to the SSM 
        LiftTraj = construct_SSM(np.dot(E.T,DataTestTrunc[i]),mfldCoeffs, mfldExps)
        MFE += np.mean(np.linalg.norm(DataTestTrunc[i,:,:]-LiftTraj[:], axis = 0), axis = -1)/np.max(DataTestTrunc[i])

    return MFE/nTest


####### Time-Dependent SSMs (Weak Forcing) #######



# Anchor traj at order 1
def compute_anchorO1(N, d, t0, nTimesteps, evecs, evals, ns, stable = True, tau = 0.01, dt = 1e-3, tfin = 700):

    Sevals = np.copy(evals)

    if stable == False:
        Sevals[:d] = np.zeros((d), dtype= complex)
        Uevals = np.copy(evals)

        Uevals[d:] = np.zeros((N-d), dtype= complex)
    At = np.exp(np.outer(np.arange(0,nTimesteps*dt, dt), (Sevals)))
    Gt = np.zeros((nTimesteps, N,N), dtype = complex)
    for i in range(nTimesteps):
        Gt[i] = np.diag(At[i])
    Gt = np.matmul(evecs,np.matmul((Gt),np.linalg.inv(evecs)))
    if stable == False: 
        Atu = np.exp(np.outer(-np.arange(0,nTimesteps*dt, dt), (Uevals)))
        Gtu = np.zeros((nTimesteps, N,N), dtype = complex)
        for i in range(nTimesteps):
            Gtu[i] = np.diag(Atu[i])
        Gtu = np.matmul(evecs,np.matmul((Gtu),np.linalg.inv(evecs)))
    T, N, M = Gt.shape
    _, M2 = ns.T.shape
    assert M == M2, "Dimensions of G(t) and f(t) must align."

    # Initialize output
    h = np.zeros((T, N), dtype = complex)

    # Perform the convolution-like operation
    for t in range(T):
        for s in range(t ):
            h[t] += Gt[t - s] @ ns[:,s]*dt/tau
        if stable == False:
            for s in range(t,tfin):
                h[t] -= Gtu[s-t]@ ns[:,s]*dt/tau

    return h

def plotParamSSM(m,tangent_space,eta1_vec,all_coeffsP, fixed_pointsP,ps, p_fpt0, p0, expsP,rd_coeffsP, rd_expsP,parname,colors =['purple','orange','darkgreen', 'darkcyan', 'salmon'], fsize = 17, tsize = 13, dpi = 100 ):
   # fig = plt.figure(figsize=(15,8))
    fig = plt.figure(layout='constrained', figsize=(15, 6), dpi = dpi)
    subfigs = fig.subfigures(1, 2, )
    #ax1 = fig.add_subplot(2, 2, 1, )
    ax2 = subfigs[1].subplots(subplot_kw={"projection": "3d"})
    axsLeft = subfigs[0].subplots(len(ps), 2, sharex=True)

    for e, epsilon in enumerate(ps): 
    
 
        for pt in fixed_pointsP[e]:
            ptC = (np.tensordot(tangent_space.T,np.array(pt).T-p_fpt0 , axes =1))
            ptSSM = construct_SSM([ptC, (epsilon-p0)], all_coeffsP, expsP)
            #ax2.plot(ptC,(epsilon-p0),(np.array(pt).T-p_fpt0 )[m],'x', color = colors[e])
            ax2.plot(ptC,(epsilon-p0),ptSSM[m],'X', color = colors[e], markersize = 8)
            axsLeft[e,0].plot(ptC,(np.array(pt).T-p_fpt0 )[m],'X', color = colors[e], markersize = 8)
            #axsLeft[e,0].plot(ptC,ptSSM[m],'', color = colors[e])
            axsLeft[e,1].plot(ptC,0,'X', color = colors[e], markersize = 8)
        # plot the manifold
       
        ssm = construct_SSM([eta1_vec, np.ones(len(eta1_vec))*(epsilon-p0)], all_coeffsP, expsP)
        axsLeft[e,0].plot(eta1_vec,ssm[m], '-',color =colors[e], label = r'$%s =%s$'%(parname,epsilon))
        ax2.plot(eta1_vec,(epsilon-p0)*np.ones(len(eta1_vec)),ssm[m], '-',color =colors[e], label = r'$%s =%s$'%(parname,epsilon))
        eta1_dot = ROM0([eta1_vec, (epsilon-p0)*np.ones(len(eta1_vec))], rd_coeffsP, rd_expsP)
        axsLeft[e,1].plot(eta1_vec, eta1_dot[0], label =  r'$%s =%s$'%(parname,epsilon), color = colors[e])
        axsLeft[e,1].plot(eta1_vec,np.zeros(len(eta1_vec)),'--', color = 'red',)
        axsLeft[-1,0].set_xlabel(r'$\eta_1$', fontsize = fsize)
        axsLeft[-1,1].set_xlabel(r'$\eta_1$', fontsize = fsize)
        axsLeft[e,0].set_ylabel(r'$y_%s$'%(m+1), fontsize = fsize)
        axsLeft[e,1].set_ylabel(r'$\dot{\eta}_1$', fontsize = fsize)
        axsLeft[e,0].tick_params(axis='both', which='major', labelsize=tsize)
        axsLeft[e,1].tick_params(axis='both', which='major', labelsize=tsize)
        axsLeft[e,1].legend(fontsize = fsize, loc = 'upper center')
        axsLeft[e,0].legend(fontsize = fsize, loc = 'upper center')
        axsLeft[e,1].set_ylim(-1,1)

    p_vals = np.linspace(ps[0]-0.001,ps[-1]+0.001,100)
    ax2.set_xlabel(r'$\eta_1$',fontsize = fsize)
    ax2.set_ylabel(r'$%s-%s0$'%(parname, parname),fontsize =fsize)
    ax2.set_zlabel(r'$y_%s$'%(m+1), fontsize = fsize)
    #eta1_vec = np.linspace(lims[0]+xmargins[0],lims[1]+xmargins[1],100)
    ETA, P = np.meshgrid(eta1_vec, p_vals)

    ssm = construct_SSM([ETA, P-p0], all_coeffsP, expsP)

    ax2.plot_surface(ETA,P-p0,ssm[m],color = 'mintcream', alpha = 0.2,  edgecolors='lightgray', lw = 0.01, zorder = 2)
    ax2.legend(fontsize = fsize , )

    ax2.tick_params(axis='both', which='major', labelsize=tsize )

    ax2.view_init(30, 40, )
    #ax2.set_yticks(np.linspace(p_vals[0], p_vals[-1],5))
    ax2.set_yticks([])
    return fig

# Calculate first order coefficients
'''def calculate_H11(N,d,t0, epsilon,evals, evecs, idx, anchor,d2fy2, ui, tsteps, dtInt, dt = 1e-3  ):

    P = evecs[:,np.flip(np.argsort(evals))]
    P_inv = np.linalg.inv(P)    
    A1 = np.copy(evals[d:])
    A1 = A1-evals[idx]*np.identity(N-d)   
    ki = 0
    summand = np.zeros(( N-d, int(tsteps+t0/dtInt)), dtype= complex)
    h11 = np.zeros((N-d,tsteps),dtype= complex)

    for i in np.arange(t0,tsteps+t0, 1):# t
    
        
        kj = 0
        for j in np.arange(t0,i, dtInt): # s
            G = np.exp(A1*(i-j)*dt)
            m11 = np.matmul(P_inv, np.dot(np.dot(d2fy2,anchor[:,int(j)]/epsilon), np.matmul(P,ui)))[d:]
            summand[:,kj] = np.matmul(G, m11)*dt*dtInt
            kj = kj+1

        h11[:,ki] = (np.sum(summand, axis = 1))
        ki = ki+1
    return h11'''

def calculate_H11_vectorized(N, d, evals_sorted, P, P_inv, d2fy2, anchor, ui, nTimesteps, div=1, dt=0.001, tau=0.01):
    # Step 1: Compute m11 (shape: (N-d, nTimesteps))
    tmp = np.dot(d2fy2, anchor)                # Shape: (M, M)
    tmp2 = np.matmul(P, ui)                    # Shape: (M, nTimesteps)
    m11 = np.matmul(P_inv, np.tensordot(tmp, tmp2, axes=(1, 0)))  # Shape: (N, nTimesteps)
    m11 = m11[d:]  # Shape: (N-d, nTimesteps)

    # Step 2: Compute diagonal matrices A(t) using broadcasting
    time = np.arange(0, nTimesteps * dt, dt / div)  # Shape: (T,)
    At = np.exp(np.outer(time, evals_sorted[d:]))   # Shape: (T, N-d)

    h = np.einsum('tn,nt->tn', At, m11) * dt / tau

    return h


def calculate_A1(d, evals, eig):
    Ak = np.copy(evals[d:])
    Ak = Ak-eig
    return Ak