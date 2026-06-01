#RNN-specific functions of the Multitasking RNN 

import numpy as np

# Load RNN params
object_file = np.load('mtRNN/model_params.npz')
W= object_file['w_in'].T
N = W.shape[0]
Win = W[:,:20]
Wrec= W[:,20:]
Wout = object_file['w_out'].T

Bout = object_file['b_out']
Brec = object_file['b_in']
gamma = 0.2




# Readout Function
def zout(y):
    return np.dot(Wout, y)+ np.tile(np.expand_dims(Bout,1), (1,np.shape(y)[1]))


# Nonlinear activation function
def S(x): 
    return np.log(1+np.exp(x))

# Function to evolve the discrete dynamical system
def evolve(f, x0, kmax, uin): 
    xt = np.empty((*x0.shape, kmax))
    for k in range(kmax):
        xt[:,k] = f(x0, uin[:,k])
        x0 = xt[:,k]
    return xt

# RHS of the discrete dynamical system

def RHS(x, uin, fpt=np.zeros(N), mat = np.identity(N), mat_inv = np.identity(N), tau = 0.1):
    Z= np.matmul(Wrec, np.matmul(mat_inv,x)+fpt) + np.matmul(Win, uin)+ Brec
    alpha = 1/tau
    return (1-gamma)*(x + np.matmul(mat, fpt)) + gamma*np.matmul(mat,S(Z))


# Solve for fixed points of the RHS of the discrete dynamical system
def solveZeros(x,uin,tau = 0.1):
    Z = np.matmul(Wrec, x) + np.matmul(Win, uin)+ Brec
    alpha = 1/tau
    
    return (-alpha)*x + alpha*S(Z)
    

# RHS of the continuous dynamical system
def cRHS (t,x, uin,fpt=np.zeros(N), mat = np.identity(N), mat_inv = np.identity(N),tau = 0.1, dt = 0.02):
    i = int(t/dt)
    Z= np.matmul(Wrec, np.matmul(mat_inv,x)+fpt) + np.matmul(Win, uin[:,i])+ Brec
    alpha = 1/tau
    return (-alpha)*(x+np.matmul(mat, fpt)) + alpha*np.matmul(mat,S(Z))
# To find fixed points
def cRHS0 (x, uin,fpt=np.zeros(N), mat = np.identity(N), mat_inv = np.identity(N),tau = 0.1):
    
    Z= np.matmul(Wrec, np.matmul(mat_inv,x)+fpt) + np.matmul(Win, uin[:,0])+ Brec
    alpha = 1/tau
    return (-alpha)*(x+np.matmul(mat, fpt)) + alpha*np.matmul(mat,S(Z))

# To compute the Jacobian of the RHS
def S1(x,tau = 0.1):
    return 1/(1+np.exp(-x))
def J(fpt,uin,tau = 0.1):
    v = (np.matmul(Wrec, fpt) + np.matmul(Win,uin) + Brec)
    alpha = 1/tau
    return (-alpha)*np.identity(N) + alpha*np.multiply(Wrec,S1(v))


def spectralProp(fpts, uin,d = 1):
    stable_fpts = []
    unstable_fpts = []
    center_fpts = []
    labels = np.zeros(len(fpts))
    for j, pt in enumerate(fpts):
        Jevals, Jevecs = np.linalg.eig(J(pt,uin))
        zeros = np.where(np.real(Jevals) == 0)[0]

        if np.all(np.real(Jevals)<=0):
            stable_fpts.append(pt)
            labels[j] = 1 
            spectral_gaps = np.abs(np.diff(np.flip(np.sort(np.real(Jevals)))))
            print(r'The %s fixed point is stable, with spectral gap'%(j), spectral_gaps[:d],r'compared to the real part of the next eigenvalue', np.flip(np.sort(np.real(Jevals)))[1:d+1])
        else:
            zeros = np.where(np.real(Jevals) == 0)[0]

            if len(zeros)!= 0:
                print('The %s fixed point has center manifold of dimension %s'%(j,len(zeros)))
                center_fpts.append(pt)
            if np.any(np.real(Jevals)>0):
                unstable_fpts.append(pt)
                unstable_Jevals = Jevals[np.where(np.real(Jevals)>0)]
                unstable_mfld_dim = len(unstable_Jevals)
                spectral_gaps = np.abs(np.diff(np.flip(np.sort(np.real(Jevals)))))
                print('The %s fixed point is unstable, with unstable manifold of dimension %s with spectral gap'%(j, unstable_mfld_dim),spectral_gaps[:d],r'compared to the real part of the next eigenvalue', np.flip(np.sort(np.real(Jevals)))[1:d+1])
    
    return labels, stable_fpts, unstable_fpts, center_fpts
    
    