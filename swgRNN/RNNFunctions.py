import numpy as np
import os 
from matplotlib import pyplot as plt

network_type = 'swg' 
dim_type = 'columns' 
base_dir_data = './swgRNN'

from swgRNN.utils.UtilsIO import UtilsIO
from swgRNN.utils.UtilsOpDims import UtilsOpDims

UIO  = UtilsIO()
UOD = UtilsOpDims()
net_id = 1
path_to_weights = os.path.join(base_dir_data,'pretrained_networks', network_type, network_type+'_weights.h5')
W = UIO.load_weights(path_to_weights, net_id)[1]

Y = UIO.load_weights(path_to_weights, net_id)[2]

B = UIO.load_weights(path_to_weights, net_id)[0]
N = 100

# RHS of the RNN system
def RHS(t,y, f1, fp = np.zeros(N), mat = np.identity(N), mat_inv = np.identity(N),tau = 0.01, dt = 1e-3):
    i = int(t/dt)
    return (-y -np.matmul(mat,fp)+ np.dot(np.matmul(mat,W), np.tanh(np.matmul(mat_inv,y)+ fp)) +f1[:,i])/tau

# to find fixed points
def RHS0(y, f1, tau = 0.01):
    return (-y + np.dot(W, np.tanh(y)) +f1[:,0])/tau


def J(fpt, tau = 0.01):
    return (- np.identity(N) + np.multiply(W,1/(np.cosh(fpt)**2)))/tau

def f0(x,fp, tau = 0.01):
    return (np.matmul(W, np.tanh(x))- np.matmul(np.multiply(W,1/(np.cosh(fp)**2)), x))/tau

from rnn_paper.SSMfunctions import construct_SSM
def ROMt(t,u, coeffs, exps, ss, f1, fp, tau = 0.01, dt = 1e-3):
    
    x = construct_SSM(u, coeffs[:,:], exps)
    return np.dot(ss.T,RHS(t,x, f1, fp))

def spectralProp(fpts, d = 1, showplot= False):
    stable_fpts = []
    unstable_fpts = []
    center_fpts = []
    labels = np.zeros(len(fpts))
    for j, pt in enumerate(fpts):
        Jevals, Jevecs = np.linalg.eig(J(pt))
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
    
    if showplot == True:
        fig, axs = plt.subplots()
        Time = np.linspace(0,10,100)
        for j,pt in enumerate(fpts):
            if labels[j]==1:
                axs.plot(Time,np.ones(len(Time))*np.dot(Y,np.tanh(pt))[0],'-', label = r' $|x_{0,%s}| = %s$'%(j,round(np.dot(Y,pt)[0],2)),color = 'blue')
            if labels[j]==0: 
                axs.plot(Time,np.ones(len(Time))*np.dot(Y,np.tanh(pt))[0],'--', label = r' $|x_{0,%s}| = %s$'%(j,round(np.dot(Y,pt)[0],2)), color = 'red')
            
        axs.set_xlabel(r'Time [s]', fontsize = 15)
        axs.set_ylabel(r'$z(x_0)$', fontsize = 15)

        axs.legend()
    return labels, stable_fpts, unstable_fpts, center_fpts