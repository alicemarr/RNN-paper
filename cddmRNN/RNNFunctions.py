import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

N = 100
W = np.load('./cddmRNN/W.npy')
Y = np.load('./cddmRNN/Y.npy')
n_inputs = 5
nIntegrators = 2
B = np.load('./cddmRNN/B.npy')
dt = 0.001
#t_ev = np.arange(0,2, dt)

def RHS(t,y, f1, fp = np.zeros(N),mat = np.identity(N), mat_inv = np.identity(N),tau = 0.01, dt = 0.001):
    i = int(t/dt)
    return (-y -np.matmul(mat,fp)+ np.dot(np.matmul(mat,W), np.tanh(np.matmul(mat_inv,y)+ fp)) +np.matmul(mat,f1[:,i]))/tau

def RHS0(y, f1):
    return (-y + np.dot(W, np.tanh(y)) +f1[:,0])

def J(fpt, tau = 0.01):
    return (- np.identity(N) + np.multiply(W,1/(np.cosh(fpt)**2)))/tau

# Function that tells you the stability type + spectral gaps given the fixed points + plots fp readout location (if needed)
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





# Fixed Point finder with suggestion (for parameter dependent case)
def find_fptPar(parVar, sugg, scale, itovar, parFixed1, parFixed2):
    # parVar is the vector of parameter values
    # sugg is the suggestion for the location of the fixed point
    # itovar is the index of the input to vary: 0 for sensory input 1, 1 for sensory input 2
    fixed_points = []
    nTimesteps = 1
    inputs = np.zeros([n_inputs, nTimesteps])
    for i, par in enumerate(parVar):
        # set inputs
        inputs = generate_inputP(itovar, par, parFixed1, parFixed2)

        
        f1 = (np.dot(B, inputs))
        if i ==0:
            it = 1000
        else: 
            it = 1000
        s = np.zeros((it, N))
        fpts = []
        for i in range(it):
            x0 = np.random.uniform(-scale*i,scale*i,N)  
            if i == 0:
                if i ==0:
                    x0 = sugg
            if (i !=0) & (len(fixed_points)!=0) : 
                if (i< len(fixed_points[-1])):
                    x0 = fixed_points[-1][i]
            if len(fpts) == 7:
                break
            sol = sp.optimize.root(RHS0, x0, args=(f1), tol = 1e-10)
            if sol.success == True:
                s[i] = sol.x
                if len(fpts)!=0:
                    if np.all(np.linalg.norm(np.array(fpts)-s[i], axis = 1)>1e-8):
                        fpts.append(s[i])
                else: 
                    fpts.append(s[i])
        fixed_points.append(fpts)
    return fixed_points


# Generate input when varying the sensory inputs parametrically
def generate_inputP(itovar, c, cfx, ctxt, nTimesteps = 4000):
    inputs = np.zeros([n_inputs, nTimesteps])
    
    if itovar ==0:
            Cs = [c, cfx]
            context = ctxt
            inputs[ (n_inputs - nIntegrators - 2) + context :] = 1
    if itovar ==1:
            Cs = [cfx, c]
            context = ctxt
            inputs[ (n_inputs - nIntegrators - 2) + context, :] = 1
    for i in range(nIntegrators):
            inputs[i,:] = Cs[i]
    return inputs


# Generate trajectories for parameter-dependent dynamics
def generate_trajectoriesP(Fpts,unstable_fp,c1, inputs, c2 = 0.036, shift = True, t_ev = np.arange(0,4,dt)):
    # Fpts is a list of fixed points for every parameter value
    # unstable_fp is the location of the unstbale fixed point for every parameter value, epsilon
    training_trials = 2*len(Fpts)+2
    test_trials = 2
    sols = np.zeros((training_trials+test_trials, N,len(t_ev[:])))
    dataTrain = np.zeros((training_trials,N, len(t_ev[:])))
    dataTest = np.zeros((test_trials,N, len(t_ev[:])))
    Jevals, Jevecs = np.linalg.eig(J(unstable_fp))
    e1 = Jevecs[:,np.argsort(Jevals)][:,-1]
    coherence = [c1,c2]
    # set sensory inputs
    for i in range(nIntegrators):
        inputs[i,:] = coherence[i]
    f1 = (np.dot(B, inputs))
    k = 0
    for i in range(2*len(Fpts)):
        pt = Fpts[i//2]
        if k%2 == 0:
            x0 = pt + 0.01*(k+1)*e1
        if k%2 == 1: 
            x0 = pt - 0.01*(k+1)*e1
        
        sols[k,:] = sp.integrate.solve_ivp(RHS, [t_ev[0], t_ev[-1]],  x0,args= (f1,),t_eval = t_ev[:], rtol = 1e-6, atol = 1e-8).y
        if shift == True:
            dataTrain[k] = sols[k]-np.tile(np.expand_dims(unstable_fp,1), len(t_ev[:]))
        if shift == False:
            dataTrain[k] = sols[k]

        k = k+1
    for i in range(k,training_trials+test_trials):
        if i%2 == 0:
            x0 = np.zeros(N) + 0.01*(i+1)*e1
        if i%2 == 1: 
            x0 = np.zeros(N) - 0.01*(i+1)*e1

        sols[i,:] = sp.integrate.solve_ivp(RHS, [t_ev[0], t_ev[-1]],  x0, args= (f1,),t_eval = t_ev[:], rtol = 1e-12, atol = 1e-15).y
        if shift == True:
            if i<training_trials: 
                dataTrain[i] = sols[i]-np.tile(np.expand_dims(unstable_fp,1), len(t_ev[:]))
            else: 
                dataTest[i-training_trials] = sols[i]-np.tile(np.expand_dims(unstable_fp,1), len(t_ev[:]))
        if shift == False:
            if i<training_trials: 
                dataTrain[i] = sols[i]
            else: 
                dataTest[i-training_trials] = sols[i]



    n_trials = training_trials+test_trials
    time_instF = np.tile(np.expand_dims(t_ev[:],0),(n_trials,1),)
    time_instF = np.reshape(time_instF, (n_trials,1,len(t_ev[:])))
    return dataTest, dataTrain, time_instF

def plot_dataP(dataTest, dataTrain):
    fig, ax = plt.subplots()
    test_trials = 2
    n_trials = np.shape(dataTest)[0]+np.shape(dataTrain)[0]
    for i in range(n_trials-test_trials):
        ax.plot(np.dot(Y,np.tanh(dataTrain[i]))[0], '-',markersize = 1, linewidth = 1, color = 'black',)
        if i == n_trials-test_trials-1:
            ax.plot(np.dot(Y,np.tanh(dataTrain[i]))[0], '-',markersize = 1, linewidth = 1, color = 'black',label = 'training trajectories')
    for i in range(test_trials):
        ax.plot(np.dot(Y,np.tanh(dataTest[i]))[0], '--',markersize = 1, linewidth = 1, color = 'red',)
        if i == n_trials-1:
            ax.plot(np.dot(Y,np.tanh(dataTest[i]))[0], '--',markersize = 1, linewidth = 1, color = 'red',label = 'test trajectories')
    ax.set_xlabel('time steps')
    ax.set_ylabel(r'z')
    ax.set_title('Data trajectories')
    ax.legend()
