import numpy as np
import scipy as sp
import sympy as sym
from matplotlib import pyplot as plt


# From complex eig matrix to real eig matrix
def compute_tSpace(evals,evecs):
    Es = np.copy(np.real(evecs))


    Ces = Es[:,np.nonzero(np.imag(evals))[0] ]
    Rsev = np.real(evecs[:,np.nonzero(np.imag(evals))[0] ])
    Csev = np.imag(evecs[:,np.nonzero(np.imag(evals))[0] ])
    Ces[:,::2] = Rsev[:,::2]
    Ces[:,1::2] = Csev[:,1::2]

    Es[:,np.nonzero(np.imag(evals))[0] ]=Ces

    return Es[:,np.flip(np.argsort(evals))]



# Find fixed points given RHSc
# Find fixed points given RHSc
def find_fpt(it, N, scale, ics, RHS0, *args, breakat1 = False):
    
    fpts = []
    if np.all(ics!=np.zeros(np.shape(ics))):
        print(1)
        for i, ic in enumerate(ics):
            sol = sp.optimize.root(RHS0, ic, tol = 1e-10, args = (*args,))
            if sol.success == True:
                s = sol.x
            # Check fixed points are distinct
                if len(fpts)!=0:
                    if np.all(np.linalg.norm(np.array(fpts)-s, axis = 1)>1e-8):
                        fpts.append(s)
                else: 
                    fpts.append(s)
    for i in range(it):
        x0 = np.random.uniform(-i*scale,i*scale,N)  
        ics[i] = x0
        
        if (breakat1) & (len(fpts)==1): 
             break
        sol = sp.optimize.root(RHS0, x0, tol = 1e-10, args = (*args,))
        if sol.success == True:
            s = sol.x
        # Check fixed points are distinct
            if len(fpts)!=0:
                if np.all(np.linalg.norm(np.array(fpts)-s, axis = 1)>1e-8):
                    fpts.append(s)
            else: 
                fpts.append(s)
    return fpts

# Plot training and test trajectories
def plot_data(Y,data_test, data_train):
    fig, ax = plt.subplots()
    test_trials = 2
    n_trials = np.shape(data_test)[0]+np.shape(data_train)[0]
    for i in range(n_trials-test_trials):
        ax.plot(np.dot(Y,np.tanh(data_train[i]))[0], '-',markersize = 1, linewidth = 1, color = 'black',)
        if i == n_trials-test_trials-1:
            ax.plot(np.dot(Y,np.tanh(data_train[i]))[0], '-',markersize = 1, linewidth = 1, color = 'black',label = 'Training trajectories')
    for i in range(test_trials):
        ax.plot(np.dot(Y,np.tanh(data_test[i]))[0], '--',markersize = 1, linewidth = 1, color = 'red',)
        if i == test_trials-1:
            ax.plot(np.dot(Y,np.tanh(data_test[i]))[0], '--',markersize = 1, linewidth = 1, color = 'red',label = 'Test trajectories')
    ax.set_xlabel('Time [s]', fontsize = 15)
    ax.set_ylabel(r'$z$', fontsize = 15)
    ax.legend()


# Generate train and test trajectories
def generate_traj(N, nTrain, nTest, tev, fpt, pt, dir1, dir2, scale, RHS,  u, anchorFp, ev= False,):

    # N is the system dimension
    # nTrain and nTest number of train and test traj
    # tev evolution time vector
    # fpt is the fixed point we shift at zero
    # pt is the point where we may want our ics to start
    # dir is the direction along we launch them with a certain scale (if 'False', then ics are unif distributed in [-scale, scale])
    # args are the arguments of the RHS  function

    nTraj = nTrain + nTest
    np.random.seed(3)
    sols = np.zeros((nTraj, N, len(tev)))
    
    if ev:
        for i in range(nTraj):
            if i%2 == 0:
                x0 = pt-fpt + scale*(i+1)*dir1
        
            if i%2 == 1:
                x0 =pt -fpt - scale*i*dir2
                
            sols[i]= sp.integrate.solve_ivp(RHS, [tev[0], tev[-1]], x0, t_eval=tev, args = (u, anchorFp,)).y

    else: 
        ics = np.random.uniform(-scale, scale, (nTraj,N))
        for i in range(nTraj):
            sols[i]= sp.integrate.solve_ivp(RHS, [tev[0], tev[-1]], ics[i], t_eval=tev, args = (u,anchorFp,)).y

    np.random.shuffle(sols)

    DataTrain = sols[:nTrain]
    DataTest = sols[nTrain:]
    timeS = np.tile(np.expand_dims(np.expand_dims(tev, 0),0), (nTraj, 1,1))
    return DataTrain, DataTest, timeS



#Plot training and test trajectories
def plot_data(Y,data_test, data_train, fsize = 17):
    fig, ax = plt.subplots()
    test_trials = 2
    n_trials = np.shape(data_test)[0]+np.shape(data_train)[0]
    for i in range(n_trials-test_trials):
        ax.plot(np.dot(Y,np.tanh(data_train[i]))[0], '-',markersize = 1, linewidth = 1, color = 'black',)
        if i == n_trials-test_trials-1:
            ax.plot(np.dot(Y,np.tanh(data_train[i]))[0], '-',markersize = 1, linewidth = 1, color = 'black',label = 'Training trajectories')
    for i in range(test_trials):
        ax.plot(np.dot(Y,np.tanh(data_test[i]))[0], '--',markersize = 1, linewidth = 1, color = 'red',)
        if i == test_trials-1:
            ax.plot(np.dot(Y,np.tanh(data_test[i]))[0], '--',markersize = 1, linewidth = 1, color = 'red',label = 'Test trajectories')
    ax.set_xlabel('Time [s]', fontsize = fsize)
    ax.set_ylabel(r'$z$', fontsize = fsize)
    ax.legend()

# Plot 1D SSM and Trajectories
def plotSSMandTraj(ax,m,tangent_space, ssm1d, ssmcolor, red_trajs, full_trajs, anchorpt, stable_fpts, unstable_fpts, dim = 2, pca = False, t0 = 0, fsize = 17, tsize = 13): #for a 1D SSM
    if (pca == True) or (dim == 3):
        pca_space = tangent_space
    
    for stable_fpt in stable_fpts:
        if dim == 3:
            ax.plot( *np.dot(pca_space.T,stable_fpt-anchorpt), stable_fpt[m]-anchorpt[m],'X',markersize = 8, color = 'blue',label = 'Stable Fixed Point')
        elif pca == True:
            ax.plot( *np.dot(pca_space.T,stable_fpt-anchorpt),'X',markersize = 8, color = 'blue',label = 'Stable Fixed Point')
        else:
            ax.plot(np.dot(tangent_space.T,stable_fpt-anchorpt), stable_fpt[m]-anchorpt[m],'X',markersize = 8, color = 'blue',label = 'Stable Fixed Point')

    #ax.plot( *np.dot(tangent_space.T,stable_fpt-anchorpt), stable_fpt[m]-anchorpt[m],'X',markersize = 8, color = 'blue',label = 'Stable Fixed Point')
    #ax.legend('Stable Fixed Point', fontsize = 17)
    for unstable_fpt in unstable_fpts:
        if dim == 3:
            ax.plot( *np.dot(pca_space.T,unstable_fpt-anchorpt), unstable_fpt[m]-anchorpt[m],'X',markersize = 8, color = 'red',label = 'Unstable Fixed Point')
        elif pca == True:
            ax.plot( *np.dot(pca_space.T,unstable_fpt-anchorpt),'X',markersize = 8, color = 'red',label = 'Unstable Fixed Point')
        else:
            ax.plot(np.dot(tangent_space.T,unstable_fpt-anchorpt), unstable_fpt[m]-anchorpt[m],'X',markersize = 8, color = 'red',label = 'Unstable Fixed Point')

    #ax.plot( *np.dot(tangent_space.T,unstable_fpt-anchorpt), unstable_fpt[m]-anchorpt[m],'X',markersize = 8, color = 'red',label = 'Unstable Fixed Point')
    #ax.legend('Unstable Fixed Point', fontsize = 17)

    for red_traj in red_trajs:
        if dim ==3:
            ax.plot(*np.dot(pca_space.T,red_traj),red_traj[m,],'--', color = 'purple', linewidth = 3, label = 'Reduced-Order')
            ax.plot(*np.dot(pca_space.T,red_traj)[:,0],red_traj[m,0],'o', color = 'purple', linewidth = 3,)
        elif pca == True:
            ax.plot(*np.dot(pca_space.T,red_traj),'--', color = 'purple', linewidth = 3, label = 'Reduced-Order')
            ax.plot(*np.dot(pca_space.T,red_traj)[:,0],'o', color = 'purple', linewidth = 3,)
        else:
            ax.plot(np.dot(tangent_space.T,red_traj),red_traj[m,],'--', color = 'purple', linewidth = 3, label = 'Reduced-Order')
            ax.plot(np.dot(tangent_space.T,red_traj)[0],red_traj[m,0],'o', color = 'purple', linewidth = 3, )
            
    #ax.legend('Reduced-Order', fontsize = 17)
    #ax.plot(*np.dot(tangent_space.T,red_traj),red_traj[m,],'--', color = 'purple', linewidth = 3, label = 'Reduced-Order')
    for full_traj in full_trajs:
        if dim == 3:
            ax.plot(*np.dot(pca_space.T, full_traj)[:,t0:], full_traj[m,t0:],'-', color = 'black',  label = 'Full-Order')
            ax.plot(*np.dot(pca_space.T, full_traj)[:,t0], full_traj[m,t0],'o', color = 'black', )
        elif pca == True:
            ax.plot(*np.dot(pca_space.T, full_traj)[:,t0:], '-', color = 'black',  label = 'Full-Order')
            ax.plot(*np.dot(pca_space.T, full_traj)[:,t0],'o', color = 'black', )
        else: 
            ax.plot(np.dot(tangent_space.T, full_traj)[t0:], full_traj[m, t0:],'-', color = 'black', label = 'Full-Order' )
            ax.plot(np.dot(tangent_space.T, full_traj)[t0], full_traj[m,t0],'o', color = 'black', )

    #ax.plot(*np.dot(tangent_space.T, full_traj), full_traj[m],'-', color = 'black', label = 'Full-Order')
    #ax.legend('Full-Order', fontsize = 17)
    if dim ==3:
        ax.plot(*np.dot(pca_space.T,ssm1d)[:], ssm1d[m, ],color = ssmcolor, alpha = 0.5)
    elif pca == True:
        ax.plot(*np.dot(pca_space.T,ssm1d)[:],color = ssmcolor, alpha = 0.5)
    else:
        ax.plot(np.dot(tangent_space.T,ssm1d),\
         ssm1d[m],color = ssmcolor, alpha = 0.5)
    ax.set_xlabel(r'$\eta_1$', fontsize = fsize)
    ax.set_ylabel(r'$\eta_2$', fontsize = fsize)
    if dim == 3:
        ax.set_zlabel(r'$y_{%s}$'%(m+1), fontsize = fsize, )
    elif pca == False:
        ax.set_ylabel(r'$y_{%s}$'%(m+1), fontsize = fsize, )



# Generate symbols
def GenerateSymbols(d):
    etas = []
    detas = []
    for i in range(d):
            etas.append( sym.symbols('\\eta_{%s}'%(i+1)))
            detas.append( sym.symbols('\\dot{\\eta}_{%s}'%(i+1)))
    return etas, detas


# FTLE
def evolution(func, t0, tf, x, *args, dt = 0.001):
    y = np.zeros((x.T).shape)
    for i,ic in enumerate(x.T):
        sol = sp.integrate.solve_ivp(func, [t0*dt, tf*dt],
                                    ic, t_eval = [tf*dt],
                                    method='DOP853',
                                    rtol=1e-4,
                                    atol=1e-5, args = (*args,))
        y[i] = sol.y[:, -1]
    return y

def calculate_FTLE(N,sols, x1, x2, M, t0, tf):
    diff = np.zeros((N, 2, M,M))
    for s, sol in enumerate(sols):
        diff[s,0], diff[s,1] = (arr for arr in np.gradient(sol, x1, x2))    
    FTLE = np.empty((M,M))
    DF = np.empty((2, N))

    for i in range(FTLE.shape[0]):
        for j in range(FTLE.shape[1]):
            for k, d in enumerate(diff):
                DF[:,k] = np.array([d[0,i, j],d[1,i, j]])
            C = np.dot(np.transpose(DF), DF)
            eigenvalues = np.linalg.eigh(C)[0]
            max = np.argmax(eigenvalues)
            FTLE[i, j] = np.log(eigenvalues[max]) / (2 * np.abs(tf - t0))