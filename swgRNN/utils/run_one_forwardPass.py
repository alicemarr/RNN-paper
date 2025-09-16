import numpy as np
import h5py


from rnnFramework.run_rnn import run_rnn

def run_one_forwardPass(n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1,
                        inputs, conditionIds, seed_run, net_noise):
    # run network for one forward pass (several trials)
    # n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1: network weights
    # inputs: [2*nIntegrators+1 x T x N], sensory and context input over time and trials
    # conditionIds = [1 x N], context ID per trial
    # seed_run: (integer), random seed to get frozen noise
    # net_noise: (bool), if true: add gaussin noise to hidden unit activity

    # 'forwardPass'
    # n_x0_1: [n_units, n_trials], initial condition(s)
    # n_x_t: [n_units, n_timesteps, n_trials], membrane potentials of recurrent units (linear)
    # n_r0_1: [n_initial_conditions, n_trials], tanh(initial condition(s))
    # n_r_t: [n_units, n_timesteps, n_trials], activities of recurrent units (incl. tanh)
    # m_z_t: [n_outputs, n_timesteps, n_trials], activities of readout unit

    if not (seed_run is None):
        np.random.seed(seed_run)

    net = {}
    net["tau"] = 0.01
    net["dt"] = 0.001
    net["noiseSigma"] = 0.1

    if not (net_noise == 'default'):
        net["noiseSigma"] = net_noise

    forwardPass = {};
    forwardPass['n_r_t'], forwardPass['m_z_t'], forwardPass['n_r0_1'], \
    forwardPass['n_x_t'], forwardPass['n_x0_1'] = run_rnn(n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c,
                                                          n_bx_1, m_bz_1, net["dt"], net["tau"],
                                                          net["noiseSigma"], inputs, conditionIds)

    return forwardPass
