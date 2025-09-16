import numpy as np


def get_mse(m_z_t, targets, valid_trial_ids):
    # mse = mean squared error --> mean((y_hat-y)**2)
    # m_z_t:  [n_outputs, n_timesteps, n_trials], network output over t and trials
    # targets: [n_outputs, n_timesteps, n_trials], correct network output targets over t and trials
    # valid_trial_ids: [list], list of trials numbers which should be considered to calculate the mse

    if str(valid_trial_ids) == 'all':
        n_trials = np.shape(m_z_t)[2]
        valid_trial_ids = np.arange(n_trials)

    mse = np.mean(np.square((m_z_t[0, :, valid_trial_ids] - targets[0, :, valid_trial_ids])))
    return mse
