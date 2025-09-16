import numpy as np


class InputGeneratorCtxt(object):
    # class to generate input/targets for context-dependent integration
    def __init__(self):
        self.n_inputs = 5
        self.n_outputs = 1
        self.nIntegrators = 2
        self.maxBound = 1
        self.minBound = -1
        self.integrationTau = 5 * 0.01

        self.burnLength = 0.650
        self.dt = 0.001
        self.tau = 0.01
        self.time = 0.75

        self.all_coherencies = np.array([-0.5, -0.12, -0.03, 0.03, 0.12, 0.5]) * 0.3

    def get_ctxt_dep_integrator_inputOutputDataset(self, n_trials, with_inputnoise):
        # main method to generate input/outputs
        # n_trials: (integer), number of trials per context and per coherency
        # with_inputnoise: (bool), if true: add noise to sensory inputs

        # coherencies_trial = [nIntegrators x N], input coherencies of sensory input 1 and 2 over trials
        # coherencies are shuffled. in particular the different coherencies within a
        # trial are independent.
        # conditionIds = [1 x N], context ID per trial
        # inputs: [2*nIntegrators+1 x T x N], sensory and context input over time and trials
        # targets: [1 x T x N], target output over time and trials

        # noise settings
        input_noise = 0
        if with_inputnoise:
            input_noise = 3.16228 * np.sqrt(self.dt)
        noiseSigma = input_noise

        # generate inputs/targets
        n_trials_total = n_trials * len(self.all_coherencies) * self.nIntegrators
        nTimesteps = int(self.time / self.dt) + int(self.burnLength / self.dt)
        inputs = np.zeros([self.n_inputs, nTimesteps, n_trials_total])
        targets = np.zeros([self.n_outputs, nTimesteps, n_trials_total])
        conditionIds = np.tile(range(1, self.nIntegrators + 1),
                               [1, int(n_trials_total / self.nIntegrators)]) # array full of 1s and 2s of shape (1,n_trials_total)

        # set input coherencies per trial
        set_all_cohs = np.zeros([1, n_trials_total])
        for coh_nr in range(len(self.all_coherencies)):
            idx_start = coh_nr * (self.nIntegrators * n_trials)
            idx_end = (coh_nr + 1) * (self.nIntegrators * n_trials)
            set_all_cohs[0, idx_start:idx_end] = np.ones([1, self.nIntegrators * n_trials])\
                                                 * self.all_coherencies[coh_nr]


        # pick a length-(n_trials*coherence_values) long sequence and assign it to sensory inputs/
        # for both context 1 trials and context 2 trials

        coherencies_trial = np.zeros([self.nIntegrators, n_trials_total])
        for i in range(self.nIntegrators):
            coherencies_trial[i, np.where(conditionIds[0, :] == 1)] = \
            set_all_cohs[0, np.random.choice(range(n_trials_total),
                                             size=n_trials * len(self.all_coherencies),
                                             replace=False)] # context 1 trials
            coherencies_trial[i, np.where(conditionIds[0, :] == 2)] = \
            set_all_cohs[0, np.random.choice(range(n_trials_total),
                                             size=n_trials * len(self.all_coherencies),
                                             replace=False)] # context 2 trials

        # generate one trial
        for trial_nr in range(n_trials_total):
            inputs[:, :, trial_nr], targets[:, :,trial_nr] = self.generate_one_trial(
                coherencies_trial[0:2, trial_nr], conditionIds[0, trial_nr], noiseSigma)

        return coherencies_trial, conditionIds, inputs, targets

    def generate_one_trial(self, all_drifts, conditionId, noiseSigma):
        # method to generate input and targets over time for one trial
        # all_drifts: [n_integrators, 1], input coherency for sensory input 1 and 2
        # conditionId: (int), context of trial (1 or 2)
        # noiseSigma: (float), sigma of input noise

        # inputs: (n_inputs, n_timesteps)
        # targets: (n_outputs, n_timesteps)

        nTimesteps = int(self.time / self.dt) + int(self.burnLength / self.dt)
        inputs = np.zeros([self.n_inputs, nTimesteps])
        targets = np.zeros([self.n_outputs, nTimesteps])

        # set context inputs
        inputs[(self.n_inputs - self.nIntegrators - 2) + conditionId, :] = 1 # assign value 1 to input[2] or input[3]


        # set sensory inputs
        for i in range(self.nIntegrators):
            inputs[i, int(self.burnLength / self.dt):] = all_drifts[i]\
                 + np.random.normal(size=nTimesteps - int(self.burnLength / self.dt)) * noiseSigma

        # set target values
        hit_bound = 0
        for t in range(int(self.burnLength / self.dt), nTimesteps):
            if not hit_bound:
                targets[:, t] = targets[:, t - 1] + (self.dt / self.integrationTau) * inputs[conditionId - 1, t]
                if (targets[:, t] >= self.maxBound):
                    targets[:, t] = self.maxBound
                    hit_bound = 1
                elif (targets[:, t] <= self.minBound):
                    targets[:, t] = self.minBound
                    hit_bound = 1
            else:
                targets[:, t] = targets[:, t - 1]

        return inputs, targets
