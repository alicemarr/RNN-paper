import numpy as np
import random

class simulationParameters(object):
    def __init__(self, n_trials):
        self.dt = 0.001  # simulation time step
        self.time = 0.5  # trial duration
        self.nTrials = n_trials


class inputParameters(object):
    def __init__(self):
        self.nFreqs = 51
        self.max_freq = 6
        self.min_freq = 1
        self.seed_freqs = 2001  # set to None to avoid setting seed


class InputGeneratorSWG(object):
    # class to generate input/targets for sine wave generation
    def get_sine_wave_generator_inputOutputDataset(self, n_trials):
        # generate input/targets for sine wave genration as follows:
        # input for nFreqs different frequencies equally spread between
        # min_freq - max_freq.
        # inputs are then set to a static value of
        # (freq_id/nFreqs)+0.25.
        # targets = sin(2*pi*freq*t)

        # n_trials: (int), number of trial in total (uniformly sampled across all nFreqs)

        # all_freqs: [1, n_trials], frequencies of each trial
        # all_freq_ids: [n_trials, 1], all frequency IDs of each trial
        # input: [1, n_timesteps, n_trials]
        # targets: [1, n_timesteps, n_trials]
        # conditionIds: [1, n_trials], input condition (always 1)

        simParams = simulationParameters(n_trials)
        inputParams = inputParameters()

        if not (inputParams.seed_freqs is None):
            random.seed(inputParams.seed_freqs)

        freq_range = inputParams.max_freq - inputParams.min_freq
        delta_freq = freq_range / (inputParams.nFreqs - 1)
        freq_per_idx = np.arange(inputParams.min_freq, inputParams.max_freq + delta_freq, delta_freq)

        # set frequency IDs per trial
        all_freq_ids = np.random.choice(range(1, inputParams.nFreqs + 1),
                                        size=[n_trials, 1], replace=True)

        # generate inputs and target over time per trial
        t = np.arange(simParams.dt, simParams.time + simParams.dt, simParams.dt)
        all_freqs = np.full([1, n_trials], np.nan)
        inputs = np.full([1, len(t), n_trials], np.nan)
        targets = np.full([1, len(t), n_trials], np.nan)
        for trial_nr in range(n_trials):
            freq_id = all_freq_ids[trial_nr, 0]
            freq = freq_per_idx[freq_id - 1]

            all_freqs[0, trial_nr] = 2 * np.pi * freq
            inputs[0, :, trial_nr] = (freq_id / inputParams.nFreqs) + 0.25
            targets[0, :, trial_nr] = np.sin(2 * np.pi * freq * t)

        conditionIds = np.ones([1, np.shape(inputs)[2]])

        return all_freqs, all_freq_ids, inputs, targets, conditionIds