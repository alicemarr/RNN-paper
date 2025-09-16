import numpy as np


def get_state_distance_between_trajs(traj_A, traj_B):
    # calculate Euclidean distance between network trajectories A and B (between
    # every time step per trial, then average over all time steps and trials)
    # traj_A/B: [n_units, n_timesteps, n_trajectories_per_group_to_compare], network trajectories
    #                                          to compare (population activities over time of trials)

    [_, n_timesteps, n_samples] = np.shape(traj_A)
    state_distances = np.zeros([n_timesteps, n_samples])
    for t_nr in range(n_timesteps):
        for sample_nr in range(n_samples):
            state_distances[t_nr, sample_nr] = np.linalg.norm((traj_A[:, t_nr, sample_nr]
                                                               - traj_B[:, t_nr, sample_nr]))

    state_distances = np.mean(np.mean(state_distances, axis=1), axis=0)
    return state_distances
