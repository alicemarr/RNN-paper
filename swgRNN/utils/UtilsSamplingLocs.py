import numpy as np


class UtilsSamplingLocs(object):
    # class to collect helper function to collect sampling locations

    def get_sampling_location_properties(self, network_type):
        # get properties of sampling locations,
        # here sampling locations defined along condition average
        # trajectories, equally sampled in time (ctxt: 1:100:14000, swg: 1:50:500)
        # network_type: (str), 'swg' or 'ctxt'

        # sampling_loc_props: (dict) with properties of all sampling locations
        # t_start_pt_per_loc: (list), t along trial in condition average trajectory of all sampling locations
        # freq_idx_per_loc: (list), frequency ID of all sampling locations ([] for ctxt network)
        # ctxt_per_loc: (list), context ID all sampling locations ([] for swg network)
        # signCoh1_per_loc: (list), sign of sensory input 1 of all sampling locations ([] for swg network)
        # signCoh2_per_loc: (list), sign of sensory input 2 of all sampling locations ([] for swg network)
        # t_sampling_locs_per_cond: (list), t along trial in condition average trajectory per input conditions
        # freq_idx_per_inpCond: (list), frequency ID per input conditions ([] for ctxt network)
        # ctxt_per_inpCond: (list), context ID per input conditions ([] for swg network)
        # signCoh1_per_inpCond (list), sign of sensory input 1 per input conditions ([] for swg network)
        # signCoh2_per_inpCond: (list), sign of sensory input 2 per input conditions ([] for swg network)
        # inpCond_names: (list) human-readable names of input conditions

        # default values
        sampling_loc_props = {}
        if network_type == 'swg':
            # every 50-th time step, every 5-th input frequency ID (1:5:51)
            sampling_loc_props["inpCond_names"] = ["Freq1", "Freq6", "Freq11",
                                                   "Freq16", "Freq21",
                                                   "Freq26", "Freq31",
                                                   "Freq36", "Freq41",
                                                   "Freq46",
                                                   "Freq51"]
            sampling_loc_props["freq_idx_per_inpCond"] = np.arange(1, 51 + 1, 5)
            n_inpConds = np.size(sampling_loc_props["freq_idx_per_inpCond"])
            sampling_loc_props["t_sampling_locs_per_cond"] = [1, 50, 100, 150,
                                                              200, 250, 300, 350,
                                                              400, 450, 500]
            nT_per_cond = len(sampling_loc_props["t_sampling_locs_per_cond"])

            sampling_loc_props["t_start_pt_per_loc"] = np.tile(
                sampling_loc_props["t_sampling_locs_per_cond"], [1, n_inpConds])
            sampling_loc_props["freq_idx_per_loc"] = np.concatenate(
                [np.ones([1, nT_per_cond]) * 1, np.ones([1, nT_per_cond]) * 6,
                 np.ones([1, nT_per_cond]) * 11, np.ones([1, nT_per_cond]) * 16,
                 np.ones([1, nT_per_cond]) * 21, np.ones([1, nT_per_cond]) * 26,
                 np.ones([1, nT_per_cond]) * 31, np.ones([1, nT_per_cond]) * 36,
                 np.ones([1, nT_per_cond]) * 41, np.ones([1, nT_per_cond]) * 46,
                 np.ones([1, nT_per_cond]) * 51], axis=1)

        elif network_type == 'ctxt':
            # every 100-th time step, every input conditions (context 1/2 x pos/neg. choice x coh/incoh. sensory inputs = 8 input conditions)
            sampling_loc_props["inpCond_names"] = ["Ctxt1-pos-coh", "Ctxt1-pos-incoh",
                                                   "Ctxt1-neg-coh", "Ctxt1-neg-incoh",
                                                   "Ctxt2-pos-coh", "Ctxt2-pos-incoh",
                                                   "Ctxt2-neg-coh", "Ctxt2-neg-incoh"]
            sampling_loc_props["ctxt_per_inpCond"] = [1, 1, 1, 1, 2, 2, 2, 2]
            sampling_loc_props["signCoh1_per_inpCond"] = [1, 1, -1, -1, 1, -1, -1, 1]
            sampling_loc_props["signCoh2_per_inpCond"] = [1, -1, -1, 1, 1, 1, -1, -1]
            nInpConds_ctx = len(sampling_loc_props["ctxt_per_inpCond"])
            sampling_loc_props["t_sampling_locs_per_cond"] = np.concatenate(
                [[1], np.arange(100, 1400 + 1, 100)], axis=0)
            nT_per_cond = np.size(sampling_loc_props["t_sampling_locs_per_cond"]);

            sampling_loc_props["t_start_pt_per_loc"] = np.reshape(
                np.asarray(
                    [1400, 1400, 1300, 1300, 1200, 1200, 1100, 1100, 1000,
                     1000, 900, 900, 800, 800, 700, 700, 600, 600, 500, 500,
                     400, 400, 300, 300, 200, 200, 100, 100, 1, 1,
                     1, 1, 100, 100, 200, 200, 300, 300, 400, 400, 500, 500,
                     600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000, 1100,
                     1100, 1200, 1200, 1300, 1300, 1400, 1400,
                     1400, 1400, 1300, 1300, 1200, 1200, 1100, 1100, 1000,
                     1000, 900, 900, 800, 800, 700, 700, 600, 600, 500, 500,
                     400, 400, 300, 300, 200, 200, 100, 100, 1, 1,
                     1, 1, 100, 100, 200, 200, 300, 300, 400, 400, 500, 500,
                     600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000, 1100,
                     1100, 1200, 1200, 1300, 1300, 1400, 1400]),
                [1, -1])
            sampling_loc_props["ctxt_per_loc"] = np.concatenate(
                [np.ones([1, int(nInpConds_ctx / 2 * nT_per_cond)]),
                 np.ones([1, int(nInpConds_ctx / 2 * nT_per_cond)]) * 2],
                axis=1)
            sampling_loc_props["signCoh1_per_loc"] = np.concatenate(
                [np.ones([1, int(nInpConds_ctx / 4 * nT_per_cond)]),
                 np.ones([1, int(nInpConds_ctx / 4 * nT_per_cond)]) * -1,
                 np.tile([1, -1], [1, int(nInpConds_ctx / 8 * nT_per_cond)]),
                 np.tile([-1, 1], [1, int(nInpConds_ctx / 8 * nT_per_cond)])],
                axis=1)
            sampling_loc_props["signCoh2_per_loc"] = np.concatenate(
                [np.tile([1, -1], [1, int(nInpConds_ctx / 8 * nT_per_cond)]),
                 np.tile([-1, 1], [1, int(nInpConds_ctx / 8 * nT_per_cond)]),
                 np.ones([1, int(nInpConds_ctx / 4 * nT_per_cond)]),
                 np.ones([1, int(nInpConds_ctx / 4 * nT_per_cond)]) * -1],
                axis=1)

        else:
            raise Exception("Network type unknown, please set network_type to 'swg' or 'ctxt'")

        return sampling_loc_props

    def get_sampling_locs_on_condAvgTrajs_ctxt(self, network_activity,
                                               sampling_loc_props,
                                               conditionIds,
                                               coherencies_trial):
        # find sampling locations as network activitiy vectors from
        # network activity based on defined properties of sampling
        # locations - CONTEXT-DEPENDENT INTEGRATION NETWORK

        # network_activity: [n_units, n_timesteps, n_trials], network activity x_t over t and trials
        # sampling_loc_props: (dict) with properties of all sampling locations:
        # conditionIds = [1, n_trials], context ID per trial
        # coherencies_trial = [nIntegrators, n_trials], input coherencies of sensory input 1 and 2 over trials

        # sampling_locs: [n_inpConds, n_units, n_sampling_locs], sampling locations sorted by input conditions

        # constants
        n_inpConds = len(sampling_loc_props["ctxt_per_inpCond"])
        n_units = np.shape(network_activity)[0]
        n_timesteps_total = np.shape(network_activity)[1]

        mean_traj_per_inpCond = np.full([n_inpConds, n_units, n_timesteps_total], np.nan)
        for inpCond_nr in range(n_inpConds):
            valid_trial_ids = np.squeeze(
                conditionIds == sampling_loc_props["ctxt_per_inpCond"][inpCond_nr]) & \
                              np.squeeze(np.sign(coherencies_trial[0, :]) ==
                                         sampling_loc_props["signCoh1_per_inpCond"][inpCond_nr]) & \
                              np.squeeze(np.sign(coherencies_trial[1, :]) ==
                                         sampling_loc_props["signCoh2_per_inpCond"][inpCond_nr])
            mean_traj_per_inpCond[inpCond_nr, :, :] = np.mean(network_activity[:, :, valid_trial_ids], axis=2)

        # get sampling_locations along mean trajectory
        sampling_locs = mean_traj_per_inpCond[:, :, np.asarray(
            sampling_loc_props["t_sampling_locs_per_cond"]) - 1]

        return sampling_locs

    def get_sampling_locs_on_condAvgTrajs_swg(self, network_activity,
                                              sampling_loc_props,
                                              all_freq_ids):
        # find sampling locations as network activitiy vectors from
        # network activity based on defined properties of sampling
        # locations - SINE WAVE GENERATION NETWORK

        # network_activity: [n_units, n_timesteps, n_trials], network activity x_t over t and trials
        # sampling_loc_props: (dict) with properties of all sampling locations
        # all_freq_ids: [n_trials, 1], all frequency IDs of each trial
        # sampling_locs: [n_inpConds, n_units, n_sampling_locs], sampling locations sorted by input conditions

        # constants
        n_inpConds = len(sampling_loc_props["freq_idx_per_inpCond"])
        n_units = np.shape(network_activity)[0]
        n_timesteps_total = np.shape(network_activity)[1]

        mean_traj_per_inpCond = np.full(
            [n_inpConds, n_units, n_timesteps_total], np.nan)
        for inpCond_nr in range(n_inpConds):
            valid_trial_ids = all_freq_ids[:, 0] == \
                              sampling_loc_props["freq_idx_per_inpCond"][inpCond_nr]
            if sum(valid_trial_ids) < 1:
                raise Exception("not enough trials per inpCond. Increase n_trials.")
            mean_traj_per_inpCond[inpCond_nr, :, :] = np.mean(
                network_activity[:, :, valid_trial_ids], axis=2)

        # get sampling_locations along mean trajectory at t_sampling_locs_per_cond
        sampling_locs = mean_traj_per_inpCond[:, :, np.asarray(
            sampling_loc_props["t_sampling_locs_per_cond"]) - 1]

        return sampling_locs

    def map_optLocNr_to_startPtNr(self, opt_loc_nr, sampling_loc_props):
        # mapping from 'sampling location number' to 'start_point_number' within on input condition
        # sampling_loc_props: (dict) with properties of all sampling locations
        start_pt_nr = np.where(
            sampling_loc_props["t_sampling_locs_per_cond"] ==
            sampling_loc_props["t_start_pt_per_loc"][0, opt_loc_nr])
        return start_pt_nr

    def map_optLocNr_to_inpCondNr(self, opt_loc_nr, sampling_loc_props, network_type):
        # mapping from 'sampling location number' to 'input condition number' within on input condition
        # sampling_loc_props: (dict) with properties of all sampling locations
        if network_type == 'swg':
            inpCond_nr = np.where(
                sampling_loc_props["freq_idx_per_loc"][0, opt_loc_nr] ==
                sampling_loc_props["freq_idx_per_inpCond"])

        elif network_type == 'ctxt':
            inpCond_nr = np.where(np.squeeze(
                sampling_loc_props["ctxt_per_inpCond"] ==
                sampling_loc_props["ctxt_per_loc"][0, opt_loc_nr]) & \
                                  np.squeeze(sampling_loc_props["signCoh1_per_inpCond"] ==
                                             sampling_loc_props["signCoh1_per_loc"][0, opt_loc_nr]) & \
                                  np.squeeze(sampling_loc_props["signCoh2_per_inpCond"] ==
                                             sampling_loc_props["signCoh2_per_loc"][0, opt_loc_nr]))

        else:
            raise Exception("Network type unknown, please set network_type to 'swg' or 'ctxt'")

        return inpCond_nr
