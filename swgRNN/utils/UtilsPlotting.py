import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA


class UtilsPlotting(object):
    # class to collect helper functions to visualize results
    def __init__(self):
        self.my_fontsize = 30;

    def plot_lineplot(self, x_data, y_data, my_title, my_xlabel, my_ylabel,
                      display_names=None, my_mad=None):
        # generate standard line plot
        # x_data: [1 , N], data to plot along x-axis
        # y_data: [M , N], data to plot along y-axis
        # my_title, my_xlabel, my_ylabel: (str), axis labels for title, x- and y-axis

        # format y_data
        if len(y_data.shape) > 1:
            if y_data.shape[0] > y_data.shape[1]:
                y_data = y_data.T
        else:
            y_data = np.reshape(y_data, [1, np.shape(y_data)[0]])

        # plot
        fig, ax = plt.subplots(1, 1, figsize=[7, 7])
        plt.grid()
        for data_nr in range(
                int(float(np.size(y_data)) / float(np.size(x_data)))):
            if display_names is None:
                display_name = '';
            else:
                display_name = display_names[data_nr]
            ax.plot(x_data, y_data[data_nr, :], linewidth=3,
                    label=display_name)

        # add shaded area for MAD
        if not (my_mad is None):
            ax.fill_between(x_data, y_data + my_mad, y_data - my_mad, color='k', alpha=0.2)
            inBetween = np.concatenate([y_data + my_mad, fliplr(y_data - my_mad)])
            y_max = np.max(np.concatenate([my_mad, y_data]));
        else:
            y_max = np.max([y_data[:]])

        # labels
        ax.tick_params(axis='both', which='major', labelsize=self.my_fontsize - 10)
        ax.set_title(my_title, fontsize=self.my_fontsize)
        ax.set_xlabel(my_xlabel, fontsize=self.my_fontsize)
        ax.set_ylabel(my_ylabel, fontsize=self.my_fontsize)
        if not (display_names is None):
            plt.legend(fontsize=self.my_fontsize)

        ax.set_ylim(bottom=0, top=1.1 * y_max)

        return fig, ax

    def project_sampling_locs_onto_PCs(self, sampling_locs, PCs):
        # get projection of n-dimensional sampling location vectors
        # onto 2/3-dimensional principal component dimensions
        # sampling_loc: [n_input_conditions, n_units, n_start_pts_per_inpCond], sampling locations for opdims
        # PCs: [n_units, n_pcs], principal components of network activity

        n_inpConds = np.shape(sampling_locs)[0]
        n_sampling_locs_per_conds = np.shape(sampling_locs)[2]
        n_PCs_to_proj = np.shape(PCs)[1]

        proj_sampling_locs = np.full([n_inpConds, n_sampling_locs_per_conds, n_PCs_to_proj], np.nan)
        for inpCond_nr in range(n_inpConds):
            for start_pt_nr in range(n_sampling_locs_per_conds):
                sampl_loc = sampling_locs[inpCond_nr, :, start_pt_nr]
                for PC_nr in range(n_PCs_to_proj):
                    proj_sampling_locs[inpCond_nr, start_pt_nr, PC_nr] = np.dot(PCs[:, PC_nr], sampl_loc[:])

        return proj_sampling_locs

    def get_PCs_of_network_activity(self, network_activity):
        # get principal components of network_activity
        # network_activity: [n_units, n_timesteps, n_trials]

        [n_units, n_timesteps, n_trials] = np.shape(network_activity)

        all_netActivity = np.full([n_units, n_trials * n_timesteps], np.nan)
        for trial_nr in range(n_trials):
            all_netActivity[:,trial_nr * n_timesteps:(trial_nr + 1) * n_timesteps] = \
                network_activity[:, :, trial_nr]

        pca = PCA(n_components=n_units)
        pca.fit(all_netActivity.T)  # [n_samples, n_features]
        PCs = pca.components_.T
        return PCs

    def project_condition_average_trajectories_onto_PCs(self, network_activity, PCs, network_type,
                                                        sampling_loc_props, all_freq_ids, conditionIds, coherencies_trial):
        # get projection of network activty onto PCs of network
        # activity, but sort by different input conditions

        # network_activity: [n_units, n_timesteps, n_trials]
        # PCs: [n_units, n_pcs], principal components of network activity
        # network_type: (str), 'swg' or 'ctxt'
        # sampling_loc_props: (dict) with properties of all sampling locations
        # all_freq_ids: [n_trials, 1], all frequency IDs of each trial
        # conditionIds = [1, n_trials], context ID per trial
        # coherencies_trial = [nIntegrators, n_trials], input coherencies of sensory input 1 and 2 over trials

        # constants
        n_timesteps = np.shape(network_activity)[1]
        n_PCs_to_proj = np.shape(PCs)[1]
        if network_type == 'swg':
            n_inpConds = np.size(sampling_loc_props["freq_idx_per_inpCond"])
        elif network_type == 'ctxt':
            n_inpConds = np.size(sampling_loc_props["ctxt_per_inpCond"])
        else:
            raise Exception(
                "Network type unknown, please set network_type to 'swg' or 'ctxt'")

        # sort trials per input condition and proj onto PCs
        proj_condAvgTrajs = np.full([n_inpConds, n_timesteps, n_PCs_to_proj], np.nan)
        for inpCond_nr in range(n_inpConds):
            # sort trials per input condition
            if network_type == 'swg':
                valid_trial_ids = all_freq_ids[:, 0] == sampling_loc_props["freq_idx_per_inpCond"][inpCond_nr]
            elif network_type == 'ctxt':
                valid_trial_ids = np.squeeze(conditionIds == sampling_loc_props["ctxt_per_inpCond"][inpCond_nr])\
                                  & np.squeeze(np.sign(coherencies_trial[0, :]) == sampling_loc_props["signCoh1_per_inpCond"][inpCond_nr])\
                                  & np.squeeze(np.sign(coherencies_trial[1, :]) == sampling_loc_props["signCoh2_per_inpCond"][inpCond_nr])
            else:
                raise Exception(
                    "Network type unknown, please set network_type to 'swg' or 'ctxt'")
            mean_traj = np.mean(network_activity[:, :, valid_trial_ids], axis=2)

            # proj onto PCs
            for PC_nr in range(n_PCs_to_proj):
                for t_nr in range(n_timesteps):
                    proj_condAvgTrajs[inpCond_nr, t_nr, PC_nr] = np.dot(PCs[:, PC_nr], mean_traj[:, t_nr])

        return proj_condAvgTrajs

    def plot_condAvgTrajs(self, ax, proj_condAvgTrajs, norm, network_type, legend_on, line_style):
        # plot projection of network activities over time
        # ax: (matlab subplot object)
        # proj_condAvgTrajs: [n_input_conditions, n_timesteps, n_PCS], projected network activities to plot
        # color_per_inpCond: [n_input_conditions, 3], rgb values for each input condition
        # network_type: (str), 'swg' or 'ctxt'
        # legend_on: (bool), if true: add legend to plot
        # line_style: (matlab line style), e.g. ':', '-', ...

        # constants
        n_inpConds = np.shape(proj_condAvgTrajs)[0]

        for inpCond_nr in range(n_inpConds):
            projs = np.squeeze(proj_condAvgTrajs[inpCond_nr, :, :])
            if network_type == 'swg':
                ax.plot(projs[:, 0], projs[:, 1], projs[:, 2], linewidth=3,
                        label=str(inpCond_nr + 1), linestyle=line_style,
                        color=np.asarray(cm.viridis(norm(inpCond_nr), bytes=True)) / 255.)
            elif network_type == 'ctxt':
                ax.plot(projs[:, 0], projs[:, 1], linewidth=3,
                        label=str(inpCond_nr + 1), linestyle=line_style,
                        color=np.asarray(cm.viridis(norm(inpCond_nr), bytes=True)) / 255.)

            if legend_on:
                ax.legend(fontsize=self.my_fontsize, loc='center left',
                          bbox_to_anchor=(1.1, 0.5), title="input conditions",
                          title_fontsize=self.my_fontsize)

    def plot_sampling_locs(self, ax, proj_sampling_locs, network_type):
        # plot projection of sampling locations
        # ax: (matlab subplot object)
        # proj_sampling_locs: [n_input_conditions, n_sampling_locs_per_conds, n_PCS], projected sampling locations to plot
        # network_type: (str), 'swg' or 'ctxt'

        # constants
        [n_inpConds, n_sampling_locs_per_conds, _] = np.shape(proj_sampling_locs);

        for inpCond_nr in range(n_inpConds):
            for start_pt_nr in range(n_sampling_locs_per_conds):
                proj_Pt = np.squeeze(proj_sampling_locs[inpCond_nr, start_pt_nr, :])
                if network_type == 'swg':
                    ax.scatter(xs=proj_Pt[0], ys=proj_Pt[1], zs=proj_Pt[2],
                               color='k', s=100, marker='d')
                elif network_type == 'ctxt':
                    ax.scatter(x=proj_Pt[0], y=proj_Pt[1], color='k', s=100, marker='d')

    def plot_sampling_locs_on_condAvgTrajs(self, network_activity, sampling_locs,
                                           sampling_loc_props, all_freq_ids, conditionIds, coherencies_trial, network_type):
        # plot low-dimensional projection of condition average trajectories with added sampling locations
        # network_activity: [n_units, n_timesteps, n_trials]
        # sampling_loc: [n_input_conditions, n_units, n_start_pts_per_inpCond]
        # sampling_loc_props: (dict) with properties of all sampling locations
        # all_freq_ids: [n_trials, 1], all frequency IDs of each trial
        # conditionIds = [1, n_trials], context ID per trial
        # coherencies_trial = [nIntegrators, n_trials], input coherencies of sensory input 1 and 2 over trials
        # network_type: (str), 'swg' or 'ctxt'

        # constants
        if network_type == 'swg':
            nPCs = 3
        elif network_type == 'ctxt':
            nPCs = 2
        else:
            raise Exception(
                "Network type unknown, please set network_type to 'swg' or 'ctxt'")

        # get projections
        PCs = self.get_PCs_of_network_activity(network_activity)
        proj_sampling_locs = self.project_sampling_locs_onto_PCs(sampling_locs, PCs[:,0:nPCs])
        proj_condAvgTrajs = self.project_condition_average_trajectories_onto_PCs(
            network_activity, PCs[:, 0:nPCs], network_type, sampling_loc_props, all_freq_ids,
            conditionIds, coherencies_trial)

        # constants
        n_inpConds = np.shape(proj_sampling_locs)[0]

        # normalize item number values to colormap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=n_inpConds)

        fig = plt.figure(figsize=[13, 13])
        if network_type == 'swg':
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
        plt.grid()
        # plot condition average trajectories
        self.plot_condAvgTrajs(ax, proj_condAvgTrajs, norm, network_type, 1,'-')
        # plot sampling locations
        self.plot_sampling_locs(ax, proj_sampling_locs, network_type)
        ax.set_title("sampling locations on condition average trajectories",
                     fontsize=self.my_fontsize)
        ax.set_xlabel("PC(X)$_1$", fontsize=self.my_fontsize)
        ax.set_ylabel("PC(X)$_2$", fontsize=self.my_fontsize)
        if network_type == 'swg':
            ax.set_zlabel("PC(X)$_3$", fontsize=self.my_fontsize)
        ax.tick_params(axis='both', which='major',
                       labelsize=self.my_fontsize - 15)
        plt.show()
        return fig, ax

    def plot_full_and_reduced_rank_condAvgTrajs(self, network_activity_fr, network_activity_rr,
                                                sampling_loc_props, all_freq_ids, conditionIds, coherencies_trial,
                                                network_type, rankW):
        # plot low-dimensional projection of condition average trajectories for full-rank (solid line=) and reduced-rank (dotted line) networks
        # network_activity_fr: [n_units, n_timesteps, n_trials], network activities of reduced-rank network
        # network_activity_rr: [n_units, n_timesteps, n_trials], network activities of reduced-rank network
        # sampling_loc_props: (dict) with properties of all sampling locations
        # all_freq_ids: [n_trials, 1], all frequency IDs of each trial
        # conditionIds = [1,n_trials], context ID per trial
        # coherencies_trial = [nIntegrators, n_trials], input coherencies of sensory input 1 and 2 over trials
        # network_type: (str), 'swg' or 'ctxt'

        # constants
        if network_type == 'swg':
            nPCs = 3
            n_inpConds = np.size(sampling_loc_props["freq_idx_per_inpCond"])
        elif network_type == 'ctxt':
            nPCs = 2
            n_inpConds = np.size(sampling_loc_props["ctxt_per_inpCond"])
        else:
            raise Exception(
                "Network type unknown, please set network_type to 'swg' or 'ctxt'")

        # normalize item number values to colormap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=n_inpConds)

        # get projections
        PCs = self.get_PCs_of_network_activity(network_activity_fr)
        proj_condAvgTrajs_fr = self.project_condition_average_trajectories_onto_PCs(
            network_activity_fr, PCs[:, 0:nPCs + 1], network_type, sampling_loc_props,
            all_freq_ids, conditionIds, coherencies_trial)
        proj_condAvgTrajs_rr = self.project_condition_average_trajectories_onto_PCs(
            network_activity_rr, PCs[:, 0:nPCs + 1], network_type, sampling_loc_props,
            all_freq_ids, conditionIds, coherencies_trial)

        # plot condition average trajectories
        fig = plt.figure(figsize=[13, 13])
        if network_type == 'swg':
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
        plt.grid()
        self.plot_condAvgTrajs(ax, proj_condAvgTrajs_fr, norm, network_type, 1,'-')
        self.plot_condAvgTrajs(ax, proj_condAvgTrajs_rr, norm, network_type, 0,':')
        ax.set_title("condition average trajectories (solid lines: W; dotted lines: W$^{OD}_{k="
                     + str(rankW) + "}$)", fontsize=self.my_fontsize)
        ax.set_xlabel("PC(X)$_1$", fontsize=self.my_fontsize)
        ax.set_ylabel("PC(X)$_2$", fontsize=self.my_fontsize)
        if network_type == 'swg':
            ax.set_zlabel("PC(X)$_3$", fontsize=self.my_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=self.my_fontsize - 15)

        return fig, ax
