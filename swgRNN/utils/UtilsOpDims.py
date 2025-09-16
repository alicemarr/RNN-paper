import numpy as np
import copy
import h5py


class UtilsOpDims(object):
    # class to collect helper functions to deal with operative dimensions
    def load_local_op_dims(self, inputfilename, n_units, sampling_loc_props, network_type):
        # load local operative dimensions from file
        # inputfilename: (str), path to hdf5-file with local opeartive dimensions
        # n_units: (int), number of hidden units
        # sampling_loc_props: (dict) with properties of all sampling locations
        # network_type: (str), 'swg' or 'ctxt'

        # all_local_op_dims: [n_locs, n_op_dims, n_units], all local operative dimensions at sampling locations
        # all_fvals: [n_locs, n_op_dims], delta f for all local operative dimensions at all sampling locations

        # constants
        n_dims_to_load = n_units
        n_locs = np.size(sampling_loc_props["t_start_pt_per_loc"])

        # load
        all_local_op_dims = np.full([n_locs, n_dims_to_load, n_units], np.nan)
        all_fvals = np.full([n_locs, n_dims_to_load], np.nan)
        for loc_nr in range(n_locs):
            # find dimension name
            if network_type == 'swg':
                freq_id = sampling_loc_props["freq_idx_per_loc"][0, loc_nr]
                ctxt_id = []
                signCoh1 = []
                signCoh2 = []
            elif network_type == 'ctxt':
                ctxt_id = sampling_loc_props["ctxt_per_loc"][0, loc_nr]
                signCoh1 = sampling_loc_props["signCoh1_per_loc"][0, loc_nr]
                signCoh2 = sampling_loc_props["signCoh2_per_loc"][0, loc_nr]
                freq_id = []
            else:
                raise Exception(
                    "Network type unknown, please set network_type to 'swg' or 'ctxt'")

            t_start_pt = sampling_loc_props["t_start_pt_per_loc"][0, loc_nr]
            dim_name = self.get_name_of_local_operative_dims(network_type, t_start_pt,
                                                             ctxt_id, signCoh1, signCoh2, freq_id)

            # load
            f = h5py.File(inputfilename, 'r')
            fvals = np.asarray(f[dim_name + '/all_fvals'])
            local_op_dims = np.asarray(f[dim_name + '/local_op_dims']).T

            # ensure correct formating
            for dim_nr in range(n_dims_to_load):
                all_local_op_dims[loc_nr, dim_nr, :] = np.squeeze(local_op_dims[dim_nr, :])
                all_fvals[loc_nr, dim_nr] = np.squeeze(fvals)[dim_nr]

        return all_local_op_dims, all_fvals

    def get_name_of_local_operative_dims(self, network_type, t_start_point, ctxt_id=None,
                                         signCoh1=None, signCoh2=None, freq_id=None):
        # get name of local operative dimension (dim_name) based on properties of
        # sampling location
        # network_type: (str), 'swg' or 'ctxt'
        # t_start_pt_per_loc: (int), t along trial in condition average trajectory
        # ctxt_id: (int), context ID ([] for swg network)
        # signCoh1: (int), sign of sensory input 1 ([] for swg network)
        # signCoh2: (int), sign of sensory input 2 ([] for swg network)
        # freq_id: (int), frequency ID ([] for ctxt network)

        # dim_name: (str), human interpretable name of current sampling location

        if network_type == 'swg':
            dim_name = 'opt_dims_t' + str(int(t_start_point)) + 'Freq' + str(int(freq_id))
        elif network_type == 'ctxt':
            dim_name = 'opt_dims_t' + str(int(t_start_point)) + 'Ctxt' + str(int(ctxt_id))\
                       + 'Inps' + str(int(signCoh1)) + '_' + str(int(signCoh2))
        else:
            raise Exception("Network type unknown, please set network_type to 'swg' or 'ctxt'")

        return dim_name

    def get_global_operative_dimensions(self, sampling_locs_to_combine, sampling_loc_props,
                                        all_local_op_dims, all_fvals_dims):
        # combine local operative dimension into global operative dimensions

        # sampling_locs_to_combine: (str), defines which subset of
        #       local operative dimensions should be considered to generate
        #       the global operative dimensions (see explanations on function-specific
        #       dimensions in paper)
        #       currently only implemented for ctxt-network
        # sampling_loc_props: (dict) with properties of all sampling locations
        # all_local_op_dims: [n_locs, n_op_dims, n_units], all local operative dimensions at sampling locations
        # all_fvals: [n_locs, n_op_dims], delta f for all local operative dimensions at all sampling locations

        # constants
        n_units = np.shape(all_local_op_dims)[2]
        n_locs_total = np.shape(all_fvals_dims)[0]

        # define which sampling locations to consider
        if sampling_locs_to_combine == 'all':
            loc_nrs = range(n_locs_total)
        elif sampling_locs_to_combine == 'ctxt1':
            loc_nrs = np.where(np.squeeze(sampling_loc_props["ctxt_per_loc"] == 1))
        elif sampling_locs_to_combine == 'ctxt2':
            loc_nrs = np.where(np.squeeze(sampling_loc_props["ctxt_per_loc"] == 2))
        elif sampling_locs_to_combine == 'allPosChoice':
            loc_nrs = np.where((np.squeeze(sampling_loc_props["ctxt_per_loc"] == 1) &
                                np.squeeze(sampling_loc_props["signCoh1_per_loc"] > 0))
                       | (np.squeeze(sampling_loc_props["ctxt_per_loc"] == 2) &
                          np.squeeze(sampling_loc_props["signCoh2_per_loc"] > 0)))
        elif sampling_locs_to_combine == 'allNegChoice':
            loc_nrs = np.where((np.squeeze(sampling_loc_props["ctxt_per_loc"] == 1) &
                                np.squeeze(sampling_loc_props["signCoh1_per_loc"] < 0))
                       | (np.squeeze(sampling_loc_props["ctxt_per_loc"] == 2) &
                          np.squeeze(sampling_loc_props["signCoh2_per_loc"] < 0)))
        else:
            raise Exception("sampling_locs_to_combine unknown. Please choose a valid option")
        n_locs = np.size(loc_nrs)

        # combine all considered sampling locations into one matrix
        # as locOpDim * localDeltaF, ...
        # then SVD(L)
        counter = -1
        L = np.zeros([n_units, n_locs * n_units])
        for loc_nr in loc_nrs:
            for dim_nr in range(n_units):
                counter += 1
                if not np.isnan(all_fvals_dims[loc_nr, dim_nr]):
                    L[:, counter] = np.squeeze(all_local_op_dims[loc_nr, dim_nr, :]
                                               * all_fvals_dims[loc_nr, dim_nr])

        L[np.isnan(L)] = 0  # replace nan with 0 before SVD
        [all_lSV, all_SVals, _] = np.linalg.svd(L);

        return all_lSV, all_SVals
