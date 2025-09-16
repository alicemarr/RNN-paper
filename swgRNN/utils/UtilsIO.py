import numpy as np
import h5py


class UtilsIO(object):
    # class to collect helper functions to deal import/export
    def load_weights(self, path_to_weights, net_id):
        # load network weights
        # path_to_weights: (str), path to hdf5-file with stored weight matrices
        # net_id: (integer), number of network in hdf5-file to load

        # n_Wru_v: [n_units, n_inputs], input weights
        # n_Wrr_n: [n_units, n_units], recurrent weights
        # m_Wzr_n: [n_outputs, n_inputs], output weights
        # n_x0_c: [n_units, n_contexts], initial conditions per context
        # n_bx_1: [n_units, 1], bias of hidden units
        # m_bz_1: [n_outputs, 1], bias of output units

        name_dataset = '/NetNr' + str(net_id) + '/final'

        f = h5py.File(path_to_weights, 'r')
        n_Wru_v = np.asarray(f[name_dataset + '/n_Wru_v']).T
        n_Wrr_n = np.asarray(f[name_dataset + '/n_Wrr_n']).T
        m_Wzr_n = np.asarray(f[name_dataset + '/m_Wzr_n']).T
        n_x0_c = np.asarray(f[name_dataset + '/n_x0_c']).T
        n_bx_1 = np.asarray(f[name_dataset + '/n_bx_1']).T
        m_bz_1 = np.asarray(f[name_dataset + '/m_bz_1']).T
        f.close()

        return n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1

    def save_to_hdf5(self, outputfilename, group_name, my_data):
        # save all fields of data structure my_data to hdf5
        # outputfilename: (str), name of hdf5-file and path to it
        # group_name: (str), name of hdf5-group
        # my_data: (dict), contains data to store in fields

        with h5py.File(outputfilename, "a") as f:
            for key, value in my_data.items():
                if key == 'local_op_dims':  # flip s.t. consitent with h5-matlab version
                    value = value.T
                dset = f.create_dataset(group_name + '/' + key,
                                        data=np.asarray(value))
