import numpy as np
import copy

from utils.run_one_forwardPass import run_one_forwardPass
from utils.make_unit_length import make_unit_length
from utils.remove_dimension_from_weight_matrix import remove_dimension_from_weight_matrix
from utils.get_state_distance_between_trajs import get_state_distance_between_trajs


def get_neg_deltaFF(x0, dims_to_be_orth, samplingLocParams, n_Wru_v, n_Wrr_n, 
			m_Wzr_n, n_bx_1, m_bz_1, dim_type, network_type):
    # calculate impact of removing dimension x0 from W as euclidean
    # distance between x_t and ^x_t.

    # x0: [n_units, 1], dimension to remove from W
    # dims_to_be_orth: [n_units, n_dims], set of dimensions to which x0 has to be orthogonal
    # samplingLocParams: [structure], parameters of sampling location which is currently tested
    # n_Wru_v, n_Wrr_n, m_Wzr_n, n_x0_c, n_bx_1, m_bz_1: network weights
    # dim_type: (str), 'columns' or 'rows', decide which dimension type should be removed from W
    # network_type: (str), 'swg' or 'ctxt'

    n_units = np.size(x0)
    n_inputs = np.shape(n_Wru_v)[1]

    # make x0 orthogonal to prev found dimensions and unit length
    x0 = np.reshape(make_unit_length(x0), [n_units, 1])
    Q, _ = np.linalg.qr(np.concatenate([dims_to_be_orth, x0], axis=1))
    x0 = Q[:, -1]
    x0 = np.reshape(make_unit_length(x0), [n_units, 1])

    n_Wrr_n_modified = np.asarray(copy.deepcopy(n_Wrr_n))
    n_Wrr_n_modified = remove_dimension_from_weight_matrix(n_Wrr_n_modified, x0, dim_type)

    # run network at sampling location and collect trajectories
    inputs_relax = np.zeros([n_inputs, 1, 1])
    if network_type == 'swg':
        conditionIds_relax = np.ones([1, 1])
        init_n_x0_c = np.reshape(np.asarray(samplingLocParams["sampling_loc"]), [n_units, 1])
    elif network_type == 'ctxt':
        ctxt_id = int(samplingLocParams["ctxt_id"])
        inputs_relax[1 + ctxt_id, :, :] = 0;
        conditionIds_relax = np.reshape(np.asarray([ctxt_id, ctxt_id]), [1, 2])
        init_n_x0_c = np.concatenate([samplingLocParams["sampling_loc"],
                                      samplingLocParams["sampling_loc"]],
                                     axis=1)
    else:
        raise Exception("Network type unknown, please set network_type to 'swg' or 'ctxt'")

    net_noise_trajs = 0;
    my_seed = None
    forwardPass_modified = run_one_forwardPass(n_Wru_v, n_Wrr_n_modified, m_Wzr_n,
                                               init_n_x0_c, n_bx_1, m_bz_1,
                                               inputs_relax, conditionIds_relax,
                                               my_seed, net_noise_trajs)

    # collect state distance between trajectories
    trajs_orgWrr = np.reshape(samplingLocParams["all_trajs_org"][:, 1], [n_units, 1, 1])
    trajs_modWrr = np.reshape(forwardPass_modified["n_x_t"], [n_units, 1, 1])
    state_dist_to_org_net = np.squeeze(get_state_distance_between_trajs(trajs_orgWrr, trajs_modWrr))
    neg_deltaFF = state_dist_to_org_net * -1;  # inverse to find maxima with fminunc method...
    return neg_deltaFF
