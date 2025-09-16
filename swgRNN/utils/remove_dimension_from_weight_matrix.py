import numpy as np
import copy


def remove_dimension_from_weight_matrix(W, all_dims_to_remove, dim_type):
    # remove a given dimension from matrix W, assuming W has
    # dimensions as columns or rows (specificed by dim_type ('columns' or 'rows')
    # W: [n_units, n_units], weight matrix from which dimensions will be removed
    # all_dims_to_remove: [n_units, n_dims], dimensions as columns, number of columns = num dims to remove
    # dim_type: (str), 'columns' or 'rows'

    # W_minusDims: [n_units, n_units], reduced-rank weight matrix W

    if dim_type == 'rows':
        W_modified = copy.deepcopy(W).T
    else:
        W_modified = copy.deepcopy(W)

    n_dims_total = np.shape(W_modified)[1]
    n_dims_to_remove = np.shape(all_dims_to_remove)[1]

    # remove dims from columns of W
    # project every column of W onto dim_to_remove to obtain
    # scaling_factor (proj_onto_dim_to_remove)
    # then remove this dim_to_remove*scaling_factor from each column of W
    for dim_nr_to_remove in range(n_dims_to_remove):
        for dim_nr in range(n_dims_total):
            dim_to_remove = all_dims_to_remove[:, dim_nr_to_remove]
            proj_onto_dim_to_remove = np.dot(W_modified[:, dim_nr], dim_to_remove)
            W_modified[:, dim_nr] = W_modified[:,dim_nr] - proj_onto_dim_to_remove * dim_to_remove

    if dim_type == 'rows':
        W_modified = W_modified.T

    return W_modified
