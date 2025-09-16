import numpy as np


def make_unit_length(a):
    #  Take vector a and make unit length by dividing by its norm
    normed_a = a / np.linalg.norm(a);
    return normed_a
