import numpy as np
import chaospy as ch


def sense_main(uhat, expansion, joint):

    dim = len(joint)
    s1 = np.zeros(dim)
    exponents = ch.lead_exponent(expansion, graded=True)

    variance = np.sum(np.array(uhat[1:]) ** 2)

    for variable, name in enumerate(expansion.names):
        mask = np.ones(dim)
        mask[variable] = False

        for idx, exponent in enumerate(exponents):
            if exponent[variable] > 0 and np.all(exponent * mask == 0):
                s1[variable] += uhat[idx] ** 2

    s1 = s1 / variance

    return s1


def sense_t(uhat, expansion, joint):

    dim = len(joint)
    st = np.zeros(dim)
    exponents = ch.lead_exponent(expansion, graded=True)

    variance = np.sum(np.array(uhat[1:]) ** 2)

    for variable, name in enumerate(expansion.names):

        mask = np.ones(dim)
        mask[variable] = False

        for idx, exponent in enumerate(exponents):
            if exponent[variable] > 0 and np.all(exponent * mask == 0):
                st[variable] += uhat[idx] ** 2

            if exponent[variable] > 0 and np.any(exponent * mask != 0):
                st[variable] += uhat[idx] ** 2

    st = st / variance

    return st
