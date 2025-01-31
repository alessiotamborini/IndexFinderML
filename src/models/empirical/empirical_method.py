import numpy as np

from matplotlib import pyplot as plt

def empirical_method(wvf):
    """ The empirical method for calculating the fiducial points. This method will follow
    the steps outlined in paper 'Underestimation of Vasodilator effects of nitroglycerin
    by upper limb blood pressure measurements' by J. R. Wilkins, et al. (1983) for the 
    identification of the inflection point. 
    
    """

    # calculate the waveform derivatives
    wvf_d1 = np.gradient(wvf)
    wvf_d2 = np.gradient(wvf_d1)
    wvf_d3 = np.gradient(wvf_d2)
    wvf_d4 = np.gradient(wvf_d3)

    # find the max value index
    max_idx = np.argmax(wvf)

    # Determine the slope of the fourth derivative at the max value index
    slope_d4_at_max = wvf_d4[max_idx]

    if slope_d4_at_max > 0:
        # Late systolic peak (p2_ind)
        p2_ind = max_idx
        # Find p1_ind before p2_ind using the second zero crossing from positive to negative
        zero_crossings = np.where(np.diff(np.sign(wvf_d4)))[0]
        zero_crossings_before_p2_ind = zero_crossings[zero_crossings < p2_ind]
        if len(zero_crossings_before_p2_ind) >= 2:
            p1_ind = zero_crossings_before_p2_ind[-2]
        else:
            p1_ind = None  # Handle case where there are not enough zero crossings
    else:
        # Early systolic peak (p1_ind)
        p1_ind = max_idx
        # Find p2_ind after p1_ind using the third zero crossing from negative to positive
        zero_crossings = np.where(np.diff(np.sign(wvf_d4)))[0]
        zero_crossings_after_p1_ind = zero_crossings[zero_crossings > p1_ind]
        if len(zero_crossings_after_p1_ind) >= 3:
            p2_ind = zero_crossings_after_p1_ind[2]
        else:
            p2_ind = None  # Handle case where there are not enough zero crossings

    
    # find the dicrotic notch as the first peak on the second derivative after the maximum negative first derivative
    d1_argmin = np.argmin(wvf_d1)                           # arg minimum of the first derivative
    dn_idx = np.argmax(wvf_d2[d1_argmin:]) + d1_argmin

    # debug plot of wvf, wvf_f1, and wvf_d2
    # plt.subplot(311), plt.plot(wvf)
    # plt.subplot(312), plt.plot(wvf_d1)
    # plt.subplot(313), plt.plot(wvf_d2)
    # plt.show()

    # # debug plot - fiducial points
    # plt.plot(wvf)
    # if p1_ind is not None:
    #     plt.plot(p1_ind, wvf[p1_ind], 'go')
    # if p2_ind is not None:
    #     plt.plot(p2_ind, wvf[p2_ind], 'bo')
    # plt.plot(dn_idx, wvf[dn_idx], 'mo')
    # plt.tight_layout()
    # plt.show()

    return p1_ind, p2_ind, dn_idx