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

    # find the dicrotic notch as the first peak on the second derivative after the maximum negative first derivative
    # assume dicrotic notch can be max at 0.5*len(wvf) 
    max_loc = int(0.5*len(wvf)) 
    d1_argmin = np.argmin(wvf_d1)                                   # arg minimum of the first derivative
    max_loc = max_loc if max_loc>d1_argmin else d1_argmin + 200     # max_loc should be greater than d1_argmin
    dn_idx = np.argmax(wvf_d2[d1_argmin:max_loc]) + d1_argmin

    # find the max value index
    p_ind = np.argmax(wvf[:dn_idx])
    
    # Determine the slope of the fourth derivative at the max value index
    slope_d4_at_max = wvf_d4[p_ind]

    if slope_d4_at_max > 0: # Late systolic peak
        # Find p1_ind before p2_ind using the second zero crossing from positive to negative
        zero_crossings = np.where(np.diff(np.sign(wvf_d4)))[0]
        zero_crossings_before_p_ind = zero_crossings[zero_crossings < p_ind]
        if len(zero_crossings_before_p_ind) >= 2:
            i_ind = zero_crossings_before_p_ind[-2]
        else:
            i_ind = np.nan 
    else:       # Early systolic peak 
        # Find i_ind after p_ind using the third zero crossing from negative to positive
        zero_crossings = np.where(np.diff(np.sign(wvf_d4)))[0]
        zero_crossings_after_p_ind = zero_crossings[np.logical_and(zero_crossings > p_ind, zero_crossings < dn_idx)]
        if len(zero_crossings_after_p_ind) >= 3:
            i_ind = zero_crossings_after_p_ind[2]
        else:
            i_ind = np.nan

    return p_ind, i_ind, dn_idx