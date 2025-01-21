from scipy.signal import butter, filtfilt

def butterworth_filter(data, cutoff, fs, order, type_='low'):
    """
    Inputs:
        - data : (array : float) data to filter
        - cutoff : (float) frequency for filtering
        - fs : (float) sampling frequency in Hertz
        - order : (int) order of the butterworth filter
        - type_ : (string) type of butterworth filter
    Outputs:
        - data_lpf : (array : float) filtered data
    Functionality:
        - returns butterworth filtered of data
    """

    # Calculate low pass Filter of signal
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=type_)
    data_filtered = filtfilt(b, a, data, method='pad')
    return data_filtered