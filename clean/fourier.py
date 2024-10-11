import numpy as np 
from astropy.timeseries import LombScargle

def fourier(x, y, dy=None, oversampling=1, freqMin=None, freqMax=None, freq=None, return_val="power"):
    """
    Calculate the power spectrum density for a discrete time series.
    https://en.wikipedia.org/wiki/Spectral_density


    Input:
    x: array-like[N,]
        The time array.

    y: array-like[N,]
        The flux array.


    Optional input:
    dy: float, or array-like[N,]
        errors on y
    
    oversampling: float, default: 1
        The oversampling factor to control the frequency grid.
        The larger the number, the denser the grid.

    freqMin: float, default: frequency resolution

    freqMax: float, default: nyquist frequency


    Output:
    freq: np.array
        The frequency, in unit of [x]^-1.

    psd: np.array
        The power spectrum density, in unit of [y]^2/[x].
        https://en.wikipedia.org/wiki/Spectral_density


    Examples:
    >>> ts = np.load("flux.npy")
    >>> t = ts["time_d"]   # the time in day
    >>> f = ts["flux_mf"]   # the relative flux fluctuated around 1
    >>> f = (f-1)*1e6   # units, from 1 to parts per million (ppm)

    >>> freq, psd = fourier(t, f, return_val="psd_new")
    >>> freq = freq/(24*3600)*1e6   # c/d to muHz
    >>> psd = psd*(24*3600)*1e-6   # ppm^2/(c/d) to ppm^2/muHz

    """

    if not (return_val in ["psd_old", "periodogram", "power", "amplitude", "psd", "window"]):
        raise ValueError("return_val should be one of ['psd_old', 'periodogram', 'power', 'amplitude', 'psd_new', 'window'] ")

    Nx = len(x)
    dx = np.median(x[1:]-x[:-1]) 
    fs = 1.0/dx
    Tobs = dx*len(x)
    fnyq = 0.5*fs
    dfreq = fs/Nx

    if freqMin is None: freqMin = dfreq
    if freqMax is None: freqMax = fnyq

    if freq is None: freq = np.arange(freqMin, freqMax, dfreq/oversampling)

    if dy is None:
        dy_scale_factor = 1
    else:
        w = dy**-2.0
        w /= np.sum(w)
        dy = w**-0.5
        dy_scale_factor = 1/np.sqrt(np.mean(dy**-2.0))
    
    if return_val == "psd_old":
        p = LombScargle(x, y, dy=dy).power(freq, normalization='psd')*dx*4. * dy_scale_factor**2.0
    if return_val == "periodogram":
        # text book def of periodogram
        # don't think seismologists use this
        p = LombScargle(x, y, dy=dy).power(freq, normalization='psd')
    if return_val == "power":
        # normalized according to kb95
        # a sine wave with amplitude A will have peak height of A^2 in the power spectrum.
        p = LombScargle(x, y, dy=dy).power(freq, normalization='psd')/Nx*4. * dy_scale_factor**2.0
    if return_val == "amplitude":
        # normalized according to kb95
        # a sine wave with amplitude A will have peak height of A in the amp spectrum.
        p = np.sqrt(LombScargle(x, y, dy=dy).power(freq, normalization='psd')/Nx*4.) * dy_scale_factor
    if return_val == "psd": 
        # normalized according to kb95
        # a sine wave with amplitude A will have peak height of A^2*Tobs in the power density spectrum.
        # should be equivalent to psd_old
        nu = 0.5*(freqMin+freqMax)
        freq_window = np.arange(freqMin, freqMax, dfreq/10)
        power_window = LombScargle(x, np.sin(2*np.pi*nu*x)).power(freq_window, normalization="psd")/Nx*4.
        Tobs = 1.0/np.sum(np.median(freq_window[1:]-freq_window[:-1])*power_window)
        p = (LombScargle(x, y, dy=dy).power(freq, normalization='psd')/Nx*4.) * dy_scale_factor**2.0 * Tobs
    if return_val == "window":
        # give spectral window
        nu = 0.5*(freqMin+freqMax)
        freq_window = np.arange(freqMin, freqMax, dfreq/oversampling)
        power_window1 = LombScargle(x, np.sin(2*np.pi*nu*x), dy=dy).power(freq_window, normalization="psd")/Nx*4. * dy_scale_factor**2.0
        power_window2 = LombScargle(x, np.cos(2*np.pi*nu*x), dy=dy).power(freq_window, normalization="psd")/Nx*4. * dy_scale_factor**2.0
        power_window = (power_window1 + power_window2)/2.
    
        freq, p = freq_window-nu, power_window

    return freq, p


def echelle(freq, ps, Dnu, fmin=None, fmax=None, echelletype="single", offset=0.0):
    '''
    Make an echelle plot used in asteroseismology.
    
    Input parameters
    ----
    freq: 1d array-like, freq
    ps: 1d array-like, power spectrum
    Dnu: float, length of each vertical stack (Dnu in a frequency echelle)
    fmin: float, minimum frequency to be plotted
    fmax: float, maximum frequency to be plotted
    echelletype: str, `single` or `replicated`
    offset: float, an amount by which the diagram is shifted horizontally
    
    Return
    ----
    z: a 2d numpy.array, folded power spectrum
    extent: a list, edges (left, right, bottom, top) 
    x: a 1d numpy.array, horizontal axis
    y: a 1d numpy.array, vertical axis
    
    Users can create an echelle diagram with the following command:
    ----
    
    import matplotlib.pyplot as plt
    z, ext = echelle(freq, power, Dnu, fmin=numax-4*Dnu, fmax=numax+4*Dnu)
    plt.imshow(z, extent=ext, aspect='auto', interpolation='nearest')
    
    '''
    
    if fmin is None: fmin=0.
    if fmax is None: fmax=np.nanmax(freq)

    fmin -= offset
    fmax -= offset
    freq -= offset

    fmin = 1e-4 if fmin<Dnu else fmin - (fmin % Dnu)

    # define plotting elements
    resolution = np.median(np.diff(freq))
    # number of vertical stacks
    n_stack = int((fmax-fmin)/Dnu) 
    # number of point per stack
    n_element = int(Dnu/resolution) 

    fstart = fmin - (fmin % Dnu)
    
    z = np.zeros([n_stack, n_element])
    base = np.linspace(0, Dnu, n_element) if echelletype=='single' else np.linspace(0, 2*Dnu, n_element)
    for istack in range(n_stack):
        z[-istack-1,:] = np.interp(fstart+istack*Dnu+base, freq, ps)
    
    extent = (0, Dnu, fstart, fstart+n_stack*Dnu) if echelletype=='single' else (0, 2*Dnu, fstart, fstart+n_stack*Dnu)
    
    x = base
    y = fstart + np.arange(0, n_stack+1, 1)*Dnu + Dnu/2
    
    return z, extent, x, y
