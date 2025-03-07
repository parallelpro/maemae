import numpy as np
import finufft

def window_function_brute(ωω, tt, TT):
    '''
    Brute-force evaluation of the spectral window function.
    Given a time series with midpoint times t_k and exposure
    times T_k, this is

    n = [\sum_k e^(i ω t_k) T_k sinc(ω T_k / 2)]/sqrt(π).

    We moreover divide this by d = sqrt(\sum_k T_k) so that
    the integral of |n/d|^2 with respect to ω is 1 by Parseval's Theorem.

    Inputs
    ------

    ωω: 1D ndarray of angular frequencies
    tt: 1D ndarray of midpoint timestamps
    TT: 1D ndarray of exposure times, or scalar if all identical.

    The units of ωω and tt should be such that ωω * tt
    is in units of radians.

    The units of TT and tt should be the same.

    Outputs
    -------
    W: Fourier transform of the window function, with unit Parseval
       normalisation (n / d in the above description)
    '''
    TT = TT * np.ones_like(tt)
    
    summand = np.exp(1j * ωω[:,None]  * tt[None, :]) * (TT[None,:] * np.sinc(ωω[:,None]/2/np.pi*TT[None,:]))
    s = np.sum(summand, axis=1)
    
    return s / (np.sqrt(np.pi) * np.sum(TT))

def window_function_nufft(N, δω, tt, TT):
    '''
    Fast evaluation of the spectral window function,
    using the Flatiron Nonuniform FFT package.

    Inputs
    ------

    N: number of frequency samples
    δω: Desired frequency sampling
    tt: 1D ndarray of midpoint timestamps
    TT: 1D ndarray of exposure times, or scalar if all identical.

    The units of δω and tt should be such that δω * tt
    is in units of radians.

    The units of TT and tt should be the same.

    Outputs
    -------
    ωω: Frequency samples
    W: Fourier transform of the window function,
       with unit Parseval normalisation
    '''
    
    # in finufft, the fourier components are indexed by simply integers.
    NN = np.arange(N) - np.ceil(N/2)
    ωω = NN * δω
    
    ones = np.ones_like(tt).astype(np.complex128)
    times = TT * ones
    x1 = (tt + TT/2) * δω
    x2 = (tt - TT/2) * δω
    
    tr1 = finufft.nufft1d1(x1, ones, n_modes=N)
    tr2 = finufft.nufft1d1(x2, ones, n_modes=N)
    s = np.where(ωω == 0, np.sum(times), (tr1 - tr2) / ωω / 1j)
    
    return ωω, s / np.sqrt(np.pi * np.sum(times))