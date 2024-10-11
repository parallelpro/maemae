import numpy as np

# Conversion factor from cycles per day (cpd) to microhertz (μHz)
CPD_TO_MICROHZ = 1/86400 * 1e6


def gold_deconvolution(y, kernel, n_iterations=100,):
    '''
    gold algorithm deconvlution, Morhac (2003), matrix multiplication.
    '''

    L = kernel.shape[0] # length of kernel
    N = y.shape[0] # length of output data
    
    x0 = np.copy(y) # length of input data

    ks = 0
    kc = kernel.shape[0]//2
    kn = kernel.shape[0]

    H = np.zeros((N, N))
    for i in range(N):
        i_start = max(0, i-(kc-ks))
        i_end = min(H.shape[1], i + kn - kc)
        H[i, i_start:i_end] = kernel[(kc-(i-i_start)):(kc+(i_end-i))]

    # gold algorithm
    # Hp = H.T @ H @ H.T @ H
    # yp = H.T @ H @ H.T @ y
    Hp = H.T @ H @ H.T @ H
    yp = H.T @ H @ H.T @ y

    for i in range(n_iterations):
        x0 = x0 * yp / (Hp @ x0)

    return x0


def opt_gold_deconvolution(y, kernel, n_iterations=1000,):
    '''
    gold algorithm deconvlution, Morhac (2003), use vector to speed up.
    '''

    # L = kernel.shape[0] # length of kernel
    # N = y.shape[0] # length of input data
    # M = N + L - 1 # length of output data

    x0 = np.copy(y) # length of input data

    h = kernel
    vector_B = np.correlate(h, h, mode='same')
    vector_c = np.convolve(vector_B, vector_B, mode='full')

    vector_p = np.correlate(y, h, mode='full')
    vector_yp = np.correlate(vector_p, vector_B, mode='valid')

    for _ in range(n_iterations):
        vector_z = np.correlate(x0, vector_c, mode='same')
        x0 = x0 * vector_yp / vector_z
        
    return x0

def richardson_lucy_deconvolution(y, kernel, n_iterations=1000, γ=1):
    '''
    richardson lucy deconvolution, Morháč & Matoušek (2011)
    '''

    # L = kernel.shape[0] # length of kernel
    # N = y.shape[0] # length of input data
    # M = N + L - 1 # length of output data

    x0 = np.copy(y) # length of input data

    for _ in range(n_iterations):
        # den = np.correlate(x0, kernel, mode='same')
        x0 = x0 * np.correlate(y / np.correlate(x0, kernel, mode='same') ** γ, kernel, mode='same')

    return x0


def chi2_deconvolution(y, kernel, n_iterations=1000):
    '''
    deconvolution assuming the noise model is distributed as χ^2 with dof = 2.
    '''

    # L = kernel.shape[0] # length of kernel
    # N = y.shape[0] # length of input data
    # M = N + L - 1 # length of output data

    x0 = np.copy(y) # length of input data

    h = kernel

    for _ in range(n_iterations):
        num = np.correlate(y / np.correlate(x0, kernel, mode='same')**2., kernel, mode='same')
        den = np.correlate(1 / np.correlate(x0, kernel, mode='same'), kernel, mode='same')
        x0 = x0 * num / den

    return x0

def maximum_a_posteriori_deconvolution(y, kernel, n_iterations=1000,):
    '''
    richardson lucy deconvolution, V. Matousek & M. Morhac (2014)
    '''

    # L = kernel.shape[0] # length of kernel
    # N = y.shape[0] # length of input data
    # M = N + L - 1 # length of output data

    x0 = np.copy(y) # length of input data

    for _ in range(n_iterations):
        # den = np.correlate(x0, kernel, mode='same')
        x0 = x0 * np.exp( np.correlate(y / np.correlate(x0, kernel, mode='same') - 1, kernel, mode='same') )

    return x0
