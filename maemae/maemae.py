#!/usr/bin/env python3
from functools import partial
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

from jax import jit
from jax import numpy as jnp

# Conversion factor from cycles per day (cpd) to microhertz (μHz)
CPD_TO_MICROHZ = 1/86400 * 1e6

@partial(jit, static_argnums=(2,))
def gold_deconvolution(y, kernel, n_iterations=100,):
    '''
    gold algorithm deconvlution, Morhac (2003), matrix multiplication.
    '''

    L = kernel.shape[0] # length of kernel
    N = y.shape[0] # length of output data
    
    x0 = jnp.copy(y) # length of input data

    ks = 0
    kc = kernel.shape[0]//2
    kn = kernel.shape[0]

    H = jnp.zeros((N, N))
    for i in range(N):
        i_start = max(0, i-(kc-ks))
        i_end = min(H.shape[1], i + kn - kc)
        H = H.at[i, i_start:i_end].set(kernel[(kc-(i-i_start)):(kc+(i_end-i))])

    # gold algorithm
    # Hp = H.T @ H @ H.T @ H
    # yp = H.T @ H @ H.T @ y
    Hp = H.T @ H @ H.T @ H
    yp = H.T @ H @ H.T @ y

    for i in range(n_iterations):
        x0 = jnp.nan_to_num(x0 * yp / (Hp @ x0))

    return x0

@partial(jit, static_argnums=(2,))
def opt_gold_deconvolution(y, kernel, n_iterations=1000,):
    '''
    gold algorithm deconvolution, Morhac (2003), use vector to speed up.
    '''

    # L = kernel.shape[0] # length of kernel
    # N = y.shape[0] # length of input data
    # M = N + L - 1 # length of output data

    x0 = jnp.copy(y) # length of input data

    h = kernel
    vector_B = jnp.correlate(h, h, mode='same')
    vector_c = jnp.convolve(vector_B, vector_B, mode='same')

    vector_p = jnp.correlate(y, h, mode='same')
    vector_yp = jnp.correlate(vector_p, vector_B, mode='same')

    def _body(i, x0):
        vector_z = jnp.correlate(x0, vector_c, mode='same')
        return x0 * vector_yp / vector_z

    return jax.lax.fori_loop(0, n_iterations, _body, x0)

@partial(jit, static_argnums=(2,3))
def richardson_lucy_deconvolution(y, kernel, n_iterations=1000, γ=1):
    '''
    richardson lucy deconvolution, Morháč & Matoušek (2011)
    '''

    # L = kernel.shape[0] # length of kernel
    # N = y.shape[0] # length of input data
    # M = N + L - 1 # length of output data

    x0 = jnp.copy(y) # length of input data

    def _body(i, x0):
        # den = jnp.correlate(x0, kernel, mode='same')
        return x0 * jnp.correlate(y / jnp.correlate(x0, kernel, mode='same') ** γ, kernel, mode='same')

    return jax.lax.fori_loop(0, n_iterations, _body, x0)

@partial(jit, static_argnums=(2,))
def chi2_deconvolution(y, kernel, n_iterations=1000):
    '''
    deconvolution assuming the noise model is distributed as χ^2 with dof = 2.
    '''

    # L = kernel.shape[0] # length of kernel
    # N = y.shape[0] # length of input data
    # M = N + L - 1 # length of output data

    x0 = jnp.copy(y) # length of input data

    h = kernel

    for _ in range(n_iterations):
        num = jnp.correlate(y / jnp.correlate(x0, kernel, mode='same')**2., kernel, mode='same')
        den = jnp.correlate(1 / jnp.correlate(x0, kernel, mode='same'), kernel, mode='same')
        x0 = x0 * num / den

    return x0

@partial(jit, static_argnums=(2,))
def maximum_a_posteriori_deconvolution(y, kernel, n_iterations=1000,):
    '''
    richardson lucy deconvolution, V. Matousek & M. Morhac (2014)
    '''

    # L = kernel.shape[0] # length of kernel
    # N = y.shape[0] # length of input data
    # M = N + L - 1 # length of output data

    x0 = jnp.copy(y) # length of input data

    for _ in range(n_iterations):
        # den = jnp.correlate(x0, kernel, mode='same')
        x0 = x0 * jnp.exp( jnp.correlate(y / jnp.correlate(x0, kernel, mode='same') - 1, kernel, mode='same') )

    return x0
