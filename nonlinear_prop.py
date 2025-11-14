import numpy as np
import jax
import jax.numpy as jnp
import functools
jax.config.update('jax_enable_x64', True)
import discretization as disc

# continuous dynamics
def P_dot(params, t, x, k, uk, sigma):
    a, b = disc.get_alphas(params, t, k)
    u = a * uk[:, k] + b * uk[:, k + 1]
    return sigma * disc.E_f(x, u).reshape((-1,))

# rk4 single step function
def rk41(tk, xk, dt, k, uk, sigmak, params):
    k1 = P_dot(params, tk, xk,k, uk, sigmak)
    k2 = P_dot(params, tk + 0.5 * dt,  xk + 0.5 * dt*k1,  k, uk, sigmak)
    k3 = P_dot(params, tk + 0.5 * dt,  xk + 0.5 * dt*k2,  k, uk, sigmak)
    k4 = P_dot(params, tk + dt, xk + dt * k3, k, uk, sigmak)
    return xk + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

P_dot_jit = jax.jit(P_dot)
rk41_jit  = jax.jit(rk41)

def propagate_nonlin(params, x_opt, u_opt, sigma_opt):
    n    = params['n']
    nk   = params['nk']
    dt   = params['dt']
    nsub = params['nsub']
    dts  = dt / (nsub + 1)

    N = (nk - 1) * (nsub + 1) + 1

    X = jnp.zeros((n, N))
    T = jnp.zeros((N,))

    x     = jnp.asarray(x_opt[:, 0]).reshape((-1,))
    uk    = jnp.asarray(u_opt)
    sigma = jnp.asarray(sigma_opt)


    X = X.at[:, 0].set(x)
    T = T.at[0].set(0.0)

    idx = 0
    for i in range(nk - 1):
        for j in range(nsub + 1):
            t = i * dt + j * dts
            x = rk41_jit(t, x, dts, i, uk, sigma, params)
            idx += 1
            X = X.at[:, idx].set(x)
            T = T.at[idx].set(t + dts)

    return np.asarray(X), np.asarray(T)