import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import marsgram_dens_lut_jax as dens
import functools

from params import params

# returns linear interpolation of a vector between two time steps 
def get_alphas(params, t, k):
    tk = params['tau'][k]
    tk1 = params['tau'][k + 1]

    alphak = (tk1 - t) / (tk1 - tk)
    betak = 1 - alphak

    return alphak, betak

def def_jacobian_funcs(params):

    def wrapped_dyn(zs, us):
        return system_dynamics(params,zs, us)

    dfdz_jit = jax.jit(jax.jacrev(wrapped_dyn, argnums=0))
    dfdu_jit = jax.jit(jax.jacrev(wrapped_dyn, argnums=1))

    f_jit = jax.jit(wrapped_dyn)

    A_f = lambda z, u: dfdz_jit(z, u)
    B_f = lambda z, u: dfdu_jit(z, u)
    E_f = lambda z, u: f_jit(z, u).reshape(params['n'], 1)

    return A_f, B_f, E_f

A_f, B_f, E_f = def_jacobian_funcs(params)

def nonlinear_aero(params, zs, us):
    # Extract states and controls
    r, _, _, v, _, _, _, _= zs

    # Compute altitude
    rdim = r*params['nd']
    hdim = rdim - params['re']
    
    rho = jnp.interp(hdim/1e3, dens.h_grid, dens.rho_vals)

    rho_s = rho / (params['nm'] / params['nd']**3)
    sref_s = params['sref'] / params['nd']**2
    bc_s = params['bc'] / (params['nm'] / (params['nd']**2))

    D    = 0.5 * (1 / bc_s) * rho_s * v**2
    L    = D * params['LD'] 

    alpha = 0

    return {'L': L, 'D': D, 'alpha': alpha, 'rho': rho}

def system_dynamics(params, zs, us):

    # Extract constant param values from struct
    Om = params['omega_s']
    Kg = params['kg']

    # Extract states
    r, theta, phi, v, gamma, psi, _, _ = zs

    # Determine lift and drag coefficients from velocity
    aero = nonlinear_aero(params, zs, us)
    L    = aero['L']
    D    = aero['D']

    # Extract bank angle
    sigma   = us[0]
    # alpha   = us[1]

    # Extract sines and cosines of various values
    cp  = jnp.cos(phi)
    sp  = jnp.sin(phi)
    tp  = jnp.tan(phi)
    cg  = jnp.cos(gamma)
    sg  = jnp.sin(gamma)
    tg  = jnp.tan(gamma)
    cps = jnp.cos(psi)
    sps = jnp.sin(psi)

    cs  = jnp.cos(sigma)
    ss  = jnp.sin(sigma)

    nv = params['nv']
    
    # state derivative function
    xDot = jnp.array([
        v * sg,
        v * cg * sps / (r * cp),
        v * cg * cps / r, 
        - D - Kg * sg / r**2 + Om**2 * r * cp * (sg * cp - cg * sp * cps),
        (1 / v) * ( L * cs + (v**2 - Kg / r) * cg / r ) + 2 * Om * cp * sps + Om**2 * r * (1 / v) * cp * (cg * cp + sg * cps * sp),
        (1 / v) * ( L * ss / cg + v**2 * cg * sps * tp / r ) - 2 * Om * (tg * cps * cp - sp) + Om**2 * r * (1 / (v * cg)) * sps * sp * cp,

        # augmented ctcs constraint states
        (jnp.maximum(0.0, (1 / params['Qmax']) * params['kQ'] * jnp.sqrt(aero['rho']) * (v * nv) ** 3 - 1)),
        (jnp.maximum(0.0, (1 / params['qmax']) * 0.5 * aero['rho'] * (v * nv) ** 2 - 1))
    ])

    return xDot

# derivative function for rk4
def P_dot(params, t, X, k, uk_jax, sigmak_jax):

    phi_a_idx   = params['n']
    phi_b_m_idx = phi_a_idx   + params['n'] * params['n']
    phi_b_p_idx = phi_b_m_idx + params['n'] * params['m']
    phi_s_idx   = phi_b_p_idx + params['n'] * params['m']

    phi_a   = X[phi_a_idx   : phi_b_m_idx].reshape((params['n'], params['n']))
    phi_b_m = X[phi_b_m_idx : phi_b_p_idx].reshape((params['n'], params['m']))
    phi_b_p = X[phi_b_p_idx : phi_s_idx  ].reshape((params['n'], params['m']))
    phi_s   = X[phi_s_idx   :            ].reshape((params['n'],           1))

    # get FOH control
    alphak, betak = get_alphas(params, t, k)
    u = (alphak * uk_jax[:, k] + betak * uk_jax[:, k+1]).reshape((params['m'], 1))

    X_flat = X[:params['n']].flatten()
    u_flat = u.flatten()

    sigma = sigmak_jax

    P1 = (sigma * E_f(X_flat, u_flat))
    P2 = (sigma * A_f(X_flat, u_flat) @ phi_a).reshape((params['nsq'], 1))
    P3 = (sigma * A_f(X_flat, u_flat) @ phi_b_m + sigma * B_f(X_flat, u_flat) * alphak).reshape((params['nxm'], 1))
    P4 = (sigma * A_f(X_flat, u_flat) @ phi_b_p + sigma * B_f(X_flat, u_flat) * betak).reshape((params['nxm'], 1))
    P5 = (sigma * A_f(X_flat, u_flat) @ phi_s   + E_f(X_flat, u_flat))

    return jnp.vstack([P1, P2, P3, P4, P5])

# rk4 single step function
def rk41(tk, xk, dt, k, uk_jax, sigmak_jax, params):
    k1 = P_dot(params, tk, xk, k, uk_jax, sigmak_jax)
    k2 = P_dot(params, tk + dt / 2, xk + (dt / 2) * k1, k, uk_jax, sigmak_jax)
    k3 = P_dot(params, tk + dt / 2, xk + (dt / 2) * k2, k, uk_jax, sigmak_jax)
    k4 = P_dot(params, tk + dt, xk + dt * k3, k, uk_jax, sigmak_jax)
    return xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

rk41_jit = jax.jit(functools.partial(rk41, params=params))

# discretization using multiple shooting
def discretize(xk_jax, uk_jax, sigmak_jax, params):
    dt_sub = params['dt'] / (params['nsub'] + 1)

    n, m, nk = params['n'], params['m'], params['nk']

    P = jnp.zeros((params['end_idx'], 1))

    x_prop_jax = jnp.zeros((n, nk - 1))
    A_jax      = jnp.zeros((n, n * (nk - 1)))
    B_jax      = jnp.zeros((n, m * (nk - 1)))
    C_jax      = jnp.zeros((n, m * (nk - 1)))
    S_jax      = jnp.zeros((n, nk - 1))

    x_prop = jnp.zeros((n, 1))
    phi_a = jnp.zeros((n, n))
    phi_b_m = jnp.zeros((n, m))
    phi_b_p = jnp.zeros((n, m))
    phi_s = jnp.zeros((n, 1))

    for i in range(0, nk - 1):
        P = P.at[:params['phi_a_idx'],                          0].set(xk_jax[:, i])
        P = P.at[params['phi_a_idx'] :   params['phi_b_m_idx'], 0].set(jnp.eye(n).reshape(params['nsq']))
        P = P.at[params['phi_b_m_idx'] : params['phi_b_p_idx'], 0].set(jnp.zeros(params['nxm']))
        P = P.at[params['phi_b_p_idx'] : params['phi_s_idx'],   0].set(jnp.zeros(params['nxm']))
        P = P.at[params['phi_s_idx'] :,                         0].set(jnp.zeros(n))

        for j in range(0,  params['nsub'] + 1):
            sub_time = i * params['dt'] + j * dt_sub
            P = rk41_jit(sub_time, P, dt_sub, i, uk_jax, sigmak_jax)

        x_prop  = P[                      :   params['phi_a_idx']].reshape((n, 1))
        phi_a   = P[params['phi_a_idx']   : params['phi_b_m_idx']].reshape((n, n))
        phi_b_m = P[params['phi_b_m_idx'] : params['phi_b_p_idx']].reshape((n, m))
        phi_b_p = P[params['phi_b_p_idx'] :   params['phi_s_idx']].reshape((n, m))
        phi_s   = P[params['phi_s_idx']:                         ].reshape((n, 1))

        indx1 = i * n
        indx2 = i * m

        x_prop_jax = x_prop_jax.at[:, i].set(x_prop.flatten())
        A_jax      = A_jax.at[:, indx1:indx1 + n].set(phi_a)
        B_jax      = B_jax.at[:, indx2:indx2 + m].set(phi_b_m)
        C_jax      = C_jax.at[:, indx2:indx2 + m].set(phi_b_p)
        S_jax      = S_jax.at[:, i].set(phi_s.flatten())

    return x_prop_jax, A_jax, B_jax, C_jax, S_jax