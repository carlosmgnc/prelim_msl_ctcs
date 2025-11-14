import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

params = {}

# dimensioned vehicle parameters
params['mass']     = 2900
params['re']       = 3396190
params['mars_rot'] = 0
params['omega']    = 7.088e-5 * params['mars_rot']
params['ge']       = 3.7132
params['mue']      = 4.2828e13
params['sref']     = 15.9

params['kQ'] = 9.4369*10**(-5)
params['kq'] = params['ge'] * params['re'] / 2

params['Qmax'] = 340 * 10**3
params['qmax'] = 12 * 10**3

# new, specific to msl
params['LD'] = 0.24
params['bc'] = 120

# nondim scaling factors
params['nt']      = (params['re'] / params['ge'])**0.5
params['nt_inv']  = 1.0 / params['nt']
params['nd']      = params['re']
params['nv']      = (params['re'] * params['ge'])**0.5
params['na']      = params['ge']
params['nm']      = params['mass']
params['nm_dot']  = params['nm'] / params['nt']
params['nf']      = params['nm'] * params['na']

params['kg']      = params['mue'] / (params['na'] * params['nd']**2)
params['omega_s'] = params['omega'] * params['nt'] 

# initial conditions
params['h0']     = 126000
params['theta0'] = jnp.deg2rad(0)
params['phi0']   = jnp.deg2rad(0)
params['v0']     = 5845
params['gamma0'] = jnp.deg2rad(-15.47)
params['psi0']   = jnp.deg2rad(0)

# #aero look-up table
# lut = luts.create_interp_struct()

#discrete time grid
params['nk'] = 10
params['K'] = jnp.arange(0, params['nk'])
params['dt'] = 1 / (params['nk'] - 1)
params['tau'] = jnp.linspace(0, 1, params['nk'])

# optimization problem parameters
# indeces for state (including augmented ctcs states)
params['n_aug']     = 2
params['n']         = 6 + params['n_aug']
params['m']         = 1
params['nsq']       = params['n'] * params['n']
params['nxm']       = params['n'] * params['m']

params['r_idx']     = 0
params['theta_idx'] = 1
params['phi_idx']   = 2
params['v_idx']     = 3
params['gamma_idx'] = 4
params['psi_idx']   = 5
params['heat_idx']  = 6
params['qdyn_idx']  = 7

# indeces for flattened state used during discretization
params['phi_a_idx'] = params['n']
params['phi_b_m_idx'] = params['phi_a_idx']   + params['n'] * params['n']
params['phi_b_p_idx'] = params['phi_b_m_idx'] + params['n'] * params['m']
params['phi_s_idx'] = params['phi_b_p_idx'] + params['n'] * params['m']
params['end_idx'] = params['phi_s_idx']   + params['n']

params['nsub'] = 50
params['beta_ctcs'] = 1e-16