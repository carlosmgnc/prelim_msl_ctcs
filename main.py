import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt
import time

from params import params
import discretization as disc
import plotting

from subproblem import optProblem

opt = optProblem()
opt.def_cvx_problem(ctcs=True)

max_iter = 20
converged_iter = max_iter - 1
sigma_list = np.empty((max_iter, 1))
trajectory_list = np.zeros((max_iter, params['n'], params['nk']))

x = np.zeros((params['n'], params['nk']))
u = np.zeros((params['m'], params['nk']))
sigma = 0
cost = 0
nu = np.zeros((params['n'], params['nk']-1))

for i in range(0, max_iter):
    print(f"\niteration: {i+1}")

    t0 = time.perf_counter()
    x_prop_jax, A_jax, B_jax, C_jax, S_jax = disc.discretize(opt.xk_jax, opt.uk_jax, opt.sigmak_jax, params=params)

    opt.x_prop_jax = x_prop_jax
    opt.A_jax      = A_jax
    opt.B_jax      = B_jax
    opt.C_jax      = C_jax
    opt.S_jax      = S_jax

    opt.x_prop_cp.value = np.asarray(x_prop_jax)
    opt.A_cp.value      = np.asarray(A_jax)
    opt.B_cp.value      = np.asarray(B_jax)
    opt.C_cp.value      = np.asarray(C_jax)
    opt.S_cp.value      = np.asarray(S_jax)

    t1 = time.perf_counter()
    ms = (t1-t0)*1000
    print(f"discretization time: {ms:.2f} ms")

    x, u, sigma, cost, nu = opt.solve_cvx_problem()

    sigma_list[i] = sigma
    trajectory_list[i, :, :] = x

    # TODO: update convergence criteria
    if np.abs(x[params['v_idx'], -1] - cost) <= 0.000025:
        converged_iter = i
        break

plotting.plot(params, x, u, sigma)