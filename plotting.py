import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt
import time

# import generate_luts_jax as luts
import marsgram_dens_lut_jax as dens

from params import params
import discretization as disc
import nonlinear_prop as prop

from subproblem import optProblem

def plot(params, x, u, sigma):

    x_nl_prop, t_nl = prop.propagate_nonlin(params, x, u, sigma)

    t_nl = t_nl * sigma * params['nt']

    t = params['tau']*sigma*params['nt']

    r           = params['nd'] * x[0, :].T
    v           = params['nv'] * x[3, :].T
    alt         = r - params['re']

    r_nl_prop           = params['nd'] * x_nl_prop[0, :].T
    v_nl_prop           = params['nv'] * x_nl_prop[3, :].T
    alt_nl_prop         = r_nl_prop - params['re']

    plt.figure(figsize=(5, 2.5))
    # plt.title('h vs time')
    plt.ylabel("Altitude [km]")
    plt.xlabel("Time [s]")
    plt.plot(t_nl, (params['nd'] * x_nl_prop[0, :] - params['re'])/1e3, color='mediumblue', markersize=3, label='nl propagation')
    plt.plot(t, (params['nd'] * x[0, :] - params['re'])/1e3, 'o', color='lime', markersize=4, label='optimal knot points')
    plt.grid(True)

    plt.figure(figsize=(5, 2.5))
    # plt.title('h vs time')
    plt.ylabel("Velocity [km/s]")
    plt.xlabel("Time [s]")
    plt.plot(t_nl, params['nv'] * x_nl_prop[3, :]/1e3, color='mediumblue', markersize=3, label='nl propagation')
    plt.plot(t, params['nv'] * x[3, :]/1e3, 'o', color='lime', markersize=4, label='optimal knot points')
    plt.grid(True)

    plt.figure(figsize=(5*6/8, 5))
    plt.plot(np.rad2deg(x[1, :]), np.rad2deg(x[2, :]), color='mediumblue', marker='o', markersize=3)
    plt.xlabel("Longitude θ [deg]")
    plt.ylabel("Latitude φ [deg]")
    plt.xlim(-2, 2)
    plt.ylim(-2, 12)
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(5, 2.5))
    # plt.title('bank angle vs time')
    plt.ylabel("$\sigma$ [deg]")
    plt.xlabel("Time [s]")
    plt.plot(params['tau']*sigma*params['nt'], np.rad2deg(u[0,:]), color='mediumblue', marker='o', markersize=3)
    plt.grid(True)

    # plt.figure(figsize=(5, 2.5))
    # # plt.title('alpha vs time')
    # plt.ylabel(r"$\alpha$ [deg]")
    # plt.xlabel("Time [s]")
    # plt.plot(params['tau']*sigma*params['nt'], np.rad2deg(u[1,:]), color='mediumblue', marker='o', markersize=3)
    # plt.grid(True)

    # import generate_luts_jax as luts
    import marsgram_dens_lut_np as dens_np

    heat = np.zeros_like(t)
    q = np.zeros_like(t)
    load = np.zeros_like(t)

    heat_nl_prop = np.zeros_like(t_nl)
    q_nl_prop = np.zeros_like(t_nl)
    load_nl_prop = np.zeros_like(t_nl)

    kQ = 9.4369*10**(-5)
    kq = params['ge'] * params['re'] / 2

    for i in range(t.shape[0]):
        # Compute altitude

        # rho = mars_atmospheric_density(hdim)
        rho = np.interp(alt[i]/1e3, dens_np.h_grid, dens_np.rho_vals)

        heat[i] = kQ * np.sqrt(rho) * (v[i])**3 / 1e3
        q[i] = 0.5* rho * v[i]**2 / 1e3
        # load[i] =

    for i in range(t_nl.shape[0]):
        # Compute altitude

        # rho = mars_atmospheric_density(hdim)
        rho_nl_prop = np.interp(alt_nl_prop[i]/1e3, dens_np.h_grid, dens_np.rho_vals)

        heat_nl_prop[i] = kQ * np.sqrt(rho_nl_prop) * (v_nl_prop[i])**3 / 1e3
        q_nl_prop[i] = 0.5* rho_nl_prop * v_nl_prop[i]**2 / 1e3

    plt.figure(figsize=(5, 2.5))
    # plt.title('alpha vs time')
    plt.xlabel("Time [s]")
    plt.ylabel("Heat Rate [kW / $m^{2}$]")
    plt.plot(t_nl, heat_nl_prop, color='mediumblue', markersize=3, label='nl propagation')
    plt.plot(t, heat, 'o', color='lime', markersize=4, label='optimal nodes')
    plt.axhline(y=params['Qmax']/1e3, color='k', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.tight_layout()

    # plt.savefig("heat.pdf", format="pdf", bbox_inches="tight")

    plt.figure(figsize=(5, 2.5))
    # plt.title('alpha vs time')
    plt.xlabel("Time [s]")
    plt.ylabel("Dynamic Pressure [kPa]")
    plt.plot(t_nl, q_nl_prop, color='mediumblue', markersize=3, label='nl propagation')
    plt.plot(t, q, 'o', color='lime', markersize=4, label='optimal nodes')
    plt.axhline(y=params['qmax']/1e3, color='k', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.tight_layout()

    # plt.savefig("q.pdf", format="pdf", bbox_inches="tight")

    # plt.figure()
    # plt.figure(figsize=(5, 2.5))
    # # plt.title('alpha vs time')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Normal Load [kPa]")
    # plt.plot(t, load, color='mediumblue', markersize=3)
    # plt.grid(True)
    # plt.tight_layout()

    # save figures to desktop
    import os
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")

    names= ['alt', 'vel', 'long_lat', 'bank', 'aoa']
    for i in range(5):  # suppose you have 3 figures
        plt.figure(i+1)
        filename = os.path.join(desktop, f"prelim_scp_" + names[i]+ ".png")
        plt.savefig(filename, format="pdf", bbox_inches="tight")

    plt.show(block=False)
    input()
    plt.close('all')