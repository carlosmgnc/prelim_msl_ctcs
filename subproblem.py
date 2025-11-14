import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import matplotlib.pyplot as plt

# import generate_luts_jax as luts
from params import params

# from cvxpygen import cpg
# from scp_socp_solver.cpg_solver import cpg_solve 

class optProblem:
    def __init__(self):

        alpha1_list = (1 - params['tau']) / (1)
        alpha2_list = 1 - alpha1_list

        # initial trajectory guess
        ri = (params['re'] + params['h0']) / params['nd']
        rf = (params['re'] + 10e3) / params['nd']

        thetai = params['theta0']
        thetaf = jnp.deg2rad(0)

        phii = params['phi0']
        phif = jnp.deg2rad(10.5768)

        vi = params['v0'] / params['nv']
        vf = 406 / params['nv']

        gammai = params['gamma0']
        gammaf = jnp.deg2rad(-10)

        psii = params['psi0']
        psif = jnp.deg2rad(0)

        self.tfguess = 240 / params['nt']

        params['zi'] = jnp.vstack([ri, thetai, phii, vi, gammai, psii, 0, 0])
        params['zf'] = jnp.vstack([rf, thetaf, phif, vf, gammaf, psif, 0, 0])

        # straight line initial guess
        zk_jax = alpha1_list * params['zi'] + alpha2_list * params['zf']
        uk_jax = jnp.deg2rad(-5)*jnp.ones((1, params['nk']))

        zk_np = np.asarray(zk_jax)
        uk_np = np.asarray(uk_jax)

        # optimization problem variables and parameters
        self.x = cp.Variable((params['n'], params['nk']), name='x')
        self.u = cp.Variable((params['m'], params['nk']), name='u')
        self.sigma = cp.Variable(nonneg=True, name='sigma')
        self.nu = cp.Variable((params['n'], params['nk'] - 1), name="nu")
        self.delta = cp.Variable((params['nk'], 1), nonneg=True, name="delta")
        self.delta_sigma = cp.Variable(nonneg=True, name="delta_sigma")

        self.xk_cp = cp.Parameter((params['n'], params['nk']),name='xk')
        self.uk_cp = cp.Parameter((params['m'], params['nk']),name='uk')
        self.sigmak_cp = cp.Parameter(nonneg=True, name='sigmak')
        
        self.A_cp = cp.Parameter((params['n'], params['n'] * (params['nk'] - 1)), name='Ak')
        self.B_cp = cp.Parameter((params['n'], params['m'] * (params['nk'] - 1)), name='Bk')
        self.C_cp = cp.Parameter((params['n'], params['m'] * (params['nk'] - 1)), name='Ck')
        self.S_cp = cp.Parameter((params['n'], params['nk'] - 1), name='Sk')
        self.x_prop_cp = cp.Parameter((params['n'], params['nk'] - 1), name='x_prop')

        self.w_nu = cp.Parameter(nonneg=True, name='w_nu')
        self.w_delta = cp.Parameter(nonneg=True, name='w_delta')
        self.w_sigma = cp.Parameter(nonneg=True, name='w_sigma')
        
        # matching weights
        self.w_nu.value = 1e5
        self.w_delta.value = 10
        self.w_sigma.value = 1

        # initialize problem parameters
        self.xk_cp.value = zk_np
        self.uk_cp.value = uk_np
        self.sigmak_cp.value = self.tfguess

        self.xk_jax = zk_jax
        self.uk_jax = uk_jax
        self.sigmak_jax = jnp.asarray(self.tfguess)

        self.A_jax      = jnp.zeros((params['n'], params['n'] * (params['nk'] - 1)))
        self.B_jax      = jnp.zeros((params['n'], params['m'] * (params['nk'] - 1)))
        self.C_jax      = jnp.zeros((params['n'], params['m'] * (params['nk'] - 1)))
        self.S_jax      = jnp.zeros((params['n'], params['nk'] - 1))
        self.x_prop_jax = jnp.zeros((params['n'], params['nk'] - 1))

        self.A_cp.value      = np.zeros((params['n'], params['n'] * (params['nk'] - 1)))
        self.B_cp.value      = np.zeros((params['n'], params['m'] * (params['nk'] - 1)))
        self.C_cp.value      = np.zeros((params['n'], params['m'] * (params['nk'] - 1)))
        self.S_cp.value      = np.zeros((params['n'], params['nk'] - 1))
        self.x_prop_cp.value = np.zeros((params['n'], params['nk'] - 1))

    def def_cvx_problem(self, ctcs=False):

        self.constraints = []
        self.cost = 0

        # initial conditions
        self.constraints += [self.x[:, 0] == self.xk_cp[:, 0]]

        # final conditionas
        self.constraints += [self.x[params['r_idx'], -1] == self.xk_cp[params['r_idx'], -1]]
        self.constraints += [self.x[params['theta_idx'], -1] == self.xk_cp[params['theta_idx'], -1]]
        self.constraints += [self.x[params['phi_idx'], -1] == self.xk_cp[params['phi_idx'], -1]]
        
        # ctcs constraints
        if ctcs == True:
            # beta_phys = (self.sigma * params['nt'] / (params['nk'] - 1)) * 1e-12
            self.constraints += [self.x[params['heat_idx'], 1:] - self.x[params['heat_idx'], :-1] <= params['beta_ctcs']]
            self.constraints += [self.x[params['qdyn_idx'], 1:] - self.x[params['qdyn_idx'], :-1] <= params['beta_ctcs']]

        # constraints that need to be satisfied throughout
        self.constraints += [self.x[0, :] >= (params['re'] + 10) / params['nd']]
        self.constraints += [self.x[params['v_idx'], :] >=  10 / params['nv']]

        self.constraints += [self.u[0, :] >= np.deg2rad(-170)]
        self.constraints += [self.u[0, :] <= np.deg2rad(170)]

        self.constraints += [self.sigma >= 50 / params['nt']]

        # compute deltas
        dsigma = self.sigma - self.sigmak_cp
        self.constraints += [dsigma <= self.delta_sigma]
        self.constraints += [dsigma >= -self.delta_sigma]

        for k in range(params['nk']):
            # difference in state, control and time from last iterate

            dx = self.x[:,[k]] - self.xk_cp[:, [k]]
            du = self.u[:, [k]] - self.uk_cp[:, [k]]

            self.constraints += [cp.sum_squares(dx) + cp.sum_squares(du) <= self.delta[k,0]]

        for k in range(0, params['nk'] - 1):
            indx1 = k*params['n']
            indx2 = k*params['m']

            self.constraints += [
                self.x[:, k + 1] - self.x_prop_cp[:, k]
                == self.A_cp[:, indx1:indx1 + params['n']] @ (self.x[:, k] - self.xk_cp[:, k])
                + self.B_cp[:, indx2:indx2 + params['m']] @ (self.u[:, k] - self.uk_cp[:, k])
                + self.C_cp[:, indx2:indx2 + params['m']] @ (self.u[:, k + 1] - self.uk_cp[:, k + 1])
                + self.S_cp[:, k] * (self.sigma - self.sigmak_cp)
                + self.nu[:, k]
            ]

            # control rate constraints
            self.constraints += [cp.abs(self.u[0, k+1] - self.u[0, k]) <= np.deg2rad(5.0) * (self.sigma * params['nt'] / (params['nk']-1))]
            #control_dev_cost += cp.sum_squares(self.u[:, k+1] - self.u[:, k])
        
        # self.nu_cost = cp.sum_squares(self.nu)
        self.nu_cost = cp.norm(cp.vec(self.nu[:, :], order='C'),1)

        self.delta_cost = cp.sum_squares(self.delta)
        self.sigma_cost = cp.sum_squares(self.delta_sigma)

        self.cost = (
              self.x[params['v_idx'], -1]
            #+ control_dev_cost
            + self.w_nu * self.nu_cost
            + self.w_delta * self.delta_cost
            + self.w_sigma * self.sigma_cost
        )

        objective = cp.Minimize(self.cost)
        self.prob = cp.Problem(objective, self.constraints)
        total_param_entries = sum(np.prod(param.shape) for param in self.prob.parameters())
        print("Total number of scalar parameters:", total_param_entries)

        # cpg.generate_code(self.prob, code_dir='scp_socp_solver', solver='QOCO', prefix="scp_socp")
        # self.prob.register_solve("scp_socp", cpg_solve) 
    
    def solve_cvx_problem(self):
        
        print("----------------------------------")
        self.prob.solve(solver="CLARABEL", ignore_dpp=True, warm_start=False, verbose=False)
        # self.prob.solve(method="scp_socp")
        print("solver status : " + self.prob.status)
        print("solve time    : " + f"{self.prob.solver_stats.solve_time * 1000:.2f}" + " ms")
        print("cost          : " + f"{self.prob.objective.value:.3f}")
        print("final time    : " + f"{self.sigma.value*params['nt']:.3f} s")
        print("norm(nu)      : " + f"{np.linalg.norm(self.nu.value.flatten())}")
        print("----------------------------------")

        self.xk_cp.value = np.array(self.x.value, copy=True)
        self.uk_cp.value = np.array(self.u.value, copy=True)
        self.sigmak_cp.value = np.array(self.sigma.value, copy=True)

        self.xk_jax = jnp.asarray(self.x.value)
        self.uk_jax = jnp.asarray(self.u.value)
        self.sigmak_jax = jnp.asarray(self.sigma.value)

        return self.x.value, self.u.value, self.sigma.value, self.cost.value, self.nu.value 

if __name__ == "__main__":
    pass