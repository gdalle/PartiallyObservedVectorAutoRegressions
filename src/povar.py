import cvxpy as cp

# import gurobipy
from joblib import Parallel, delayed
import numba
import numpy as np
from tqdm.notebook import tqdm

# If you have Gurobi installed
# SOLVER = cp.GUROBI
# SOLVER_KWARGS = {"Method": 0}
# Otherwise
SOLVER = cp.ECOS
SOLVER_KWARGS = {}


class VAR:
    """Vector AutoRegressive process with known variance and transition matrix."""

    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma

    def simulate(self, N, T):
        return simulate_var(self.theta, self.sigma, N, T)


class IndependentSampler:
    """Independent sampling mechanism with probability p."""

    def __init__(self, p):
        self.p = p

    def sample(self, N, T, D):
        return sample_indep(self.p, N, T, D)

    def scaling(self, h, T, D, **kwargs):
        """Compute S(h)."""
        p = self.p
        S = np.full((D, D), p ** 2)
        if h == 0:
            S[np.diag_indices(D)] = p
        return S

    def noise_correction(self, D, h):
        """Compute C(h)."""
        return (h == 0) * np.eye(D)


class FixedSizeSampler:
    """Fixed-size sampling mechanism with probability p."""

    def __init__(self, p):
        self.p = p

    def sample(self, N, T, D):
        return sample_fixed_size(self.p, N, T, D)

    def scaling(self, h, T, D):
        """Compute S(h)."""
        p = self.p
        if h == 0:
            S = np.full((D, D), 1 - (1 - 2 / D) ** (p * D))
            S[np.diag_indices(D)] = 1 - (1 - 1 / D) ** (p * D)
        else:
            S = np.full((D, D), (1 - (1 - 1 / D) ** (p * D)) ** 2)
        return S

    def noise_correction(self, D, h):
        """Compute C(h)."""
        b = np.random.binomial(int(self.p * D), 1 / D, size=1000)
        num = np.sum(1 / b[b > 0]) / b.shape[0]
        den = self.p
        return (h == 0) * (num / den) * np.eye(D)


class MarkovSampler:
    """Markov sampling mechanism with transition probabilities a and b."""

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.p = a / (a + b)

    def sample(self, N, T, D):
        return sample_markov(self.a, self.b, N, T, D)

    def scaling(self, h, T, D, **kwargs):
        """Compute S(h)."""
        p, a, b = self.p, self.a, self.b
        S = np.full((D, D), p ** 2)
        if h == 0:
            S[np.diag_indices(D)] = p
        else:
            S[np.diag_indices(D)] = p ** 2 + p * (1 - p) * (1 - a - b) ** h
        return S

    def noise_correction(self, D, h):
        """Compute C(h)."""
        return (h == 0) * np.eye(D)


class Observer:
    """Observation noise."""

    def __init__(self, omega):
        self.omega = omega

    def observe(self, X, obs_ind):
        return observe_povar(X, obs_ind, self.omega)


class POVAR:
    """Complete Partially-Observed Vector AutoRegression."""

    def __init__(self, var, sampler, observer):
        self.var = var
        self.sampler = sampler
        self.observer = observer

    def simulate(self, N, T):
        X = self.var.simulate(N, T)
        D = X.shape[-1]
        obs_ind = self.sampler.sample(N, T, D)
        Y = self.observer.observe(X, obs_ind)
        Z = obs_ind, Y, D
        return Z

    def estimate_covariance(self, X, h):
        N, T, D = X.shape
        scale = self.sampler.scaling(h, T, D)
        G = np.matmul(
            np.transpose(X[:N, h:T, :D], (0, 2, 1)), X[:N, 0 : (T - h), :D]
        ).sum(axis=0) / (N * (T - h) * scale)
        return G - (self.observer.omega ** 2) * self.sampler.noise_correction(D, h)

    def estimate_theta(
        self,
        Z=None,
        X=None,
        h0=0,
        target_density=None,
        lambda0_range=None,
        n_jobs=5,
        show_progress=False,
    ):
        if X is None:
            X = approximate_state(*Z)
        G1 = self.estimate_covariance(X, h0)
        G2 = self.estimate_covariance(X, h0 + 1)
        if target_density == 1:
            return G2 @ np.linalg.pinv(G1)
        elif target_density is not None:
            return estimate_theta_target_density(
                G1,
                G2,
                target_density=target_density,
                n_jobs=n_jobs,
            )
        else:
            return estimate_theta(
                G1,
                G2,
                lambda0_range=lambda0_range,
                n_jobs=n_jobs,
                show_progress=show_progress,
            )


@numba.jit(nopython=True)
def simulate_var(theta, sigma, N, T):
    D = theta.shape[0]
    X = np.empty((N, T, D))
    Sigma0 = (sigma ** 2) * np.linalg.inv(np.eye(D) - theta @ theta.T)
    U, s, Vt = np.linalg.svd(Sigma0)
    Sigma0_sqrt = U @ np.diag(np.sqrt(s)) @ Vt
    for n in range(N):
        X[n, 0, :] = Sigma0_sqrt @ np.random.randn(D)
        for t in range(1, T):
            innov = sigma * np.random.randn(D)
            X[n, t, :] = theta @ X[n, t - 1, :] + innov
    return X


# Numba-ized functions must be outside user-defined classes for optimal performance


@numba.jit(nopython=True)
def sample_indep(p, N, T, D):
    """Simulate an independent sampling mask."""
    obs_ind = np.full((N, T, D), -1)
    for n in range(N):
        for t in range(T):
            pi = np.random.binomial(n=1, p=p, size=D)
            ind = np.where(pi == 1)[0]
            count = ind.shape[0]
            obs_ind[n, t, :count] = ind
    return obs_ind


@numba.jit(nopython=True)
def sample_fixed_size(p, N, T, D):
    """Simulate a fixed-size sampling mask."""
    obs_ind = np.full((N, T, int(p * D)), -1)
    for n in range(N):
        for t in range(T):
            ind = np.random.choice(a=D, size=int(p * D), replace=True)
            count = int(p * D)
            obs_ind[n, t, :count] = ind
    return obs_ind


@numba.jit(nopython=True)
def sample_markov(a, b, N, T, D):
    """Simulate a Markov sampling mask."""
    obs_ind = np.full((N, T, D), -1)
    p = a / (a + b)
    av = np.full(D, a)
    bv = np.full(D, b)
    for n in range(N):
        for t in range(T):
            if t == 0:
                pi = np.random.binomial(n=1, p=p, size=D)
            else:
                last_pi = pi
                pi = np.random.binomial(
                    n=1, p=np.where(last_pi == 0, av, 1 - bv)[0], size=D
                )
            ind = np.where(pi == 1)[0]
            count = ind.shape[0]
            obs_ind[n, t, :count] = ind
    return obs_ind


@numba.jit(nopython=True)
def observe_povar(X, obs_ind, omega):
    """Simulate Y from X and the sampling mask obs_ind."""
    N, T, O = obs_ind.shape
    Y = np.full((N, T, O), np.NaN)
    for n in range(N):
        for t in range(T):
            for o in range(O):
                d = obs_ind[n, t, o]
                if d != -1:
                    Y[n, t, o] = X[n, t, d] + omega * np.random.randn()
    return Y


@numba.jit(nopython=True)
def approximate_state(obs_ind, Y, D):
    """Compute \Pi^+ Y as a proxy for the unknown state X."""
    N, T, O = Y.shape
    X = np.zeros((N, T, D))
    state_count = np.zeros((N, T, D), dtype=np.int64)
    for n in range(N):
        for t in range(T):
            for o in range(O):
                d = obs_ind[n, t, o]
                if d != -1:
                    state_count[n, t, d] += 1
                    k = state_count[n, t, d]
                    X[n, t, d] = ((k - 1) * X[n, t, d] + Y[n, t, o]) / k
    return X


# Estimation routines


def estimate_theta_row(G1, G2, d, lambda0_range):
    """Estimate row d of theta given covariance matrices G1 and G2, for all penalization values of lambda0_range."""
    D = G1.shape[0]
    theta_row_values = np.full((len(lambda0_range), D), np.NaN)
    # Declare variable
    theta_row = cp.Variable(D)
    # Declare parameter
    max_norm = cp.Parameter()
    # Declare constraint
    constraints = [cp.max(cp.abs(G2[d, :] - theta_row @ G1)) <= max_norm]
    # Declare objective
    objective = cp.Minimize(cp.sum(cp.abs(theta_row)))
    # Declare problem
    problem = cp.Problem(objective, constraints)
    # Enumeration
    for (k, lambda0) in enumerate(lambda0_range):
        # Set parameter value
        max_norm.value = lambda0
        # Solve
        problem.solve(solver=SOLVER, verbose=False, warm_start=True, **SOLVER_KWARGS)
        if problem.status == cp.OPTIMAL:
            # Store solution
            theta_row_values[k, :] = theta_row.value.copy()
        else:
            raise ValueError("Optimization did not converge")
    if len(lambda0_range) == 1:
        return theta_row_values[0]
    else:
        return theta_row_values


def estimate_theta(G1, G2, lambda0_range, n_jobs, show_progress):
    """Estimate all rows of theta."""
    D = G1.shape[0]
    dims = tqdm(range(D), desc="Estimating theta") if show_progress else range(D)
    return np.stack(
        Parallel(n_jobs=n_jobs)(
            delayed(estimate_theta_row)(G1, G2, d, lambda0_range=lambda0_range)
            for d in dims
        ),
        axis=1,
    )


def density(x, weight_threshold=1e-5):
    """Compute the proportion of non-negligible coefficients in an array."""
    return np.sum(np.abs(x) > weight_threshold) / np.prod(x.shape)


def estimate_theta_target_density(G1, G2, target_density, n_jobs):
    """Use dichotomy to estimate theta with a given target density."""
    D = G1.shape[0]
    # Dichotomy
    lambda0_min, lambda0_max = 0, 1
    while (lambda0_max - lambda0_min) / lambda0_max > 0.2:
        lambda0 = (lambda0_max + lambda0_min) / 2
        # Estimate row by row
        theta = np.stack(
            Parallel(n_jobs=n_jobs)(
                delayed(estimate_theta_row)(G1, G2, d, lambda0_range=[lambda0])
                for d in range(D)
            ),
        )
        # Split on density
        if density(theta) > target_density:
            lambda0_min = (lambda0_max + lambda0_min) / 2
        else:
            lambda0_max = (lambda0_max + lambda0_min) / 2
    assert 0 < lambda0_min < lambda0_max < 1
    return theta


# Misc.


def random_theta(D, s=None, norm=0.5):
    """Generate a random transition matrix with independent Gaussian entries."""
    if s is None:
        s = D
    theta = np.random.normal(loc=0.0, scale=1.0, size=(D, D))
    for d in range(D):
        column_zeros = np.random.choice(a=D, size=D - s, replace=False)
        theta[column_zeros, d] = 0
    return norm * theta / np.linalg.norm(theta, 2)
