import numpy as np
from scipy.stats import qmc

def gen_normal_rn(shape, dt=1/252):
    return np.random.normal(0, np.sqrt(dt), size=shape)

def gen_uniform_rn(shape, dt=1 / 252):
    return np.random.uniform(0, 1, size=shape) * np.sqrt(dt)

def gen_rand_rn(shape, dt=1 / 252):
    return np.random.rand(*shape) * np.sqrt(dt)

def gen_sobol_rn(shape, dt=1 / 252):
    sobol = qmc.Sobol(d=shape[0], scramble=True)
    sobol_samples = sobol.random(n=shape[1]).T
    return sobol_samples * np.sqrt(dt)

def gen_halton_rn(shape, dt=1 / 252):
    halton = qmc.Halton(d=shape[0], scramble=True)
    halton_samples = halton.random(n=shape[1]).T
    return halton_samples * np.sqrt(dt)

def gen_lhs_rn(shape, dt=1 / 252):
    lhs = qmc.LatinHypercube(d=shape[0])
    lhs_samples = lhs.random(n=shape[1]).T
    return lhs_samples * np.sqrt(dt)

def generate_random_numbers(method, shape, dt=1 / 252):
    """
    Generate random numbers based on the specified method.
    """
    if method == "normal":
        return gen_normal_rn(shape, dt)
    elif method == "uniform":
        return gen_uniform_rn(shape, dt)
    elif method == "rand":
        return gen_rand_rn(shape, dt)
    elif method == "sobol":
        return gen_sobol_rn(shape, dt)
    elif method == "halton":
        return gen_halton_rn(shape, dt)
    elif method == "lhs":
        return gen_lhs_rn(shape, dt)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'normal', 'uniform', 'rand', 'sobol', 'halton', 'lhs'.")