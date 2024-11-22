import numpy as np
from scipy.stats import qmc

def gen_normal_rn(shape, seed=None, dt=1/252):
    if seed is not None:
        np.random.seed(seed)
    #return np.random.normal(0, np.sqrt(dt), size=shape)  # Include scaling for dt
    return np.random.normal(0, 1, size=shape) * np.sqrt(dt)  # Apply scaling directly here

def gen_uniform_rn(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 1, size=shape)

def gen_sobol_rn(shape, seed=None, dt=1/252):
    sobol = qmc.Sobol(d=np.prod(shape[:-1]), scramble=True)
    sobol_samples = sobol.random(shape[-1]).reshape(shape)
    return sobol_samples * np.sqrt(dt)  # Scale with dt

def gen_halton_rn(shape, seed=None, dt=1/252):
    halton = qmc.Halton(d=np.prod(shape[:-1]), scramble=True)
    halton_samples = halton.random(shape[-1]).reshape(shape)
    return halton_samples * np.sqrt(dt)  # Scale with dt

def gen_hammersley_rn(shape, seed=None, dt=1/252):
    hammersley = qmc.Hammersley(d=np.prod(shape[:-1]), scramble=True)
    hammersley_samples = hammersley.random(shape[-1]).reshape(shape)
    return hammersley_samples * np.sqrt(dt)  # Scale with dt

def generate_random_numbers(method, shape, seed=None, dt=1/252):
    if method == "normal":
        return gen_normal_rn(shape, seed, dt)
    elif method == "uniform":
        return gen_uniform_rn(shape, seed)
    elif method == "sobol":
        return gen_sobol_rn(shape, seed, dt)
    elif method == "halton":
        return gen_halton_rn(shape, seed, dt)
    elif method == "hammersley":
        return gen_hammersley_rn(shape, seed, dt)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'normal', 'uniform', 'sobol', 'halton', 'hammersley'.")