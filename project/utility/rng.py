import numpy as np
from scipy.stats import qmc

def gen_normal_rn(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0, 1, size=shape)

def gen_uniform_rn(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(0, 1, size=shape)

def gen_sobol_rn(shape, seed=None):
    sobol = qmc.Sobol(d = np.prpd(shape[:-1]), scarmble=True)
    sobol_samples = sobol.random(shape[-1]).reshape(shape)
    return sobol_samples * np.sqrt(1/252)

def gen_halton_rn(shape, seed=None):
    halton = qmc.Halton(d = np.prod(shape[:-1]), scramble=True)
    halton_samples = halton.random(shape[-1]).reshape(shape)
    return halton_samples * np.sqrt(1/252)

def gen_hammersley_rn(shape, seed=None):
    hammersley = qmc.Hammersley(d = np.prod(shape[:-1]), scramble=True)
    hammersley_samples = hammersley.random(shape[-1]).reshape(shape)
    return hammersley_samples * np.sqrt(1/252)

def generate_random_numbers(method, shape, seed=None):
    if method == "normal":
        return gen_normal_rn(shape, seed)
    elif method == "uniform":
        return gen_uniform_rn(shape, seed)
    elif method == "sobol":
        return gen_sobol_rn(shape, seed)
    elif method == "halton":
        return gen_halton_rn(shape, seed)
    elif method == "hammersley":
        return gen_hammersley_rn(shape, seed)
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'normal', 'uniform', 'sobol', 'halton', 'hammersley'.")
    