import numpy as np

def mu_law(x, mu):
    ml = np.sign(x) * np.log(mu * np.abs(x) + 1.0) / np.log(mu + 1.0)
    return ((ml + 1.0) / 2.0 * mu + 0.5).astype(int)

def de_mu_law(y, mu):
    scaled = 2 * (y / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** abs(scaled) - 1)
    return np.sign(scaled) * magnitude

def one_hot(y, values):
    array = np.zeros((y.size, values))
    array[np.arange(y.size), y] = 1
    return array
