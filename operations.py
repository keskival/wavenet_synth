import numpy as np

def de_mu_law(y, mu):
    scaled = 2 * (y / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** abs(scaled) - 1)
    return np.sign(scaled) * magnitude

