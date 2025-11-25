import random
import numpy as np

def Bernoulli(p):
    """Return a function that generates Bernoulli rewards with probability p."""
    def reward():
        return 1 if random.random() < p else 0
    return reward

def Gaussian(mu, sigma):
    """Return a function that generates Gaussian rewards with mean mu and std sigma."""
    def reward():
        return np.random.normal(mu, sigma)
    return reward

def linTransformBernoulli(p, min, max):
    """Return a function that generates linearly transformed Bernoulli rewards."""
    def reward():
        base = 1 if random.random() < p else 0
        return min + (max - min) * base
    return reward

def uniform(a, b):
    """Return a function that generates Uniform rewards between a and b."""
    def reward():
        return np.random.uniform(a, b)
    return reward

def exponential(scale):
    """Return a function that generates Exponential rewards with given scale."""
    def reward():
        return np.random.exponential(scale)
    return reward

def constant(value):
    """Return a function that always returns a constant reward."""
    def reward():
        return value
    return reward

def discreteRV(values, probabilities):
    """Return a function that generates rewards from a discrete random variable."""
    def reward():
        return np.random.choice(values, p=probabilities)
    return reward
