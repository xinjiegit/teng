import pickle
from collections import namedtuple
from math import prod, sqrt
from typing import (
    Any,
    Tuple,
)

import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.numpy import pi

from src.sampler import AbstractSampler

NN = Any
Sampler = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PyTreeDef = Any


def gen_circle_sample_funcs(nb_samples, radius=1.0):
    """Generate functions to sample points exactly on the circumference of a circle with a given radius."""

    def gen_grid():
        # Set radius to the fixed value and define angular points for full circle
        theta = jnp.linspace(0, 2 * pi, nb_samples, endpoint=False)  # Angular points evenly spaced
        x_grid = radius * jnp.cos(theta)
        y_grid = radius * jnp.sin(theta)
        circle_points = jnp.stack((x_grid, y_grid), axis=-1)  # Stack x and y as Cartesian coordinates
        return circle_points.reshape((1, nb_samples, 2))

    grid_points = gen_grid()  # Generate the fixed grid once

    @jax.jit
    def sample(rand_key):
        """Randomly permute the points on the circle for stochastic sampling."""
        rand_key, sub_rand_key = jrnd.split(rand_key)
        perm = jrnd.permutation(sub_rand_key, nb_samples)
        samples = grid_points[perm]  # Shuffle points randomly for stochastic sampling
        return samples.reshape((1, nb_samples, 2)), rand_key  # Reshape to add the extra dimension

    @jax.jit
    def sample_deterministic(starts=None):
        """Deterministic sampling, returning the grid points directly."""
        return grid_points  # Return the ordered points without randomization

    @jax.jit
    def sample_grid():
        """Return the fixed grid of points on the circle."""
        return grid_points

    return sample, sample_deterministic, sample_grid


def gen_polar_disk_sample_funcs(nb_samples, radius=1.0):
    nb_points_each_site = (round(sqrt(nb_samples)),) * 2  # For (r, theta) sampling
    nb_points_total = nb_points_each_site[0] * nb_points_each_site[1]
    assert nb_points_total == nb_samples, "nb_samples must be a perfect square for grid sampling on disk"

    def gen_grid():
        r = jnp.linspace(1 / nb_points_each_site[0], radius, nb_points_each_site[0], endpoint=False)  # Radial points
        theta = jnp.linspace(0, 2 * pi, nb_points_each_site[1], endpoint=False)  # Angular points

        # Create polar grid
        r_grid, theta_grid = jnp.meshgrid(r, theta, indexing='ij')

        # Convert polar grid to Cartesian coordinates
        x_grid = r_grid * jnp.cos(theta_grid)
        y_grid = r_grid * jnp.sin(theta_grid)

        # Stack x and y to get Cartesian points ordered by (r, theta) logic
        cartesian_grid_points = jnp.stack((x_grid.flatten(), y_grid.flatten()), axis=-1)
        return cartesian_grid_points.reshape((1, nb_samples, 2))  # Add extra dimension

    grid_points = gen_grid()  # Generate fixed grid once

    @jax.jit
    def sample(rand_key):
        """Random sampling within the disk using polar coordinates."""
        rand_key, sub_rand_key_r, sub_rand_key_theta = jrnd.split(rand_key, 3)
        r = radius * jnp.sqrt(jrnd.uniform(sub_rand_key_r, (nb_samples,)))  # sqrt(U) for radial uniform distribution
        theta = jrnd.uniform(sub_rand_key_theta, (nb_samples,)) * 2 * pi  # Uniform angle
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        samples = jnp.stack((x, y), axis=-1).reshape((1, nb_samples, 2))  # Add extra dimension
        return samples, rand_key

    @jax.jit
    def sample_deterministic(starts=None):
        """Deterministic sampling, returning the grid points directly."""
        return grid_points  # Optionally shift based on starts if needed

    @jax.jit
    def sample_grid():
        """Return the grid of points on the disk."""
        return grid_points

    return sample, sample_deterministic, sample_grid

# there may be an object oriented way to do this, but we will use a functional approach for now
def gen_periodic_quadrature_sample_funcs(nb_sites, nb_samples, minvals, maxvals):
    nb_points_each_site = (round(nb_samples ** (1 / nb_sites)),) * nb_sites
    nb_points_total = prod(nb_points_each_site)
    assert nb_points_total == nb_samples, "nb_samples must be a perfect power of nb_sites"
    ranges = maxvals - minvals

    def gen_grid():
        grid_each_site = [jnp.linspace(minvals[i], maxvals[i], nb_points_each_site[i], endpoint=False) for i in
                          range(nb_sites)]
        grid_points = jnp.stack(jnp.meshgrid(*grid_each_site, indexing='ij'), axis=-1)
        return grid_points.reshape((1, nb_points_total, nb_sites))

    grid_points = gen_grid()

    @jax.jit
    def sample(rand_key):
        rand_key, sub_rand_key = jrnd.split(rand_key)
        starts = jrnd.uniform(sub_rand_key, (nb_sites,), minval=0, maxval=ranges)
        samples = ((starts + grid_points) % ranges + minvals)
        return samples, rand_key

    @jax.jit
    def sample_deterministic(starts):
        samples = ((starts + grid_points) % ranges + minvals)
        return samples

    @jax.jit
    def sample_grid():
        return grid_points

    return sample, sample_deterministic, sample_grid


class CircleSampler(AbstractSampler):
    def __init__(self, nb_samples, radius=1.0, quad_rule=None, rand_seed=1234):
        super().__init__()
        self.radius = radius
        self.circumference = 2 * pi * radius
        self.nb_samples = nb_samples

        # Generate sampling functions specifically for the circle
        self.pure_funcs = namedtuple('CircleSamplerPure', ['sample', 'sample_deterministic', 'sample_grid']) \
            (*gen_circle_sample_funcs(nb_samples, radius))

        # Initialize quadrature weights
        if quad_rule is None:
            quad_rule = 'midpoint'
        quad_rule = quad_rule.lower()
        if quad_rule in ['midpoint', 'trapezoidal', 'mid', 'trap', 'trapezoid']:
            sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.circumference / nb_samples)
        elif quad_rule in ['simpson', 'simpsons']:
            assert nb_samples % 2 == 0, f"{nb_samples=} must be even for Simpson's rule"
            sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.circumference / (1.5 * nb_samples))
            sqrt_weights = sqrt_weights.at[::2].multiply(jnp.sqrt(2))
        elif isinstance(quad_rule, jnp.ndarray):
            sqrt_weights = quad_rule
        else:
            raise ValueError(f"{quad_rule=} is not a valid value; must be a string or a jnp.ndarray, or None")
        self.sqrt_weights = sqrt_weights.reshape(1, nb_samples)

        # Random key initialization for reproducibility
        self.rand_key = jrnd.PRNGKey(rand_seed)

    def set_var_state(self, var_state):
        # Not needed for this sampler
        pass

    def sample(self, start=None):
        if start is None:
            samples, rand_key = self.pure_funcs.sample(self.rand_key)
            self.rand_key = rand_key
        elif start == 0:
            samples = self.pure_funcs.sample_grid()
        else:
            samples = self.pure_funcs.sample_deterministic()
        return samples, None, self.sqrt_weights

    def get_state(self):
        return {'rand_key': self.rand_key}

    def set_state(self, state):
        self.rand_key = state['rand_key']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))


class DiskSampler(AbstractSampler):
    def __init__(self, nb_samples, radius=1.0, quad_rule=None, rand_seed=1234):
        super().__init__()
        self.radius = radius
        self.area = jnp.pi * radius ** 2
        self.nb_samples = nb_samples

        # Generate sampling functions
        self.pure_funcs = namedtuple('DiskSamplerPure', ['sample', 'sample_deterministic', 'sample_grid']) \
            (*gen_polar_disk_sample_funcs(nb_samples, radius))

        # Initialize quadrature weights
        if quad_rule is None:
            quad_rule = 'midpoint'
        quad_rule = quad_rule.lower()
        if quad_rule in ['midpoint', 'trapezoidal', 'mid', 'trap', 'trapezoid']:
            sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.area / nb_samples)
        elif quad_rule in ['simpson', 'simpsons']:
            assert nb_samples % 2 == 0, f"{nb_samples=} must be even for Simpson's rule"
            sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.area / (1.5 * nb_samples))
            sqrt_weights = sqrt_weights.at[::2].multiply(jnp.sqrt(2))
        elif isinstance(quad_rule, jnp.ndarray):
            sqrt_weights = quad_rule
        else:
            raise ValueError(f"{quad_rule=} is not a valid value; must be a string or a jnp.ndarray, or None")
        self.sqrt_weights = sqrt_weights.reshape(1, nb_samples)

        # Random key initialization for reproducibility
        self.rand_key = jax.random.PRNGKey(rand_seed)

    def set_var_state(self, var_state):
        # Not needed for this sampler
        pass

    def sample(self, start=None):
        if start is None:
            samples, rand_key = self.pure_funcs.sample(self.rand_key)
            self.rand_key = rand_key
        elif start == 0:
            samples = self.pure_funcs.sample_grid()
        else:
            samples = self.pure_funcs.sample_deterministic()
        return samples, None, self.sqrt_weights

    def get_state(self):
        return {'rand_key': self.rand_key}

    def set_state(self, state):
        self.rand_key = state['rand_key']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))

class PeriodicQuadratureSampler(AbstractSampler):
    """
    samples at fixed intervals with appropriate quadrature weights for periodic boundary conditions
    the starting point is uniformly sampled
    does not support multiple devices now
    """

    def __init__(self, nb_sites, nb_samples, minvals, maxvals, quad_rule=None, rand_seed=1234):
        super().__init__()
        self.nb_sites = nb_sites
        if isinstance(minvals, (int, float)):
            minvals = [minvals] * nb_sites
        else:
            assert len(minvals) == nb_sites, f"{len(minvals)=} must be equal to {nb_sites=} or a single value"
        if isinstance(maxvals, (int, float)):
            maxvals = [maxvals] * nb_sites
        else:
            assert len(maxvals) == nb_sites, f"{len(maxvals)=} must be equal to {nb_sites=} or a single value"
        self.minvals = jnp.array(minvals)
        self.maxvals = jnp.array(maxvals)
        self.area = jnp.prod(self.maxvals - self.minvals)
        # self.pure_funcs = PeriodicQuadratureSamplerPure(self.nb_sites)
        self.pure_funcs = namedtuple('PeriodicQuadratureSamplerPure', ['sample', 'sample_deterministic', 'sample_grid']) \
            (*gen_periodic_quadrature_sample_funcs(self.nb_sites, nb_samples, self.minvals, self.maxvals))
        self.devices = jax.local_devices()
        self.nb_devices = jax.local_device_count()
        assert self.nb_devices == 1, f"{self.nb_devices=} must be 1"
        self.nb_samples = nb_samples
        if quad_rule is None:
            quad_rule = 'midpoint'
        quad_rule = quad_rule.lower()
        if quad_rule == 'midpoint' or quad_rule == 'trapezoidal' or quad_rule == 'mid' or quad_rule == 'trap' or quad_rule == 'trapezoid' or quad_rule == 'trapozoidal' or quad_rule == 'trapozoid':
            sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.area / nb_samples)
        elif quad_rule == 'simpson' or quad_rule == 'simpsons':
            assert nb_samples % 2 == 0, f"{nb_samples=} must be even for (periodic) simpson's rule"
            sqrt_weights = jnp.ones(nb_samples) * jnp.sqrt(self.area / (1.5 * nb_samples))
            sqrt_weights = sqrt_weights.at[::2].multiply(jnp.sqrt(2))
        elif type(quad_rule) == str:
            raise NotImplementedError(f"{quad_rule=} is not implemented")
        elif type(quad_rule) == jnp.ndarray:
            sqrt_weights = quad_rule
        else:
            raise ValueError(f"{quad_rule=} is not a valid value, must be a string or a jnp.ndarray, or None")
        self.sqrt_weights = sqrt_weights.reshape(1, nb_samples)

        self.rand_key = jax.random.PRNGKey(rand_seed)

    def set_var_state(self, var_state):
        # we don't need this function in this case
        pass

    def sample(self, start=None):
        if start is None:
            samples, rand_key = self.pure_funcs.sample(self.rand_key)
            self.rand_key = rand_key
        elif start == 0:
            samples = self.pure_funcs.sample_grid()
        else:
            samples = self.pure_funcs.sample_deterministic(start)
        return samples, None, self.sqrt_weights

    def get_state(self):
        return {'rand_key': self.rand_key}

    def set_state(self, state):
        self.rand_key = state['rand_key']

    def save_state(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path):
        with open(path, "rb") as f:
            self.set_state(pickle.load(f))
