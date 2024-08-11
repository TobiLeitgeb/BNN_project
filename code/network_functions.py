#############################################
# Author: Tobias Leitgeb
# date: 09.2024
# Description: Functions for feedforward neural networks
#############################################

import jax
import jax.numpy as jnp
from typing import List, Callable
import numpyro
import numpyro.distributions as dist

def dense_layer(
        i: int,
        size: List[int],
):  
    #Xavier initialization
    alpha_sq = 2/(size[0] + size[1])
    #alpha_sq = 1.0
    W = numpyro.sample(f"W{i}", dist.Normal(0, alpha_sq**0.5).expand(size))
    b = numpyro.sample(f"b{i}", dist.Normal(0, alpha_sq**0.5).expand((size[-1],)))
    return W, b

def forward(
        W: List[jax.Array],
        b: List[jax.Array],
        X: jax.Array,
        activation: Callable,
):
    #input layer
    z = activation(jnp.dot(X, W[0]) + b[0])

    #hidden layers
    for i in range(1, len(W) - 1):
        z = activation(jnp.dot(z, W[i]) + b[i])

    #output layer with no activation
    z = jnp.dot(z, W[-1]) + b[-1]
    return z.squeeze()
