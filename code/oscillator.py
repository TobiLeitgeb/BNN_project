#############################################
# Author: Tobias Leitgeb
# date: 09.2024
# Description: Bayesian Neural Network with numpyro
#############################################
import numpyro
#numpyro.set_platform("cpu")
import jax
import jax.numpy as jnp
import jax.random as jr 
import numpyro.distributions as dist

from numpyro.infer import Predictive, NUTS, MCMC
from numpyro.optim import Adam

import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

from utils import sample_training_points_space_filling
from network_functions import dense_layer, forward

from typing import Tuple, List, Callable
import sys
sys.path.append("../")

sns.set_theme("paper", font_scale=1.5)

NOISE_LEVELS = (0.1, 0.6)

def bnn_model(
        X: jax.Array,
        Y: jax.Array,
        layers: List[int],
):
    N, input_dim = X.shape
    activation = jnp.tanh
    W = []
    b = []
    #build the layers with the given list
    for i, layer in enumerate(layers):
        W_, b_ = dense_layer(i, [input_dim, layer])
        W.append(W_)
        b.append(b_)
        input_dim = layer
    #forward pass through the network
    z = forward(W, b, X, activation)[:, None]

    if Y is not None:
        assert Y.shape == z.shape , f"Y shape {Y.shape} does not match z shape {z.shape}"

    precision_obs = numpyro.sample(r"observation precision", dist.Gamma(2., 1.))
    sigma_obs = 1.0 / jnp.sqrt(precision_obs)   
    sigma_obs = 0.1
    with numpyro.plate("data", N):
        numpyro.sample(
            "Y", 
            dist.Normal(z, sigma_obs).to_event(1), 
            obs=Y
        )


def main(
        layers, train, data_size, num_warmup, num_samples
):

    data = jnp.load('data/oscilator1_data.npy', allow_pickle=True).item()
    X, Y, Y_f = data['X'], data['Y'], data['Y_f']
   
    # Sample training points
    X_train, Y_train, _, _, X_test, y_test, _ = sample_training_points_space_filling(X, Y, Y_f, 50, seed=0)
    #normalize the training data
    mean_y = jnp.mean(Y_train)
    std_y = jnp.std(Y_train)
    Y_train = (Y_train - mean_y) / std_y

    try:
        render = numpyro.render_model(bnn_model, (X_train, Y_train, layers))
        render.render("plots/bnn_oscilator1_pinn")
    except:
        render = None
        print("Module not installed. (pip install graphviz), (sudo apt-get install graphviz)")



    key = jr.PRNGKey(0)
    key, inf_key, pred_key = jr.split(key, 3)

    name = f"BNN_{data_size}_{layers}"
    path = f"data/mcmc_samples/"+name+"_samples.npy"
    if train:
        mcmc = MCMC(NUTS(bnn_model), num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(inf_key, X_train, Y_train, layers)
        mcmc.print_summary()
        samples = mcmc.get_samples()
        jnp.save(path, samples, allow_pickle=True)

    else:
        try:
            samples = jnp.load(path, allow_pickle=True).item()
        except:
            print("No samples found. Train the model first.")
            return
        
    #predictive distribution
    predictive = Predictive(bnn_model, samples, return_sites=["Y"])
    predictions = predictive(pred_key, X_test, None, layers) 


    label = f"BNN/oscilator1_{data_size}_{layers}"

    plt.figure(figsize=(8, 6))
    plt.plot(X_test, y_test, 'b--', label='True function')
    plt.plot(X_train, Y_train*std_y + mean_y, 'ro', label=r' training samples')
    # Plot the predictions
    mean_prediction = jnp.mean(predictions["Y"], axis=0)
    stddev_prediction = jnp.std(predictions["Y"], axis=0)
    plt.plot(X_test, mean_prediction*std_y + mean_y, 'g', label='Predictive mean')
    plt.fill_between(
        X_test.flatten(),
        (mean_prediction - 2 * stddev_prediction).flatten()*std_y + mean_y,
        (mean_prediction + 2 * stddev_prediction).flatten()*std_y + mean_y,
        color='g',
        alpha=0.4,
        label=r"2\sigma uncertainty",
    )
    plt.legend()   
    plt.savefig("plots/"+label+".png", dpi=300, bbox_inches='tight')

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--layers", type=int, default=[40, 40, 1],nargs="+", help="Number of neurons in each layer")
    parser.add_argument("--data_size", type=int, default=50, help="Number of training points")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--num_warmup", type=int, default=1000, help="Number of warmup steps")
    args = parser.parse_args()
    main(**vars(args))
    





