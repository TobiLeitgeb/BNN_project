import numpyro
#numpyro.set_platform("cpu")
import jax
import jax.numpy as jnp
import jax.random as jr 
import numpyro.distributions as dist

from numpyro.infer import Predictive, NUTS, MCMC

import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
sns.set_theme("paper", font_scale=1.5)

from typing import Tuple, List, Callable
import sys
sys.path.append("../")
from utils import sample_training_points_space_filling
from network_functions import dense_layer, forward, relative_l2_error


NOISE_LEVELS = (0.01, 0.3)
# forward pass through the physics network
def physics_forward(
        W: List[jax.Array],
        b: List[jax.Array],
        X: jax.Array,
        activation: Callable,
):
    #input layer
    u = partial(forward, W, b, activation=activation)
    u_prime = jax.grad(u)
    u_double_prime = jax.grad(u_prime)

    #vectorize the functions
    u_prime = jax.vmap(u_prime)(X)
    u_double_prime = jax.vmap(lambda x: u_double_prime(x.squeeze()))(X)
    return u_prime, u_double_prime[:, None]

def BPINN(
        X: jax.Array,
        Y: jax.Array,
        layers: List[int],
):
    X_u, X_f = X
    
    if Y is not None:
        Y_u, Y_f = Y
    else:
        Y_u, Y_f = None, None

    N, input_dim = X_u.shape
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
    z = forward(W, b, X_u, activation)[:, None]

    dz, ddz = physics_forward(W, b, X_f, activation)
    assert dz.shape == z.shape, f"dz shape {dz.shape} does not match z shape {z.shape}"
    assert ddz.shape == z.shape, f"ddz shape {ddz.shape} does not match z shape {z.shape}"

    m = numpyro.sample(r"m", dist.Uniform(0, 2.))
    gamma = numpyro.sample(r"gamma", dist.Uniform(0, .8))
    k = numpyro.sample(r"k", dist.Uniform(1, 3.))

    if Y is not None:
        assert Y_u.shape == z.shape , f"Y shape {Y_u.shape} does not match z shape {z.shape}"
  
    sigma_obs_u = NOISE_LEVELS[0]
    sigma_obs_f = NOISE_LEVELS[1]

    #oscillator equation
    f = m * ddz + gamma * dz + k * z

    #joint likelihood
    numpyro.sample(
        r"Y_u", 
        dist.Normal(z, sigma_obs_u).to_event(1), 
        obs=Y_u
    )
    numpyro.sample(
        r"Y_f", 
        dist.Normal(f, sigma_obs_f).to_event(1), 
        obs=Y_f
    )


def main(
        layers, train, data_size, num_warmup, num_samples
):

    data = jnp.load('data/oscilator2_data.npy', allow_pickle=True).item()
    X, Y, Y_f = data['X'], data['Y'], data['Y_f']


# Sample training points
    X_u, u_train , X_f, f_train, X_test, u_test, f_test = sample_training_points_space_filling(
                                                            X, Y, Y_f, data_size, noise_levels=NOISE_LEVELS,seed=0,
                                                        )

    X_train = (X_u, X_f)
    Y_train = (u_train, f_train)

    try:
        render = numpyro.render_model(BPINN, (X_train, Y_train, layers))
        render.render("plots/BPINN/bnn_oscilator1_pinn")
    except:
        render = None
        print("Module not installed. (pip install graphviz), (sudo apt-get install graphviz)")
   
    inf_key = jax.random.PRNGKey(0)

    name = f"BPINN_{data_size}_{layers}"
    path = f"data/mcmc_samples/"+name+"_samples.npy"
    if train:
        mcmc = MCMC(NUTS(BPINN), num_warmup=num_warmup, num_samples=num_samples)
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
        
    #predictive
    predictive = Predictive(BPINN, samples, return_sites=["Y_u", "Y_f"])
    predictions = predictive(jax.random.PRNGKey(1), (X_test, X_test), None, layers)


    # Plot the predictions
    mean_prediction_u = jnp.mean(predictions["Y_u"], axis=0).ravel()
    stddev_prediction_u = jnp.std(predictions["Y_u"], axis=0).ravel()

    mean_prediction_f = jnp.mean(predictions["Y_f"], axis=0).ravel()
    stddev_prediction_f = jnp.std(predictions["Y_f"], axis=0).ravel()

    label = f"BPINN/oscilator1_{data_size}_{layers}"
    plot(
        X, u_test, X_u, u_train, X_test, mean_prediction_u, stddev_prediction_u, 
        f_test, X_f, f_train, mean_prediction_f, stddev_prediction_f, label
    )
    #boxplot_physical_parameters(
    #    samples, 
    #    ["m", "gamma", "k"],
    #    title=f"BPINN/params_oscilator1_{data_size}_{layers}"
    #)
    print("Plots saved in plots/ directory")
    print("relative L2 error relL2(u, u_hat): ", relative_l2_error(u_test.ravel(), mean_prediction_u))
    print("relative L2 error relL2(f, f_hat): ", relative_l2_error(f_test.ravel(), mean_prediction_f))

    #plot_distribution(samples, "m", label)
    #plot_distribution(samples, "gamma", label)
    #plot_distribution(samples, "k", label)

    plot_distributions(samples, ["m", "gamma", "k"], label)
def plot(
        X,
        u_test,
        X_u,
        u_train,
        X_test,
        mean_prediction_u,
        stddev_prediction_u,
        f_test,
        X_f,
        f_train,
        mean_prediction_f,
        stddev_prediction_f,
        label,
):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(X, u_test, 'b--', label='True function')
    ax[0].plot(X_u, u_train, 'ro', label=r'$u$ training samples')
    ax[0].plot(X_test, mean_prediction_u, 'g', label='u prediction')
    ax[0].fill_between(
        X_test.ravel(), 
        mean_prediction_u - 2 * stddev_prediction_u, 
        mean_prediction_u + 2 * stddev_prediction_u, 
        color='g', alpha=0.4, label=r"$2\sigma$ uncertainty"
    )
    ax[0].legend()
    ax[0].set_ylabel('u')
    #ax[0].set_xlabel('x')
    ax[0].set_title(r'$u$ prediction')

    ax[1].plot(X, f_test, 'b--', label='true function')
    ax[1].plot(X_f, f_train, 'ro', label=r'$f$ training samples')
    ax[1].plot(X_test, mean_prediction_f, 'g', label='Y_f prediction')
    ax[1].fill_between(
        X_test.ravel(), 
        mean_prediction_f - 2 * stddev_prediction_f, 
        mean_prediction_f + 2 * stddev_prediction_f, 
        color='g', alpha=0.4, label=r"$2\sigma$ uncertainty"
    )
    ax[1].set_ylabel('f')
    ax[1].set_xlabel('x')
    ax[1].legend()
    ax[1].set_title(r'$f$ prediction')
    plt.savefig("plots/"+label+".png", dpi=300, bbox_inches='tight')

def boxplot_physical_parameters(
        samples,
        labels,
        title,
):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    labels_plot = ["$m$", r"$\gamma$", "$k$"]
    for i, label in enumerate(labels):
        ax[i].boxplot(samples[label], vert=False)
        ax[i].set_title(labels_plot[i])
    #fig.suptitle(title)
    plt.savefig("plots/"+title+".png", dpi=300, bbox_inches='tight')


def plot_distribution(samples, param_name, label):
    data = samples[param_name]
    
    # Calculate summary statistics
    mean = jnp.mean(data)
    std = jnp.std(data)
    median = jnp.median(data)
    perc_5 = jnp.percentile(data, 5)
    perc_95 = jnp.percentile(data, 95)
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30, color='blue', stat='density')
    
    # Annotate the plot with summary statistics
    plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='g', linestyle='-', label=f'Median: {median:.2f}')
    plt.axvline(perc_5, color='b', linestyle='--', label=f'5th Percentile: {perc_5:.2f}')
    plt.axvline(perc_95, color='b', linestyle='--', label=f'95th Percentile: {perc_95:.2f}')
    
    plt.title(f'Distribution of {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("plots/"+label+f"_{param_name}_distribution.png", dpi=300, bbox_inches='tight')

def plot_distributions(samples, param_names, label):
    num_params = len(param_names)
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 6 * num_params))
    
    for i, param_name in enumerate(param_names):
        data = samples[param_name]
        
        # Calculate summary statistics
        mean = jnp.mean(data)
        std = jnp.std(data)
        median = jnp.median(data)
        perc_5 = jnp.percentile(data, 5)
        perc_95 = jnp.percentile(data, 95)
        
        # Plot the distribution
        sns.histplot(data, kde=True, bins=30, stat='density', ax=axes[i], color='blue')
        
        # Annotate the plot with summary statistics
        axes[i].axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
        axes[i].axvline(median, color='g', linestyle='-', label=f'Median: {median:.2f}')
        axes[i].axvline(perc_5, color='b', linestyle='--', label=f'5th Percentile: {perc_5:.2f}')
        axes[i].axvline(perc_95, color='b', linestyle='--', label=f'95th Percentile: {perc_95:.2f}')
        
        axes[i].set_title(f'Distribution of {param_name}')
        axes[i].set_xlabel(param_name)
        axes[i].set_ylabel('Density')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f"plots/{label}_combined_distribution.png", dpi=300)
    plt.show()

    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--layers", type=int,nargs="+", default=[40, 40, 1], help="Number of neurons in each layer")
    parser.add_argument("--data_size", type=int, default=50, help="Number of training points")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--num_warmup", type=int, default=1000, help="Number of warmup steps")
    args = parser.parse_args()
    main(**vars(args))
    

