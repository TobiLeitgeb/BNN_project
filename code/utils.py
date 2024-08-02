import jax
from typing import Tuple
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def get_data(
        X: jax.Array,
        Y: jax.Array,
        f: jax.Array,
        *,
        split: Tuple[float, float, float] = (0.8, 0.1),
        dense: bool = False,
        key: jax.random.PRNGKey,
):
    #split data into training testing and validation
    n = X.shape[0]
    n_train = int(n * split[0])
    n_test = int(n * split[1])
   

    #shuffle data
    key, subkey = jax.random.split(key)
    idx = jax.random.permutation(subkey, n)
    X = X[idx, :]
    Y = Y[idx, :]

    #split data
    X_train, Y_train = jnp.array(X[:n_train, :]), jnp.array(Y[:n_train, :])
    X_test, Y_test = jnp.array(X[n_train:n_train + n_test, :]), jnp.array(Y[n_train:n_train + n_test, :])
    X_val, Y_val = jnp.array(X[n_train + n_test:, :]), jnp.array(Y[n_train + n_test:, :])

    if dense:
        X_train = jnp.repeat(X_train, len(f), axis=0)
        X_train = jnp.concatenate([X_train, jnp.tile(f, len(X_train)//len(f))[:,None]], axis=1)
        X_test = jnp.repeat(X_test, len(f), axis=0)
        X_test = jnp.concatenate([X_test, jnp.tile(f, len(X_test)//len(f))[:,None]], axis=1)
        X_val = jnp.repeat(X_val, len(f), axis=0)
        X_val = jnp.concatenate([X_val, jnp.tile(f, len(X_val)//len(f))[:,None]], axis=1)
        return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)
    else:
        return (X_train, Y_train), (X_test, Y_test), (X_val, Y_val)
    
def plot_testset(
            frequency: jax.Array, 
            test_y: jax.Array, 
            prediction: jax.Array, 
            var_pred: jax.Array=None, 
            grid: tuple = (4,4),
            figsize: tuple = (15, 15)
        )-> plt.figure:
        """
        Plots the test set predictions and true values.

        Args:
            frequency (ndarray): The frequency values.
            test_y (ndarray): The true output values of shape (n_test, p).
            prediction (ndarray): The predicted output values of shape (n_test, p).
            grid (tuple): The grid size for subplots.
        """
        fig, ax  = plt.subplots(grid[0], grid[1], figsize=figsize)
        
        if grid[0] == 1 and grid[1] == 1:
            ax = [ax]

        ax = ax.flatten()

        for i in range(grid[0]*grid[1]):
            ax[i].plot(frequency, test_y[i, :], color='black')
            ax[i].plot(frequency, prediction[i, :], color='blue')
            if var_pred is not None:
                pred_std = np.sqrt(var_pred[i, :])
                ax[i].fill_between(
                    frequency, 
                    prediction[i, :] - 2*pred_std, 
                    prediction[i, :] + 2*pred_std, 
                    color='blue', alpha=0.2
                )
        
            ax[i].set_ylabel('dB')
            ax[i].legend(['True', 'Predicted'])
            #ax[i].grid()
            ax[i].set_xscale('log')

            error = minmaxrmspe(test_y[i, :][None,:], prediction[i, :][None,:])
            ax[i].set_title('minmaxRMSE: {:.2f}%'.format(error))
            #add axis only for the last row
            if i >= grid[0]*(grid[1]-1):
                ax[i].set_xlabel('Frequency (Hz)')
            plt.tight_layout()
        return ax, fig

def minmaxrmspe(
            test_y: jax.Array, 
            prediction: jax.Array
        )-> float:
        """
        Calculates the Root Mean Square Percentage Error.

        Args:
            test_y (ndarray): The true output values of shape (n, p).
            prediction (ndarray): The predicted output values of shape (n, p).

        Returns:
            float: The Root Mean Square Percentage Error.
        """
        p = test_y.shape[1]
        error = jnp.mean(
            jnp.sqrt(1 / p * jnp.sum((test_y - prediction) ** 2, axis=1))
            * 100 / (jnp.max(test_y, axis=1) - jnp.min(test_y, axis=1))
        )
        return error

def plot_samples(
          f,
          Y_test,
          predicted_mean,
          samples,
):
    fig, ax = plt.subplots()
    ax.set_ylabel('dB')
    ax.set_xscale('log')
    ax.plot(f, samples, alpha = 0.6)
    ax.plot(f, Y_test, "blue", label="Test sample")
    ax.plot(f, predicted_mean, "r--", label="mean")

    ax.legend()