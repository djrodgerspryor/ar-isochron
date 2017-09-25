from math import log, fabs
import numpy as np
import scipy
from scipy import stats
import csv


### General Maths Functions ###

# Given an iterative function and an initial value, keep generating new estimates until the values
# converge to within the given error threshold.
def converge(initial, function, error_threshold_fraction=0.0001, max_iterations=1000, log=False):
    # Start at the given initial value
    current = initial

    i = 0
    while True:
        i += 1

        # Keep-track of what the previous estimate was
        previous = current

        # Generate the next estimate
        current = function(current)

        # Work out how big the change was as a fraction of the current value
        error_fraction = fabs(current - previous) / fabs(current)

        # If we've converged (ie. if the change was sufficiently small)
        if error_fraction < error_threshold_fraction:
            # Then we can stop
            break

        # If we've been going for a long time without stopping
        if i >= max_iterations:
            # Should only happen if the function is accidentally non-convergent because we nuff-nuffed some numbers
            raise Exception("Unable to converge after %d iterations: this function probably doesn't converge at all!" % (i,))

    # We've found a result: yay!

    if log:
        # Log how long convergence took
        print("Converged to within {0:.4e}% in {1:d} iterations".format(error_fraction * 100, i))

    return current


### Data Import ###

X = []
Y = []
X_STD = []
Y_STD = []

with open('data.csv') as f:
    reader = csv.reader(f)
    next(reader, None)

    for (x, y, x_std, y_std) in reader:
        X.append(float(x))
        Y.append(float(y))
        X_STD.append(float(x_std))
        Y_STD.append(float(y_std))

    X = np.array(X)
    Y = np.array(Y)
    X_STD = np.array(X_STD)
    Y_STD = np.array(Y_STD)


### Helper Closure Functions ###

def weights(b):
    res = []
    for (x_std, y_std) in zip(X_STD.tolist(), Y_STD.tolist()):
        res.append(1.0/(y_std**2 + (x_std**2 * b**2)))

    return np.array(res)

def mean(values, weights):
    return np.dot(values, weights) / np.sum(weights)

def b_next(b_prev):
    W_PREV = weights(b_prev)
    Y_MEAN_PREV = mean(Y, W_PREV)
    X_MEAN_PREV = mean(X, W_PREV)

    V = Y - Y_MEAN_PREV
    U = X - X_MEAN_PREV

    return (
        np.dot(
            np.multiply(np.square(W_PREV), V),
            np.multiply(np.square(Y_STD), U) + b_prev * np.multiply(np.square(X_STD), V)
        ) / np.dot(
            np.multiply(np.square(W_PREV), U),
            np.multiply(np.square(Y_STD), U) + b_prev * np.multiply(np.square(X_STD), V)
        )
    )

def mswd(b):
    W = weights(b)
    Y_MEAN = mean(Y, W)
    X_MEAN = mean(X, W)
    a = Y_MEAN - b * X_MEAN

    return (
        np.dot(
            W,
            np.square(Y - (b * X) - a)
        ) / (np.shape(X)[0] - 2)
    )

### Initial Least-Squares Estimate ###

b_init, a_init, init_r_value, init_p_value, init_std_err = stats.linregress(X, Y)

print("Initial least-squares estimate: y={b:.4g}x + {a:.4g} (MSWD: {mswd:.4g})".format(b=b_init, a=a_init, mswd=mswd(b_init)))


### Iteratively Compute Weighted Estimate ###

print("Iteratively computing slope...")

# Do the thing
b = converge(b_init, b_next, log=True)

# Compute derived quantities now that we've calculated the slope
W = weights(b)
Y_MEAN = mean(Y, W)
X_MEAN = mean(X, W)
a = Y_MEAN - b * X_MEAN

print("Finial weighted estimate: y={b:.4g}x + {a:.4g} (MSWD: {mswd:.4g})".format(b=b, a=a, mswd=mswd(b)))

target_mswd = 1 + 2 * ((2/(np.shape(X)[0] - 2)) ** 0.5)

print("Target MSWD: {mswd:.4g}".format(mswd=target_mswd))
