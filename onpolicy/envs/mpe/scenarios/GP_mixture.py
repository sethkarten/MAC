from EM_algorithm import EM_algorithm
from GP import GP

import numpy as np


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from typing import List


def sklearn2custom(gp_sklearn: GaussianProcessRegressor):
    """
    Convert the given sklearn GP into a custom GP object
    (which is amenable to the EM algorithm)
    """
    rbf_kernel = gp_sklearn.kernel_.k1      # just the RBF component (noise handled separately)
    gp = GP(kernel_func=rbf_kernel)

    σ_noise = gp_sklearn.kernel_.k2.noise_level
    gp.add_training_points(
        gp_sklearn.X_train_,
        gp_sklearn.y_train_,
        σ_noises=[σ_noise] * len(gp_sklearn.X_train_)
    )
    return gp


# Specify kernel with initial hyperparameter estimates
def kernel_initial(
    σf_initial=1.0,         # covariance amplitude
    ell_initial=1.0,        # length scale
    σn_initial=0.1          # noise level
):
    return σf_initial**2 * RBF(length_scale=ell_initial) + WhiteKernel(noise_level=σn_initial)


def mix_GPs(GPs_sklearn: List[GaussianProcessRegressor]):
    """
    Compute the mixture of the given GP's using the EM algorithm to learn the gating function.

    Returns a `predictor` function that take in test points `X_test` and returns the predicted 
    mean and variance at these points `(μ_test, σ2_test)`
    """

    n = len(GPs_sklearn)
    GPs = [sklearn2custom(gp) for gp in GPs_sklearn]

    X_train_combined = np.concatenate([gp.X_train for gp in GPs])
    y_train_combined = np.concatenate([gp.y_train for gp in GPs])

    """
    P_z[j, i] represents P(z(x_j) = i)
    (aka the probability that sample j is best associated with GP component i)
    """

    # # initialize P_z s.t. each sample is assigned completely to the robot that sampled it
    # # NOTE: per discussion with Prof Wenhao, this method is faulty
    # P_z = np.ones((len(X_train_combined), n)) * 0.00001
    # j = 0
    # for i, gp in enumerate(GPs):
    #     for _ in range(len(gp.X_train)):
    #         P_z[j, i] = 1
    #         j += 1

    # initialize P_z uniformly
    P_z = np.ones((len(X_train_combined), n)) / n

    P_z, P_z_history = EM_algorithm(X_train_combined, y_train_combined, GPs, P_z, return_P_Z_history=True)

    """
    Now extend the gating function over the entire space via GP's
    A new GP is trained for each robot i to model P(z(q) = i)
    """

    gating_GPs = []
    for i in range(n):
        gp = GaussianProcessRegressor(
            kernel=kernel_initial(),
            n_restarts_optimizer=10
        )
        gp.fit(X_train_combined, P_z[:,i])
        print(gp.kernel_)
        gating_GPs.append(gp)


    # Actual mixing occurs within predictor function:
    def predictor(X_test):
        # Extend gating function via SoftMax (or something else)
        P_z_test = np.zeros((len(X_test), n))
        denominator = sum(np.exp(gp.predict(X_test)) for gp in gating_GPs)
        for i in range(n):
            numerator = np.exp(gating_GPs[i].predict(X_test))
            P_z_test[:,i] = numerator / denominator

        # # EXPERIMENTING
        # P_z_test[:,i] = gating_GPs[i].predict(X_test)

        # Compute affine combination of component GP's
        μ_mixed_test = np.zeros((len(X_test),))
        σ2_mixed_test = np.zeros((len(X_test),))

        for i in range(n):
            μ, Σ = GPs[i].query(X_test)
            μ_mixed_test += P_z_test[:,i] * μ

        for i in range(n):
            μ, Σ = GPs[i].query(X_test)
            σ2 = np.diag(Σ)
            σ2_mixed_test += P_z_test[:,i] * (σ2 + (μ - μ_mixed_test)**2)
    
        return μ_mixed_test, σ2_mixed_test


    return predictor
