import numpy as np
from scipy.stats import norm

import warnings
warnings.filterwarnings(action='ignore')

def gaussian_pdf(mean, var, y):
    return norm(mean, np.sqrt(var)).pdf(y)


def E_stage(X_train_combined, y_train_combined, GPs, P_z):
    """Perform the Expectation stage of the EM algorithm (returns updated `P_z`)"""
    
    n = len(GPs)
    N = np.zeros(P_z.shape)

    # compute observation likelihoods
    for j, i in np.ndindex(P_z.shape):
        q = X_train_combined[j]
        y = y_train_combined[j]
        # TODO: determine if this is the correct computation of var
        # TODO: make this more efficient
        y_hat, var = GPs[i].query([q])
        N[j, i] = gaussian_pdf(y_hat, var, y)
        # if np.all(np.isfinite(N[j, i])) == False:
        #     print('NaN or Inf in N[j, i]')
        #     print(y_hat)
        #     print(var)
        #     print(y)
        #     exit()
    
    # print(N)
    
    # update P_z (as one batch)
    P_z_new = P_z.copy()
    for j, i in np.ndindex(P_z.shape):
        P_z_new[j, i] = (P_z[j, i] * N[j, i]) / (sum(P_z[j, k] * N[j, k] for k in range(n)) + 1e-9)
        # if np.all(np.isfinite(P_z_new[j, i])) == False:
        #     print('NaN or Inf in P_z_new[j, i]')
        #     print(P_z_new[j, i])
        #     print(P_z[j, i] * N[j, i])
        #     print(sum(P_z[j, k] * N[j, k] for k in range(n)))
        #     print('above are components')
        #     print(P_z)
        #     print(N)
        #     exit()
    
    return P_z_new


def M_stage(GPs, P_z):
    """Perform the Maximization stage of the EM algorithm (modifies `GPs` in-place)"""
    j = 0
    for i, gp in enumerate(GPs):
        for j_local in range(len(gp.X_train)):
            # prevent divide by 0
            gp.psi_diagonals[j_local] = gp.σ_noises[j_local] / np.nanmax([P_z[j, i], 1e-9])
            j += 1


def EM_algorithm(X_train_combined, y_train_combined, GPs, P_z, return_P_Z_history=False):
    """
    Run the EM-Algorithm until convergence.
    Modifies `GPs` in place and return updated `P_z`.
    """
    # if np.all(np.isfinite(P_z)) == False:
    #     print('NaN or Inf in P_z_new[j, i]')
    #     print(P_z)
    #     exit()
    # Run EM-Algorithm until convergence
    P_z_history = []
    epsilon = 1e-4
    for _ in range(20):
        P_z_history.append(P_z.copy())
        P_z_new = E_stage(X_train_combined, y_train_combined, GPs, P_z)

        change = np.linalg.norm(P_z - P_z_new)
        # print(change)
        P_z = P_z_new
        
        M_stage(GPs, P_z)

        if change < epsilon: break

    P_z_history.append(P_z.copy())

    if return_P_Z_history:
        return P_z, P_z_history
    else:
        return P_z
