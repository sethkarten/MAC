import numpy as np
import scipy


def gaussian_pdf(mean, var, y):
    return scipy.stats.norm(mean, np.sqrt(var)).pdf(y)


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
    
    # print(N)
    
    # update P_z (as one batch)
    P_z_new = P_z.copy()
    for j, i in np.ndindex(P_z.shape):
        P_z_new[j, i] = (P_z[j, i] * N[j, i]) / sum(P_z[j, k] * N[j, k] for k in range(n))
    
    return P_z_new


def M_stage(GPs, P_z):
    """Perform the Maximization stage of the EM algorithm (modifies `GPs` in-place)"""
    j = 0
    for i, gp in enumerate(GPs):
        for j_local in range(len(gp.X_train)):
            # prevent divide by 0
            gp.psi_diagonals[j_local] = gp.σ_noises[j_local] / max(P_z[j, i], 1e-9)
            j += 1


def EM_algorithm(X_train_combined, y_train_combined, GPs, P_z, return_P_Z_history=False):
    """
    Run the EM-Algorithm until convergence.
    Modifies `GPs` in place and return updated `P_z`.
    """
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
