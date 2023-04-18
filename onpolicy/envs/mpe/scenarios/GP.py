import numpy as np
import scipy


def rbf_kernel(x1, x2, σf=1.0, l=1.0):
    sq_norm = scipy.spatial.distance.cdist(x1, x2, "sqeuclidean")
    return σf**2 * np.exp(-sq_norm / (2 * l**2))


# Gaussian process posterior
def GP_posterior(X1, y1, X2, kernel_func, σ_noise=0.0):
    """
    Given test points `X2`, calculate the corresponding posterior mean
    and covariance matrix for `y2` based on the training data `(X1, y1)`.

    Returns: `μ2, Σ2`
    """
    # Kernel of the observations
    Σ11 = kernel_func(X1, X1)

    # Account for measurement noise
    Σ11 += σ_noise**2 * np.eye(len(X1))

    # Kernel of observations vs to-predict
    Σ12 = kernel_func(X1, X2)

    # Compute (Σ11^(-1) * Σ12)^T
    try:
        solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
    except scipy.linalg.LinAlgError:
        raise ValueError(f"Given priors result in singular matrix (one of the following):\n{Σ11=}\n{Σ12=}")

    # Compute posterior mean
    μ2 = solved @ y1
    
    # Compute the posterior covariance
    Σ22 = kernel_func(X2, X2)
    Σ2 = Σ22 - (solved @ Σ12)

    return μ2, Σ2  # mean, covariance



class GP:
    def __init__(self, kernel_func=rbf_kernel):
        self.kernel_func = kernel_func
        self.σ_noises = []
        self.psi_diagonals = []

        self.X_train = []
        self.y_train = []
    
    def add_training_point(self, x, y, σ_noise):
        self.X_train.append(x)
        self.y_train.append(y)
        self.σ_noises.append(σ_noise)
        self.update_psi_diagonals()
    
    def add_training_points(self, xs, ys, σ_noises):
        self.X_train.extend(xs)
        self.y_train.extend(ys)
        self.σ_noises.extend(σ_noises)
        self.update_psi_diagonals()
    
    def update_psi_diagonals(self):
        self.psi_diagonals = [σ**2 for σ in self.σ_noises]

    def query(self, X_test):
        """
        Given test points `X_test`, calculate the corresponding posterior mean
        and covariance matrix for `y_test` based on the training data.

        Returns: `μ2, Σ2`
        """
        X1 = np.array(self.X_train)
        y1 = np.array(self.y_train)
        X2 = np.array(X_test)
        psi = np.diag(self.psi_diagonals)
        # print(psi)

        # Kernel of the observations
        Σ11 = self.kernel_func(X1, X1)

        # Account for measurement noise
        Σ11 += psi

        # Kernel of train points vs test points
        Σ12 = self.kernel_func(X1, X2)

        # Compute (Σ11^(-1) * Σ12)^T
        try:
            solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
        except scipy.linalg.LinAlgError:
            raise ValueError(f"Given priors result in singular matrix (one of the following):\n{Σ11=}\n{Σ12=}")

        # Compute posterior mean
        μ2 = solved @ y1
        
        # Compute the posterior covariance
        Σ22 = self.kernel_func(X2, X2)
        Σ2 = Σ22 - (solved @ Σ12)

        return μ2, Σ2  # mean, covariance

