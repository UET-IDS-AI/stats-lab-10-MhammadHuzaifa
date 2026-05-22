import numpy as np

# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).

    Formula:
    f_XY(x,y) = 1 / (2*pi*sigma_x*sigma_y*sqrt(1-rho^2))
                * exp( -Q / (2*(1-rho^2)) )
    where
    Q = ((x-mu_x)/sigma_x)^2
        - 2*rho*((x-mu_x)/sigma_x)*((y-mu_y)/sigma_y)
        + ((y-mu_y)/sigma_y)^2
    """
    norm_x = (x - mu_x) / sigma_x
    norm_y = (y - mu_y) / sigma_y
    Q = norm_x**2 - 2 * rho * norm_x * norm_y + norm_y**2
    coeff = 1.0 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    return coeff * np.exp(-Q / (2 * (1 - rho**2)))


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    f_X(x) = 1/(sqrt(2*pi)*sigma_x) * exp(-0.5 * ((x-mu_x)/sigma_x)^2)
    """
    coeff = 1.0 / (np.sqrt(2 * np.pi) * sigma_x)
    return coeff * np.exp(-0.5 * ((x - mu_x) / sigma_x) ** 2)


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    f_Y(y) = 1/(sqrt(2*pi)*sigma_y) * exp(-0.5 * ((y-mu_y)/sigma_y)^2)
    """
    coeff = 1.0 / (np.sqrt(2 * np.pi) * sigma_y)
    return coeff * np.exp(-0.5 * ((y - mu_y) / sigma_y) ** 2)


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix:
    [[sigma_x^2, rho*sigma_x*sigma_y],
     [rho*sigma_x*sigma_y, sigma_y^2]]
    """
    cov_xy = rho * sigma_x * sigma_y
    return np.array([[sigma_x**2, cov_xy],
                     [cov_xy, sigma_y**2]])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    """
    Numerically approximate integral of joint Gaussian PDF
    over the rectangle:
    [mu_x - 4*sigma_x, mu_x + 4*sigma_x] x [mu_y - 4*sigma_y, mu_y + 4*sigma_y]
    using the trapezoidal rule on a 2D grid.
    """
    x = np.linspace(mu_x - 4 * sigma_x, mu_x + 4 * sigma_x, n)
    y = np.linspace(mu_y - 4 * sigma_y, mu_y + 4 * sigma_y, n)
    X, Y = np.meshgrid(x, y)
    Z = joint_gaussian_pdf(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho)
    # Integrate along y (axis=0) then along x
    return np.trapz(np.trapz(Z, y, axis=0), x)


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate n samples from a jointly Gaussian distribution.
    Return x_samples, y_samples.
    """
    rng = np.random.default_rng(seed)
    cov = covariance_matrix(sigma_x, sigma_y, rho)
    samples = rng.multivariate_normal(mean=[mu_x, mu_y], cov=cov, size=n)
    return samples[:, 0], samples[:, 1]


def sample_means(x_samples, y_samples):
    """Return sample means of X and Y."""
    return np.mean(x_samples), np.mean(y_samples)


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return 2 by 2 sample covariance matrix using denominator n-1.
    """
    return np.cov(x_samples, y_samples, ddof=1)


def sample_correlation(x_samples, y_samples):
    """Return sample correlation coefficient."""
    corr_matrix = np.corrcoef(x_samples, y_samples)
    return corr_matrix[0, 1]


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables, zero correlation (rho=0)
    implies independence. Return True if rho is zero, else False.
    """
    return rho == 0.0


def zero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0 and check that sample covariance
    is approximately zero. Return True if close to zero, else False.
    """
    x, y = generate_joint_gaussian_samples(n=n, rho=0.0, seed=1)
    cov = sample_covariance_matrix(x, y)
    return np.abs(cov[0, 1]) < 0.1   # n=100000 gives very small error


def nonzero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0.6 and check that sample covariance
    is close to rho*sigma_x*sigma_y (i.e., 3.6).
    Return True if within tolerance, else False.
    """
    x, y = generate_joint_gaussian_samples(n=n, rho=0.6, seed=1)
    cov = sample_covariance_matrix(x, y)
    true_cov = 0.6 * 2 * 3   # rho * sigma_x * sigma_y = 3.6
    return np.abs(cov[0, 1] - true_cov) < 0.1
