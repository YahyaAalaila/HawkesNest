import numpy as np

def create_polynomial_kernel_func(spatial_coeffs: list, temporal_coeffs: list):
    """
    A factory that creates and returns a polynomial kernel function.
    """
    s_coeffs = np.array(spatial_coeffs)
    t_coeffs = np.array(temporal_coeffs)

    def polynomial_kernel(s: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """The actual kernel function that will be passed to SpaceTimeKernel."""
        s_poly = sum(c * s**i for i, c in enumerate(s_coeffs))
        t_poly = sum(c * tau**i for i, c in enumerate(t_coeffs))
        result = s_poly + t_poly
        return np.maximum(0, result)

    return polynomial_kernel