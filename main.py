import numpy as np
from Modified_Newton_Method import ModifiedNewtonMethod
from Truncated_Newton_Method import TruncatedNewtonMethod
from Testers import Test_settings

def main():
    np.random.seed(1)
    test = Test_settings()
    x0 = test.initialize_x0("discrete_boundary_value_problem", 10**3)
    #x0 = np.zeros(10**3)
    h = {'forward_difference': 1e-5, 'backward_difference': 1e-5, 'central_difference': 1e-6}
    NewtonBackTracking_ARG_f2= {'x0': x0,
                                'alpha0': 1,
                                'btmax': 50,
                                'rho': 0.5,
                                'c1': 1e-4,
                                'tolgrad': 1e-7,
                                'kmax': 500,
                                'eta': 0.5,
                                'function': 'discrete_boundary_value_problem',
                                #'solver_linear_system': 'cg',
                                #'H_correction_factor': 3,
                                #'precond': 'yes',
                                'rate_of_convergence': 'superlinear',
                                'derivatives': 'exact',
                                'derivative_method': 'forward',
                                'perturbation': h['forward_difference']
                            }
    
    Nbk= TruncatedNewtonMethod.TruncatedNewtonMethod(**NewtonBackTracking_ARG_f2)
    xk2,fxk2,norm_gradfxk2,k2,x_seq2,bt_seq2= Nbk.Run()
    print('-'*50)
    print(f'Newton Method after {k2} iterations:')
    print(f'xk: {np.round(xk2, 4)}')
    print(f'f2_xk: {np.round(fxk2, 4)}')
    print(f'norma_gradf2_xk: {np.round(norm_gradfxk2, 4)}')
    print('-'*50)

main()

def hessian_approx_tridiagonal(self,x,grad):    #Remove
    """
    Approximate the Hessian of a scalar function f at point x
    assuming a TRIDIAGONAL structure (diagonal + first super/subdiagonal).
    Exploits coloring to minimize function evaluations.
    Optimized for large-scale problems (n ≈ 10⁵).

    Parameters:
    f (callable): The scalar function.
    x (np.ndarray): The point at which to approximate the Hessian.
    grad (np.ndarray): The gradient at x.
    h (float): The step size for finite differences.

    Returns:
    scipy.sparse.csr_matrix: The approximated Hessian (stored as a sparse matrix).
    """
    n = len(x)

    # Use two-coloring (even and odd indices)
    even_indices = np.arange(0, n, 2)
    odd_indices = np.arange(1, n, 2)

    # Initialize arrays to store diagonal and off-diagonal elements
    diagonal = np.zeros(n)
    super_diagonal = np.zeros(n - 1)

    match self.method:
        case 'forward':
            # Perturb even indices in forward direction and compute gradient
            x_perturbed_forward_even = x.copy()
            x_perturbed_forward_even[even_indices] += self.h
            grad_perturbed_forward_even = self.approximate_gradient_parallel(x_perturbed_forward_even)

            # Perturb odd indices in forward direction and compute gradient
            x_perturbed_forward_odd = x.copy()
            x_perturbed_forward_odd[odd_indices] += self.h
            grad_perturbed_forward_odd = self.approximate_gradient_parallel(x_perturbed_forward_odd)

            # Compute diagonal and super-diagonal for even indices using forward difference
            diagonal[even_indices] = (grad_perturbed_forward_even[even_indices] - grad[even_indices]) / self.h
            valid_super_even = even_indices[even_indices < n - 1]  # Ensure i+1 is in bounds
            super_diagonal[valid_super_even] = (grad_perturbed_forward_even[valid_super_even + 1] - grad[valid_super_even + 1]) / self.h

            # Compute diagonal and super-diagonal for odd indices using forward difference
            diagonal[odd_indices] = (grad_perturbed_forward_odd[odd_indices] - grad[odd_indices]) / self.h
            valid_super_odd = odd_indices[odd_indices < n - 1]  # Ensure i+1 is in bounds
            super_diagonal[valid_super_odd] = (grad_perturbed_forward_odd[valid_super_odd + 1] - grad[valid_super_odd + 1]) / self.h

        case 'backward':
            # Perturb even indices in backward direction and compute gradient
            x_perturbed_backward_even = x.copy()
            x_perturbed_backward_even[even_indices] -= self.h
            grad_perturbed_backward_even = self.approximate_gradient_parallel(x_perturbed_backward_even)

            # Perturb odd indices in backward direction and compute gradient
            x_perturbed_backward_odd = x.copy()
            x_perturbed_backward_odd[odd_indices] -= self.h
            grad_perturbed_backward_odd = self.approximate_gradient_parallel(x_perturbed_backward_odd)

            # Compute diagonal and super-diagonal for even indices using backward difference
            diagonal[even_indices] = (grad[even_indices] - grad_perturbed_backward_even[even_indices]) / self.h
            valid_super_even = even_indices[even_indices < n - 1]  # Ensure i+1 is in bounds
            super_diagonal[valid_super_even] = (grad[valid_super_even + 1] - grad_perturbed_backward_even[valid_super_even + 1]) / self.h

            # Compute diagonal and super-diagonal for odd indices using backward difference
            diagonal[odd_indices] = (grad[odd_indices] - grad_perturbed_backward_odd[odd_indices]) / self.h
            valid_super_odd = odd_indices[odd_indices < n - 1]  # Ensure i+1 is in bounds
            super_diagonal[valid_super_odd] = (grad[valid_super_odd + 1] - grad_perturbed_backward_odd[valid_super_odd + 1]) / self.h

        case 'centered':
            # Perturb even indices in forward direction and compute gradient
            x_perturbed_forward_even = x.copy()
            x_perturbed_forward_even[even_indices] += self.h
            grad_perturbed_forward_even = self.approximate_gradient_parallel(x_perturbed_forward_even)

            # Perturb odd indices in forward direction and compute gradient
            x_perturbed_forward_odd = x.copy()
            x_perturbed_forward_odd[odd_indices] += self.h
            grad_perturbed_forward_odd = self.approximate_gradient_parallel(x_perturbed_forward_odd)

            # Perturb even indices in backward direction and compute gradient
            x_perturbed_backward_even = x.copy()
            x_perturbed_backward_even[even_indices] -= self.h
            grad_perturbed_backward_even = self.approximate_gradient_parallel(x_perturbed_backward_even)

            # Perturb odd indices in backward direction and compute gradient
            x_perturbed_backward_odd = x.copy()
            x_perturbed_backward_odd[odd_indices] -= self.h
            grad_perturbed_backward_odd = self.approximate_gradient_parallel(x_perturbed_backward_odd)

            # Compute diagonal and super-diagonal for even indices using centered difference
            diagonal[even_indices] = (grad_perturbed_forward_even[even_indices] - grad_perturbed_backward_even[even_indices]) / (2 * self.h)
            valid_super_even = even_indices[even_indices < n - 1]  # Ensure i+1 is in bounds
            super_diagonal[valid_super_even] = (grad_perturbed_forward_even[valid_super_even + 1] - grad_perturbed_backward_even[valid_super_even + 1]) / (2 * self.h)

            # Compute diagonal and super-diagonal for odd indices using centered difference
            diagonal[odd_indices] = (grad_perturbed_forward_odd[odd_indices] - grad_perturbed_backward_odd[odd_indices]) / (2 * self.h)
            valid_super_odd = odd_indices[odd_indices < n - 1]  # Ensure i+1 is in bounds
            super_diagonal[valid_super_odd] = (grad_perturbed_forward_odd[valid_super_odd + 1] - grad_perturbed_backward_odd[valid_super_odd + 1]) / (2 * self.h)

    # Construct the sparse Hessian matrix
    #hessian = diags([super_diagonal, diagonal, super_diagonal], offsets=[-1, 0, 1], shape=(n, n), format="csr")

    #return hessian.tocsr()