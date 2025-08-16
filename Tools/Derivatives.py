import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix, lil_matrix
from joblib import Parallel, delayed
from Conditions import CheckConditions

class ApproximateDerivatives:
    def __init__(self, function, derivative_method, perturbation):
        """
        Derivative Approximation using Finite Difference methods.
        Input: 
        function : function to be derivated
        derivative_method : differential method to use (exact, forward, backward, central)
        perturbation : small perturbation for finite difference
        """
        self.f = function
        self.derivative_method = derivative_method
        self.perturbation = perturbation
            
    def hessian(self,x,grad,adaptive=False):
        n = len(x)
        H = np.zeros((n, n))

        if adaptive:
            builder = CheckConditions()
            h = builder.build_perturbation_vector(x,self.perturbation)
        else:
            h = self.perturbation

        match self.derivative_method:
            case "forward":
                for i in range(n):
                    for j in range(n):
                        x_fwd_i = np.copy(x)
                        x_fwd_j = np.copy(x)
                        x_fwd_ij = np.copy(x)
                        
                        x_fwd_i[i] += h
                        x_fwd_j[j] += h
                        x_fwd_ij[i] += h
                        x_fwd_ij[j] += h
                        
                        if i == j:
                            H[i, j] = (self.f(x_fwd_i) - 2 * self.f(x) + 
                                       self.f(x - h * np.eye(n)[i])) / (h ** 2)
                        else:
                            H[i, j] = (self.f(x_fwd_ij) - self.f(x_fwd_i) - 
                                       self.f(x_fwd_j) + self.f(x)) / (h ** 2)
                return csr_matrix(H)

            case "backward":
                for i in range(n):
                    for j in range(n):
                        x_bwd_i = np.copy(x)
                        x_bwd_j = np.copy(x)
                        x_bwd_ij = np.copy(x)
                        
                        x_bwd_i[i] -= h
                        x_bwd_j[j] -= h
                        x_bwd_ij[i] -= h
                        x_bwd_ij[j] -= h
                        
                        if i == j:
                            H[i, j] = (self.f(x_bwd_i) - 2 * self.f(x) + 
                                       self.f(x + h * np.eye(n)[i])) / (h ** 2)
                        else:
                            H[i, j] = (self.f(x) - self.f(x_bwd_i) - self.f(x_bwd_j) + 
                                       self.f(x_bwd_ij)) / (h ** 2)
                return csr_matrix(H)

            case "central":
                for i in range(n):
                    for j in range(n):
                        x_fwd_i = np.copy(x)
                        x_bwd_i = np.copy(x)
                        x_fwd_j = np.copy(x)
                        x_bwd_j = np.copy(x)
                        x_fwd_ij = np.copy(x)
                        x_bwd_ij = np.copy(x)

                        x_fwd_i[i] += h
                        x_bwd_i[i] -= h
                        x_fwd_j[j] += h
                        x_bwd_j[j] -= h
                        x_fwd_ij[i] += h
                        x_fwd_ij[j] += h
                        x_bwd_ij[i] -= h
                        x_bwd_ij[j] -= h

                        x_pp = np.copy(x); x_pm = np.copy(x); x_mp = np.copy(x); x_mm = np.copy(x)
                        x_pp[i] += h; x_pp[j] += h
                        x_pm[i] += h; x_pm[j] -= h
                        x_mp[i] -= h; x_mp[j] += h
                        x_mm[i] -= h; x_mm[j] -= h
                        
                        if i == j:
                            H[i, j] = (
                                        self.f(x_fwd_i) - 2 * self.f(x) + self.f(x_bwd_i)
                                        ) / (h ** 2)
                        else:
                            H[i, j] = (
                                        self.f(x_pp) - self.f(x_pm) - self.f(x_mp) + self.f(x_mm)
                                       ) / (4 * h ** 2)
                return csr_matrix(H)
    

class SparseApproximativeDerivatives(object):
    """This class computes derivatives exploiting sparsity of matrices"""
    def __init__(self,f,method,h):
        self.f = f
        self.method = method
        self.h = h
        self.partial_derivative = self.set_partial_derivative()
    
    def set_partial_derivative(self):
        match self.method:
            case "forward":
                return self.fwd_partial
            case "backward":
                return self.bwd_partial
            case "central":
                return self.cntrl_partial

    def fwd_partial(self, x, i, h):
        x_plus = np.copy(x)
        x_plus[i] += h
        return (self.f(x_plus) - self.f(x)) / h

    def bwd_partial(self, x, i, h):
        x_minus = np.copy(x)
        x_minus[i] -= h
        return (self.f(x) - self.f(x_minus)) / h
    
    def cntrl_partial(self, x, i, h):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        return (self.f(x_plus) - self.f(x_minus)) / (2 * h) 


    def approximate_gradient_parallel(self, x, n_jobs=-1, adaptive=False):
        """
        Approximate the gradient of a scalar function f at point x using finite differences.
        Optimized for large-scale problems (n ≈ 10⁵).

        Parameters:
        x (np.ndarray): The point at which to approximate the gradient.
        h (float): The step size for finite differences.
        method (str): Finite difference method ('forward', 'backward', 'centered').
        n_jobs (int): Number of parallel jobs for gradient computation. Use -1 for all cores.

        Returns:
        np.ndarray: The approximated gradient.
        """
        n = len(x)

        if adaptive:
            builder = CheckConditions()
            h_vec = builder.build_perturbation_vector(x, self.h)
            # Use the per-coordinate step size for each partial derivative
            grad = Parallel(n_jobs=n_jobs)(
                delayed(self.partial_derivative)(x, i, h_vec[i]) for i in range(n)
            )
        else:
            step = self.h
            grad = Parallel(n_jobs=n_jobs)(
                delayed(self.partial_derivative)(x, i, step) for i in range(n)
            )

        return np.array(grad)

    def hessian_vector_product(self, x, v, grad, adaptive = False):
        """
        Approximate the Hessian-vector product (H @ v) using finite differences.

        Parameters:
        f (callable): The function whose Hessian we want to approximate.
        x (np.ndarray): Input point (n-dimensional vector).
        v (np.ndarray): Vector to multiply with the Hessian (same shape as x).
        h (float): Step size for finite differences.

        Returns:
        np.ndarray: Approximated Hessian-vector product (Hv).
        """

        match self.method:
            case "forward":
                x_plus = x + self.h * v
                grad_x_plus = self.approximate_gradient_parallel(x_plus)
                Hv = (grad_x_plus - grad) / self.h
            case "backward":
                x_minus = x - self.h * v
                grad_x_minus = self.approximate_gradient_parallel(x_minus)
                Hv = (grad -  grad_x_minus) / self.h
            case "central":
                x_plus = x + self.h * v
                x_minus = x - self.h * v
                grad_x_plus = self.approximate_gradient_parallel(x_plus)
                grad_x_minus = self.approximate_gradient_parallel(x_minus)
                Hv = (grad_x_plus - grad_x_minus) / (2 * self.h)
        return Hv

    def hessian_approx_extendedros(self, x, grad, adaptive=False):
        """
        Approximate the Hessian of a scalar function f at point x using finite differences
        and a two-coloring technique, storing the result in a sparse matrix.

        Optimized for large-scale problems (n ≈ 10⁵).

        Parameters:
        x (np.ndarray): The point at which to approximate the Hessian.
        grad (np.ndarray): The gradient at x.
        adaptive (bool): If True, build and use a per-coordinate perturbation vector.

        Returns:
        scipy.sparse.csr_matrix: The approximated Hessian (stored as a sparse matrix).
        """
        n = len(x)

        # Build (possibly vector) step size
        if adaptive:
            builder = CheckConditions()
            h_vec = builder.build_perturbation_vector(x, self.h)
        else:
            h_vec = self.h

        # Define coloring scheme (even and odd indices)
        even_indices = np.arange(0, n, 2)
        odd_indices = np.arange(1, n, 2)

        # Precompute non-zero indices and values for the sparse matrix
        rows, cols, data = [], [], []

        # Helpers to select the right step per index set
        def _step_for(idx):
            return h_vec if np.isscalar(h_vec) else h_vec[idx]

        match self.method:
            case 'forward':
                # Perturb even indices
                x_perturbed_forward_even = x.copy()
                x_perturbed_forward_even[even_indices] += _step_for(even_indices)
                grad_perturbed_forward_even = self.approximate_gradient_parallel(
                    x_perturbed_forward_even, adaptive=adaptive
                )

                # Perturb odd indices
                x_perturbed_forward_odd = x.copy()
                x_perturbed_forward_odd[odd_indices] += _step_for(odd_indices)
                grad_perturbed_forward_odd = self.approximate_gradient_parallel(
                    x_perturbed_forward_odd, adaptive=adaptive
                )

                # Diagonal elements (even indices)
                rows.extend(even_indices)
                cols.extend(even_indices)
                denom_even = _step_for(even_indices)
                data.extend((grad_perturbed_forward_even[even_indices] - grad[even_indices]) / denom_even)

                # Diagonal elements (odd indices)
                rows.extend(odd_indices)
                cols.extend(odd_indices)
                denom_odd = _step_for(odd_indices)
                data.extend((grad_perturbed_forward_odd[odd_indices] - grad[odd_indices]) / denom_odd)

                # Off-diagonal elements (even and odd pairs)
                valid_super_diag = even_indices[even_indices + 1 < n]
                denom_sd = _step_for(valid_super_diag)
                off_diag_values = (grad_perturbed_forward_even[valid_super_diag + 1] - grad[valid_super_diag + 1]) / denom_sd

            case 'backward':
                # Perturb even indices in backward direction
                x_perturbed_backward_even = x.copy()
                x_perturbed_backward_even[even_indices] -= _step_for(even_indices)
                grad_perturbed_backward_even = self.approximate_gradient_parallel(
                    x_perturbed_backward_even, adaptive=adaptive
                )

                # Perturb odd indices in backward direction
                x_perturbed_backward_odd = x.copy()
                x_perturbed_backward_odd[odd_indices] -= _step_for(odd_indices)
                grad_perturbed_backward_odd = self.approximate_gradient_parallel(
                    x_perturbed_backward_odd, adaptive=adaptive
                )

                # Diagonal elements (even indices)
                rows.extend(even_indices)
                cols.extend(even_indices)
                denom_even = _step_for(even_indices)
                data.extend((grad[even_indices] - grad_perturbed_backward_even[even_indices]) / denom_even)

                # Diagonal elements (odd indices)
                rows.extend(odd_indices)
                cols.extend(odd_indices)
                denom_odd = _step_for(odd_indices)
                data.extend((grad[odd_indices] - grad_perturbed_backward_odd[odd_indices]) / denom_odd)

                # Off-diagonal elements (even and odd pairs)
                valid_super_diag = even_indices[even_indices + 1 < n]
                denom_sd = _step_for(valid_super_diag)
                off_diag_values = (grad[valid_super_diag + 1] - grad_perturbed_backward_even[valid_super_diag + 1]) / denom_sd

            case 'central':
                # Perturb even indices
                x_perturbed_forward_even = x.copy()
                x_perturbed_forward_even[even_indices] += _step_for(even_indices)
                grad_perturbed_forward_even = self.approximate_gradient_parallel(
                    x_perturbed_forward_even, adaptive=adaptive
                )

                # Perturb odd indices
                x_perturbed_forward_odd = x.copy()
                x_perturbed_forward_odd[odd_indices] += _step_for(odd_indices)
                grad_perturbed_forward_odd = self.approximate_gradient_parallel(
                    x_perturbed_forward_odd, adaptive=adaptive
                )

                # Perturb even indices in backward direction
                x_perturbed_backward_even = x.copy()
                x_perturbed_backward_even[even_indices] -= _step_for(even_indices)
                grad_perturbed_backward_even = self.approximate_gradient_parallel(
                    x_perturbed_backward_even, adaptive=adaptive
                )

                # Perturb odd indices in backward direction
                x_perturbed_backward_odd = x.copy()
                x_perturbed_backward_odd[odd_indices] -= _step_for(odd_indices)
                grad_perturbed_backward_odd = self.approximate_gradient_parallel(
                    x_perturbed_backward_odd, adaptive=adaptive
                )

                # Diagonal elements (even indices)
                rows.extend(even_indices)
                cols.extend(even_indices)
                denom_even = _step_for(even_indices)
                data.extend((grad_perturbed_forward_even[even_indices] - grad_perturbed_backward_even[even_indices]) / (2 * denom_even))

                # Diagonal elements (odd indices)
                rows.extend(odd_indices)
                cols.extend(odd_indices)
                denom_odd = _step_for(odd_indices)
                data.extend((grad_perturbed_forward_odd[odd_indices] - grad_perturbed_backward_odd[odd_indices]) / (2 * denom_odd))

                # Off-diagonal elements (even and odd pairs)
                valid_super_diag = even_indices[even_indices + 1 < n]
                denom_sd = _step_for(valid_super_diag)
                off_diag_values = (grad_perturbed_forward_even[valid_super_diag + 1] - grad_perturbed_backward_even[valid_super_diag + 1]) / (2 * denom_sd)

        # Upper triangular part
        rows.extend(valid_super_diag)
        cols.extend(valid_super_diag + 1)
        data.extend(off_diag_values)

        # Lower triangular part (symmetric)
        rows.extend(valid_super_diag + 1)
        cols.extend(valid_super_diag)
        data.extend(off_diag_values)

        # Construct the sparse matrix directly in CSR format
        hessian = csr_matrix((data, (rows, cols)), shape=(n, n))

        return hessian
    
    def hessian_approx_tridiagonal(self, x, grad, adaptive=False):
        """
        Approximate the Hessian of a scalar function f at point x
        assuming a TRIDIAGONAL structure (diagonal + first super/subdiagonal).
        Uses three-coloring to avoid off-diagonal contamination.

        Parameters:
        x (np.ndarray): Point where to approximate the Hessian
        grad (np.ndarray): Gradient at x
        adaptive (bool): If True, build and use a per-coordinate perturbation vector.
        """
        n = len(x)

        # Build (possibly vector) step size
        if adaptive:
            builder = CheckConditions()
            h_vec = builder.build_perturbation_vector(x, self.h)
        else:
            h_vec = self.h

        # Base gradient at x (use provided grad if available)
        base_grad = grad if grad is not None else self.approximate_gradient_parallel(x, adaptive=adaptive)

        # Three-coloring to avoid contamination at distance 1
        color_classes = [np.arange(shift, n, 3) for shift in range(3)]

        # Initialize arrays to store diagonal and off-diagonal elements
        diagonal = np.zeros(n)
        super_diagonal = np.zeros(n - 1)

        # Helper for step selection
        def _step_for(idx):
            return h_vec if np.isscalar(h_vec) else h_vec[idx]

        match self.method:
            case 'forward':
                for idx in color_classes:
                    if idx.size == 0:
                        continue
                    x_pert = x.copy()
                    x_pert[idx] += _step_for(idx)
                    g_pert = self.approximate_gradient_parallel(x_pert, adaptive=adaptive)

                    # Diagonal H_ii
                    denom = _step_for(idx)
                    diagonal[idx] = (g_pert[idx] - base_grad[idx]) / denom
                    # Super-diagonal H_{i,i+1}
                    valid = idx[idx < n - 1]
                    if valid.size:
                        denom_sd = _step_for(valid)
                        super_diagonal[valid] = (g_pert[valid + 1] - base_grad[valid + 1]) / denom_sd

            case 'backward':
                for idx in color_classes:
                    if idx.size == 0:
                        continue
                    x_pert = x.copy()
                    x_pert[idx] -= _step_for(idx)
                    g_pert = self.approximate_gradient_parallel(x_pert, adaptive=adaptive)

                    denom = _step_for(idx)
                    diagonal[idx] = (base_grad[idx] - g_pert[idx]) / denom
                    valid = idx[idx < n - 1]
                    if valid.size:
                        denom_sd = _step_for(valid)
                        super_diagonal[valid] = (base_grad[valid + 1] - g_pert[valid + 1]) / denom_sd

            case 'central':
                for idx in color_classes:
                    if idx.size == 0:
                        continue
                    x_plus = x.copy(); x_plus[idx] += _step_for(idx)
                    x_minus = x.copy(); x_minus[idx] -= _step_for(idx)

                    g_plus = self.approximate_gradient_parallel(x_plus, adaptive=adaptive)
                    g_minus = self.approximate_gradient_parallel(x_minus, adaptive=adaptive)

                    denom = _step_for(idx)
                    diagonal[idx] = (g_plus[idx] - g_minus[idx]) / (2 * denom)
                    valid = idx[idx < n - 1]
                    if valid.size:
                        denom_sd = _step_for(valid)
                        super_diagonal[valid] = (g_plus[valid + 1] - g_minus[valid + 1]) / (2 * denom_sd)

            case _:
                raise ValueError(f"Unknown finite-difference method: {self.method}. Use 'forward', 'backward', or 'central'.")

        # Construct the sparse Hessian matrix
        hessian = diags([super_diagonal, diagonal, super_diagonal], offsets=[-1, 0, 1], shape=(n, n), format="csr")

        return hessian.tocsr()
    
class ExactDerivatives(object):
    """This class computes exact derivatives of a given function in a given point"""
    def __init__(self):
        pass
    
    def extended_rosenbrock(self,x, hessian= True): 
        """
        Compute the exact value, gradient and hessian of the extended rosenbrock function at point x
        Input:
        x : point to evalutate the gradient
        Returns:
        grad : value of the gradient in x
        """
        
        n = len(x)
        assert n % 2 == 0, "Dimension must be even."
        
        f = np.zeros(n)  
        J = lil_matrix((n,n))  # Sparse Jacobian
        H = lil_matrix((n,n))  # Jacobian matrix
        
        for k in range(n):
            if k % 2 == 0:  # Odd indices (k+1 in 1-based indexing)
                f[k] = 10 * (x[k]**2 - x[k+1])
                J[k, k] = 20 * x[k]  # df_k/dx_k
                J[k, k+1] = -10  # df_k/dx_k+1
            else: 
                f[k] = x[k-1] - 1
                J[k, k-1] = 1  # df_k/dx_k-1

        if hessian == True:
            # Compute Hessian as J^T J + sum(f_k * Hessian of f_k)
            H += J.T @ J
            for k in range(n):
                #non-linearity corrections
                if k % 2 == 0:
                    H[k, k] += 20 * f[k]
        
        J = J.tocsr()
        grad = J.T @ f

        if hessian == False:
            return grad
        
        return grad, csr_matrix(H)
    

    def discrete_boundary_value_problem(self,x,hessian=True):
        """
        Compute the exact value, gradient, and Hessian of the discrete boundary value problem function at point x.

        F(x) = sum over i=1..n of [2 x_i - x_{i-1} - x_{i+1} + h^2 ( x_i + i*h + 1 )^(3/2)]^2

        Boundary conditions: x_0 = x_{n+1} = 0
        step size h = 1 / (n+1)
        """

        n = len(x)
        assert n >= 2, "Dimension must be at least 2."

        h = 1.0 / (n + 1)

        f = np.zeros(n)
        J = lil_matrix((n, n))
        H = lil_matrix((n, n))

        for i in range(n):
            # term non lineare
            nonlinear_base = x[i] + (i + 1)*h + 1.0
            f[i] = 2.0 * x[i]

            # x_{i-1} esiste se i>0, altrimenti e` 0
            if i > 0:
                f[i] -= x[i-1]
            else:
                # i=0 => x_{-1}=0
                pass

            if i < n - 1:
                f[i] -= x[i+1]
            else:
                # i=n-1 => x_{n}=0 in definizione
                pass

            # Aggiunta parte +h^2 (x_i + (i+1)h + 1)^(3/2)
            f[i] += h**2 * (nonlinear_base**(3.0/2.0))

        for i in range(n):
            nonlinear_base = x[i] + (i + 1)*h + 1.0
            d_nonlinear = (3.0/2.0)* (h**2) * (nonlinear_base**(1.0/2.0))
            J[i,i] = 2.0 + d_nonlinear

            if i > 0:
                # df_i/dx_{i-1} = -1
                J[i, i-1] = -1.0
            if i < n - 1:
                # df_i/dx_{i+1} = -1
                J[i, i+1] = -1.0

        if hessian == True:
            H = J.T @ J
            for i in range(n):
                nonlinear_base = x[i] + (i+1)*h + 1.0
                #  d/dx_i( (x[i]+...)^(3/2) ) = (3/2)( ... )^(1/2)
                # second derivative => d^2/d x_i^2 = (3/2)(1/2)( ... )^(-1/2) = (3/4)( ... )^(-1/2)

                second_derivative = (3.0/4.0)* (h**2) * (nonlinear_base**(-1.0/2.0))
                H[i,i] += f[i] * second_derivative

        J = J.tocsr()
        grad = J.T @ f

        if hessian == False:
            return grad

        return grad, csr_matrix(H)
    
    
    def Broyden_tridiagonal_function(self,x,hessian=True):
        """
        Calcola gradiente e Hessiano della **Generalized Broyden tridiagonal** (Problem 5):

            F(x) = sum_{k=1}^n | f_k(x) |^p,    con p = 7/3
            f_k(x) = (3 - 2*x[k]) * x[k] + 1 - x[k-1] - x[k+1],

        con condizioni al contorno x_0 = x_{n+1} = 0.

        Parametri
        ----------
        x : array di forma (n,)

        Ritorna
        -------
        grad : array di forma (n,)
            Il gradiente di F in x
        H : scipy.sparse.csr_matrix
            La matrice Hessiana di F in x
        """
        p = 7.0/3.0
        n = len(x)
        assert n >= 2, "La dimensione n deve essere almeno 2."

        # Residuali f_k(x)
        f = np.zeros(n)
        for i in range(n):
            x_im1 = x[i-1] if i > 0 else 0.0
            x_ip1 = x[i+1] if i < n-1 else 0.0
            f[i] = (3.0 - 2.0*x[i]) * x[i] + 1.0 - x_im1 - x_ip1

        # Jacobiano sparso J di f wrt x
        J = lil_matrix((n, n))
        for i in range(n):
            if i > 0:
                J[i, i-1] = -1.0
            if i < n-1:
                J[i, i+1] = -1.0
            # d/ dx_i [(3-2x_i)x_i] = 3 - 4 x_i
            J[i, i] = (3.0 - 4.0 * x[i])
        J = J.tocsr()

        # Pesi per p-norma composita: 
        # phi(t) = |t|^p  =>  phi'(t) = p|t|^{p-2} t,   phi''(t) = p(p-1)|t|^{p-2}  (per t != 0)
        abs_f = np.abs(f)
        w = p * (abs_f ** (p - 2.0)) * f               # lunghezza n
        a = p * (p - 1.0) * (abs_f ** (p - 2.0))        # lunghezza n (>=0)

        # Gradiente: grad = J^T w
        grad = J.T @ w

        if hessian is False:
            return grad

        # Hessiano: H = J^T diag(a) J + sum_i phi'(f_i) * Hessian(f_i)
        # Qui Hessian(f_i) ha solo elemento (i,i) = -4.
        D = diags(a, 0, shape=(n, n), format='csr')
        H = (J.T @ D) @ J

        # Correzione diagonale con phi'(f_i) * (-4)
        # aggiorna la diagonale: H_ii += (-4) * w[i]
        H = H.tolil()
        diag = H.diagonal()
        for i in range(n):
            diag[i] += (-4.0) * w[i]
        H.setdiag(diag)
        H = H.tocsr()

        return grad, H
    
    def exact_rosenbrock(self, x,hessian=True):
        """ Compute the gradient and hessian of the Rosenbrock function. """
            
        n = len(x)
        grad = np.zeros_like(x)
        for i in range(len(x) - 1):
            grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
            grad[i+1] += 200 * (x[i+1] - x[i]**2)
    
        if hessian == True:
            H = np.zeros((n, n))
            for i in range(n - 1):
                H[i, i] += -400 * (x[i+1] - 3*x[i]**2) + 2
                H[i, i+1] += -400 * x[i]
                H[i+1, i] += -400 * x[i]
                H[i+1, i+1] += 200
        else:
            return grad
        
        return grad, csr_matrix(H)
    
    def rosenbrock_hessian_vector_product(self, x, v, grad):
        """
        Compute the product of the Rosenbrock function's Hessian with a vector v,
        without explicitly forming the Hessian matrix.

        Parameters:
        x (np.ndarray): Input point (n-dimensional vector).
        v (np.ndarray): Vector to multiply with Hessian (same shape as x).

        Returns:
        np.ndarray: The result of H @ v.
        """
        n = len(x)
        Hv = np.zeros(n)

        for i in range(n - 1):
            Hv[i] += (-400 * (x[i+1] - 3*x[i]**2) + 2) * v[i]  # Main diagonal contribution
            Hv[i] += -400 * x[i] * v[i+1]  # Super-diagonal contribution
            Hv[i+1] += -400 * x[i] * v[i]  # Sub-diagonal contribution
            Hv[i+1] += 200 * v[i+1]  # Lower diagonal contribution

        return Hv

    def dbv_hessian_vector_product(self, x, v, grad):
        """
        Compute the Hessian-vector product for the DBVP function:
        F(x) = sum_{i=1..n} [2x_i - x_{i-1} - x_{i+1} + (h²/2)(x_i + i h + 1)^3]^2.

        Boundary conditions: x_0 = x_{n+1} = 0, h = 1/(n+1).

        Args:
            x (np.ndarray): Current point (x_1, ..., x_n).
            v (np.ndarray): Vector to multiply with the Hessian.

        Returns:
            np.ndarray: Hessian-vector product H(x) @ v.
        """
        n = len(x)
        assert n >= 2, "Dimension must be at least 2."
        assert len(v) == n, "Vector v must match dimension of x."

        h = 1.0 / (n + 1)
        h_sq_over_2 = (h ** 2) / 2  # Precompute h²/2
        f = np.zeros(n)  # Residuals f_i(x)
        J = lil_matrix((n, n))  # Jacobian (sparse)
        Hv = np.zeros(n)  # Hessian-vector product

        # Precompute terms
        i_array = np.arange(1, n + 1)  # i = 1..n
        V = x + i_array * h + 1.0  # x_i + i h + 1
        V_cubed = V ** 3  # (x_i + i h + 1)^3
        V_squared = V ** 2  # (x_i + i h + 1)^2 (for derivatives)

        # Compute residuals f_i and Jacobian J
        for i in range(n):
            f[i] = 2 * x[i] - (x[i - 1] if i > 0 else 0) - (x[i + 1] if i < n - 1 else 0)
            f[i] += h_sq_over_2 * V_cubed[i]

            # Diagonal of Jacobian: 2 + (3 h²/2) (x_i + i h + 1)^2
            J[i, i] = 2 + 3 * h_sq_over_2 * V_squared[i]

            # Off-diagonals: -1 (from FD Laplacian)
            if i > 0:
                J[i, i - 1] = -1
            if i < n - 1:
                J[i, i + 1] = -1

        J = J.tocsr()  # Convert to CSR for efficient operations

        # Hessian-vector product: Hv = J^T (J v) + (sum f_i ∇²f_i) v
        Jv = J @ v
        Hv = J.T @ Jv  # First term: J^T J v

        # Second term: sum f_i ∇²f_i v (correction term)
        # ∇²f_i = 3 h² (x_i + i h + 1) = 3 h² V_i
        correction = 3 * (h ** 2) * V  # ∇²f_i = 3 h² V_i
        Hv += f * correction * v  # Add (f_i ∇²f_i) v component-wise

        return Hv
    
    def extended_rosenbrock_hessian_vector_product(self, x, v, grad):
        """
        Compute the Hessian-vector product for the extended Rosenbrock function at point x using sparse matrices.

        Input:
        x : point to evaluate the Hessian-vector product (numpy array of shape (n,))
        v : vector to multiply with the Hessian (numpy array of shape (n,))
        
        Returns:
        Hv : Hessian-vector product (numpy array of shape (n,))
        """
        n = len(x)
        assert n % 2 == 0, "Dimension must be even."
        assert len(v) == n, "Vector v must have the same dimension as x."

        # Initialize sparse Jacobian in List of Lists (LIL) format
        J = lil_matrix((n, n))

        # Compute residuals f and Jacobian J
        f = np.zeros(n)
        for k in range(n):
            if k % 2 == 0:  # Odd indices (k+1 in 1-based indexing)
                f[k] = 10 * (x[k]**2 - x[k + 1])
                J[k, k] = 20 * x[k]  # df_k/dx_k
                J[k, k + 1] = -10  # df_k/dx_{k+1}
            else:  # Even indices
                f[k] = x[k - 1] - 1
                J[k, k - 1] = 1  # df_k/dx_{k-1}

        # Convert Jacobian to Compressed Sparse Row (CSR) format for efficient operations
        J = J.tocsr()

        # Compute Hessian-vector product: Hv = J^T (J v) + correction term
        Jv = J @ v  # Matrix-vector product J * v
        Hv = J.T @ Jv  # Matrix-vector product J^T * (J * v)

        # Add the correction term for the Hessian-vector product
        for k in range(0, n, 2):  # Only odd indices (k % 2 == 0)
            Hv[k] += 20 * f[k] * v[k]  # Correction term: 20 * f_k * v_k

        return Hv
    
    def broyden_hessian_vector_product(self, x, v, grad):
        """
        Hessian-vector product per la Generalized Broyden tridiagonal (Problem 5):

            F(x) = sum_i |f_i(x)|^p,  p = 7/3
            f_i(x) = (3 - 2*x[i]) * x[i] + 1 - x[i-1] - x[i+1],
            con x_0 = x_{n+1} = 0.

        Hv = (J^T diag(phi''(f)) J) v + [\sum_i phi'(f_i) * \nabla^2 f_i] v,
        con phi'(t) = p|t|^{p-2} t,  phi''(t) = p(p-1)|t|^{p-2}.
        """
        p = 7.0/3.0
        n = len(x)
        assert n >= 2, "Dimension n must be at least 2."
        assert len(v) == n, "Vector v must have the same dimension as x."

        # Residuali f e Jacobiano J
        f = np.zeros(n)
        for i in range(n):
            x_im1 = x[i - 1] if i > 0 else 0.0
            x_ip1 = x[i + 1] if i < n - 1 else 0.0
            f[i] = (3.0 - 2.0 * x[i]) * x[i] + 1.0 - x_im1 - x_ip1

        J = lil_matrix((n, n))
        for i in range(n):
            if i > 0:
                J[i, i - 1] = -1.0
            if i < n - 1:
                J[i, i + 1] = -1.0
            J[i, i] = (3.0 - 4.0 * x[i])
        J = J.tocsr()

        abs_f = np.abs(f)
        w = p * (abs_f ** (p - 2.0)) * f                 # phi'(f)
        a = p * (p - 1.0) * (abs_f ** (p - 2.0))          # phi''(f)

        # Primo termine: J^T diag(a) J v  =>  J^T ( a ⊙ (J v) )
        Jv = J @ v
        Hv = J.T @ (a * Jv)

        # Secondo termine: [sum_i phi'(f_i) * Hessian(f_i)] v
        # Hessian(f_i) ha solo (i,i) = -4  => contributo componente-wise
        Hv += (-4.0) * (w * v)

        return Hv