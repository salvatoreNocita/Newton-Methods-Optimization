import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from scipy.sparse import diags
from scipy.sparse import csr_matrix, lil_matrix
from joblib import Parallel, delayed
from conditions import checkConditions

class approximativeDerivatives:
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
            builder = checkConditions()
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
    

class sparseApproximativeDerivatives(object):
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
        adaptive (bool): If True, build and use a per-coordinate perturbation vector.
        
        Returns:
        np.ndarray: The approximated gradient
        """
        n = len(x)

        if adaptive:
            builder = checkConditions()
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
                grad_x_plus = self.approximate_gradient_parallel(x_plus, adaptive=adaptive)
                Hv = (grad_x_plus - grad) / self.h
            case "backward":
                x_minus = x - self.h * v
                grad_x_minus = self.approximate_gradient_parallel(x_minus, adaptive=adaptive)
                Hv = (grad -  grad_x_minus) / self.h
            case "central":
                x_plus = x + self.h * v
                x_minus = x - self.h * v
                grad_x_plus = self.approximate_gradient_parallel(x_plus, adaptive = adaptive)
                grad_x_minus = self.approximate_gradient_parallel(x_minus, adaptive = adaptive)
                Hv = (grad_x_plus - grad_x_minus) / (2 * self.h)
        return Hv

    def hessian_approx_extendedros(self, x, grad, adaptive=False):
        """
        Approximate the Hessian of a scalar function f at point x using finite differences
        and a two-coloring technique, storing the result in a sparse matrix.

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
            builder = checkConditions()
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
                rows.extend(even_indices)       #extend takes each element of even_indices and append to row
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

    def hessian_approx_broyden_tridiagonal(self, x, grad, adaptive=False):
        """
        Approximate the Hessian for the Generalized Broyden tridiagonal function
        using finite differences while exploiting its *penta-diagonal* structure.

        Structure rationale:
        - F(x) = sum_i |f_i(x)|^p with f_i depending on (x_{i-1}, x_i, x_{i+1}).
        - J is tridiagonal, so J^T diag(a) J is penta-diagonal (bandwidth=2).
        - The nonlinear correction adds only to the diagonal.

        We therefore approximate only the diagonal, first, and second off-diagonals
        using a distance-2 coloring (5-coloring) to avoid cross-contamination when
        perturbing multiple coordinates at once.

        Parameters
        ----------
        x : np.ndarray
            Point where to approximate the Hessian
        grad : np.ndarray | None
            Gradient at x. If None, it is computed internally.
        adaptive : bool
            If True, use a per-coordinate step vector.

        Returns
        -------
        scipy.sparse.csr_matrix
            Approximated Hessian in CSR format with offsets [-2, -1, 0, 1, 2].
        """
        n = len(x)
        assert n >= 2, "Dimension must be at least 2."

        # Build (possibly vector) step size
        if adaptive:
            builder = checkConditions()
            h_vec = builder.build_perturbation_vector(x, self.h)
        else:
            h_vec = self.h

        # Base gradient at x (use provided grad if available)
        base_grad = grad if grad is not None else self.approximate_gradient_parallel(x, adaptive=adaptive)

        # 5-coloring to avoid distance-2 contamination (bandwidth=2)
        color_classes = [np.arange(shift, n, 5) for shift in range(5)]

        # Allocate bands
        diagonal = np.zeros(n)
        super1 = np.zeros(max(n - 1, 0))  # H_{i,i+1}
        super2 = np.zeros(max(n - 2, 0))  # H_{i,i+2}

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

                    # Diagonal H_{ii}
                    denom = _step_for(idx)
                    diagonal[idx] = (g_pert[idx] - base_grad[idx]) / denom

                    # First super-diagonal H_{i,i+1}
                    valid1 = idx[idx < n - 1]
                    if valid1.size:
                        denom1 = _step_for(valid1)
                        super1[valid1] = (g_pert[valid1 + 1] - base_grad[valid1 + 1]) / denom1

                    # Second super-diagonal H_{i,i+2}
                    valid2 = idx[idx < n - 2]
                    if valid2.size:
                        denom2 = _step_for(valid2)
                        super2[valid2] = (g_pert[valid2 + 2] - base_grad[valid2 + 2]) / denom2

            case 'backward':
                for idx in color_classes:
                    if idx.size == 0:
                        continue
                    x_pert = x.copy()
                    x_pert[idx] -= _step_for(idx)
                    g_pert = self.approximate_gradient_parallel(x_pert, adaptive=adaptive)

                    denom = _step_for(idx)
                    diagonal[idx] = (base_grad[idx] - g_pert[idx]) / denom

                    valid1 = idx[idx < n - 1]
                    if valid1.size:
                        denom1 = _step_for(valid1)
                        super1[valid1] = (base_grad[valid1 + 1] - g_pert[valid1 + 1]) / denom1

                    valid2 = idx[idx < n - 2]
                    if valid2.size:
                        denom2 = _step_for(valid2)
                        super2[valid2] = (base_grad[valid2 + 2] - g_pert[valid2 + 2]) / denom2

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

                    valid1 = idx[idx < n - 1]
                    if valid1.size:
                        denom1 = _step_for(valid1)
                        super1[valid1] = (g_plus[valid1 + 1] - g_minus[valid1 + 1]) / (2 * denom1)

                    valid2 = idx[idx < n - 2]
                    if valid2.size:
                        denom2 = _step_for(valid2)
                        super2[valid2] = (g_plus[valid2 + 2] - g_minus[valid2 + 2]) / (2 * denom2)
            case _:
                raise ValueError(f"Unknown finite-difference method: {self.method}. Use 'forward', 'backward', or 'central'.")

        # Assemble sparse penta-diagonal Hessian
        offsets = [0]
        datas = [diagonal]
        if n >= 2:
            offsets.extend([-1, 1])
            datas.extend([super1, super1])
        if n >= 3:
            offsets.extend([-2, 2])
            datas.extend([super2, super2])

        H = diags(datas, offsets, shape=(n, n), format='csr')

        return H

    def hessian_approx_extended_powell(self, x, grad, adaptive=False):
        """
        Approximate the Hessian of the **Extended Powell (Powell singular)** objective using
        finite differences while exploiting its *block-diagonal* sparsity (independent 4×4 blocks).

        The function couples variables only within 4-length blocks, so the Hessian is block-diagonal.
        We perturb disjoint index sets using a 4-coloring by index mod 4 so that within each block
        only one coordinate is perturbed at a time (no contamination).

        Parameters
        ----------
        x : np.ndarray
            Point where to approximate the Hessian (n must be a multiple of 4).
        grad : np.ndarray | None
            Gradient at x. If None, it is computed internally via finite differences.
        adaptive : bool
            If True, use a per-coordinate step vector.

        Returns
        -------
            Approximated Hessian in CSR format (block-diagonal with 4x4 blocks).
        """
        n = len(x)

        # Build (possibly vector) step size
        if adaptive:
            builder = checkConditions()
            h_vec = builder.build_perturbation_vector(x, self.h)
        else:
            h_vec = self.h

        def _step_for(idx):
            return h_vec if np.isscalar(h_vec) else h_vec[idx]

        # 4-coloring by block position: indices 0..n-1 split by (i % 4)
        color_classes = [np.arange(shift, n, 4) for shift in range(4)]

        # We'll accumulate the full (sparse) Hessian by columns using LIL for efficiency
        H = lil_matrix((n, n))

        match self.method:
            case 'forward':
                for cls in color_classes:
                    if cls.size == 0:
                        continue
                    x_pert = x.copy()
                    x_pert[cls] += _step_for(cls)
                    g_pert = self.approximate_gradient_parallel(x_pert, adaptive=adaptive)

                    # For each perturbed coordinate k, fill its column within its 4×4 block
                    for k in cls:
                        denom = _step_for(k)
                        b0 = (k // 4) * 4  # block start
                        block_js = range(b0, b0 + 4)
                        for j in block_js:
                            H[j, k] = (g_pert[j] - grad[j]) / denom

            case 'backward':
                for cls in color_classes:
                    if cls.size == 0:
                        continue
                    x_pert = x.copy()
                    x_pert[cls] -= _step_for(cls)
                    g_pert = self.approximate_gradient_parallel(x_pert, adaptive=adaptive)

                    for k in cls:
                        denom = _step_for(k)
                        b0 = (k // 4) * 4
                        block_js = range(b0, b0 + 4)
                        for j in block_js:
                            H[j, k] = (grad[j] - g_pert[j]) / denom

            case 'central':
                for cls in color_classes:
                    if cls.size == 0:
                        continue
                    x_plus = x.copy();  x_plus[cls]  += _step_for(cls)
                    x_minus = x.copy(); x_minus[cls] -= _step_for(cls)

                    g_plus = self.approximate_gradient_parallel(x_plus, adaptive=adaptive)
                    g_minus = self.approximate_gradient_parallel(x_minus, adaptive=adaptive)

                    for k in cls:
                        denom = _step_for(k)
                        b0 = (k // 4) * 4
                        block_js = range(b0, b0 + 4)
                        for j in block_js:
                            H[j, k] = (g_plus[j] - g_minus[j]) / (2.0 * denom)
            case _:
                raise ValueError(f"Unknown finite-difference method: {self.method}. Use 'forward', 'backward', or 'central'.")

        return csr_matrix(H)
    
class exactDerivatives(object):
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

    def extended_powell(self, x, hessian=True):
        """
        Compute the exact gradient and (optionally) Hessian of the **Extended Powell singular function**
        arranged in 4-variable blocks, keeping the same signature/name for compatibility.

        Block residuals for each j = 0, 1, ..., m-1 with variables
            a = x[4j], b = x[4j+1], c = x[4j+2], d = x[4j+3]:

            r1 = a + 10 b
            r2 = sqrt(5) (c - d)
            r3 = (b - 2 c)^2
            r4 = sqrt(10) (a - d)^2

        Objective: F(x) = (1/n) * sum_k r_k(x)^2  (the 1/n factor does not change
        the location of stationary points but you can drop it if your global
        objective definition differs; here we follow the residual/J/H pattern so
        gradient = J^T r and Hessian = J^T J + sum_k r_k * \nabla^2 r_k).

        Parameters
        ----------
        x : np.ndarray of shape (n,)
            n must be a multiple of 4.
        hessian : bool
            If True, also return the sparse Hessian (CSR).

        Returns
        -------
        grad : np.ndarray
            Exact gradient at x.
        H : scipy.sparse.csr_matrix, optional
            Exact Hessian at x (returned only if hessian=True).
        """

        x = np.asarray(x, dtype=float)
        n = len(x)
        assert n % 4 == 0, "Dimension n must be a multiple of 4."

        m = n  # number of residuals equals number of variables here
        r = np.zeros(m, dtype=float)
        J = lil_matrix((m, n))
        H = lil_matrix((n, n)) if hessian else None

        # Process each 4-variable block
        for j in range(n // 4):
            idx = 4 * j
            a, b, c, d = x[idx], x[idx + 1], x[idx + 2], x[idx + 3]

            # Residual indices
            k1, k2, k3, k4 = idx, idx + 1, idx + 2, idx + 3

            # r1 = a + 10 b
            r1 = a + 10.0 * b
            r[k1] = r1
            J[k1, idx] = 1.0
            J[k1, idx + 1] = 10.0
            # Hessian(r1) = 0

            # r2 = sqrt(5) (c - d)
            s5 = np.sqrt(5.0)
            r2 = s5 * (c - d)
            r[k2] = r2
            J[k2, idx + 2] = s5
            J[k2, idx + 3] = -s5
            # Hessian(r2) = 0

            # r3 = (b - 2 c)^2
            t = (b - 2.0 * c)
            r3 = t * t
            r[k3] = r3
            # grad r3
            J[k3, idx + 1] = 2.0 * t
            J[k3, idx + 2] = -4.0 * t
            # Hessian(r3) = [[0,0,0,0],[0,2,-4,0],[0,-4,8,0],[0,0,0,0]] in the block
            # (correction added later)

            # r4 = sqrt(10) (a - d)^2
            s10 = np.sqrt(10.0)
            u = (a - d)
            r4 = s10 * (u * u)
            r[k4] = r4
            # grad r4
            J[k4, idx] = s10 * 2.0 * u
            J[k4, idx + 3] = -s10 * 2.0 * u
            # Hessian(r4) = s10 * [[2,0,0,-2],[0,0,0,0],[0,0,0,0],[-2,0,0,2]] in the block
            # (correction added later)

        # Gradient
        J = J.tocsr()
        grad = J.T @ r

        if not hessian:
            return grad

        # Start with Gauss-Newton term
        H = (J.T @ J).tolil()

        # Add nonlinearity corrections: sum_k r_k * Hessian(r_k)
        for j in range(n // 4):
            idx = 4 * j
            a, b, c, d = x[idx], x[idx + 1], x[idx + 2], x[idx + 3]

            # r3 correction (Hessian of (b - 2c)^2)
            t = (b - 2.0 * c)
            r3 = t * t
            H[idx + 1, idx + 1] += r3 * 2.0
            H[idx + 1, idx + 2] += r3 * (-4.0)
            H[idx + 2, idx + 1] += r3 * (-4.0)
            H[idx + 2, idx + 2] += r3 * 8.0

            # r4 correction (Hessian of sqrt(10) * (a - d)^2 = s10 * [a - d]^2)
            s10 = np.sqrt(10.0)
            u = (a - d)
            r4 = s10 * (u * u)
            H[idx, idx]           += r4 * 2.0
            H[idx, idx + 3]       += r4 * (-2.0)
            H[idx + 3, idx]       += r4 * (-2.0)
            H[idx + 3, idx + 3]   += r4 * 2.0

        return grad, H.tocsr()
    
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

    import numpy as np

    def extended_powell_hessian_vector_product(self, x, v, grad):
        """
        Compute the Hessian-vector product H(x) @ v for the Extended Powell function.

        Parameters
        ----------
        x : np.ndarray
            1-D array of length n, with n % 4 == 0.
        v : np.ndarray
            1-D array of length n (same size as x).

        Returns
        -------
        np.ndarray
            The product H(x) @ v.
        """
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        n = x.size
        assert v.size == n
        assert n % 4 == 0

        # Split into blocks
        xb = x.reshape(-1, 4)
        vb = v.reshape(-1, 4)

        a, b, c, d = xb[:,0], xb[:,1], xb[:,2], xb[:,3]
        va, vb_, vc, vd = vb[:,0], vb[:,1], vb[:,2], vb[:,3]

        # Residual functions
        f1 = a + 10*b
        f2 = np.sqrt(5)*(c - d)
        f3 = (b - 2*c)**2
        f4 = np.sqrt(10)*(a - d)**2

        # Gradients (per block)
        Jf1 = np.stack([np.ones_like(a), 10*np.ones_like(b), np.zeros_like(c), np.zeros_like(d)], axis=1)
        Jf2 = np.stack([np.zeros_like(a), np.zeros_like(b), np.sqrt(5)*np.ones_like(c), -np.sqrt(5)*np.ones_like(d)], axis=1)
        Jf3 = np.stack([np.zeros_like(a),
                        2*(b - 2*c),
                        -4*(b - 2*c),
                        np.zeros_like(d)], axis=1)
        Jf4 = np.stack([2*np.sqrt(10)*(a - d),
                        np.zeros_like(b),
                        np.zeros_like(c),
                        -2*np.sqrt(10)*(a - d)], axis=1)

        # Jacobian-vector products (scalars per block)
        Jv1 = np.sum(Jf1 * vb, axis=1)
        Jv2 = np.sum(Jf2 * vb, axis=1)
        Jv3 = np.sum(Jf3 * vb, axis=1)
        Jv4 = np.sum(Jf4 * vb, axis=1)

        # Grad^T * (Jv): shape (n_blocks, 4)
        term1 = (
            Jf1 * Jv1[:,None] +
            Jf2 * Jv2[:,None] +
            Jf3 * Jv3[:,None] +
            Jf4 * Jv4[:,None]
        )

        # Hessians of residuals (constant small 4x4 blocks)
        Hf3 = np.zeros((len(a), 4, 4))
        Hf3[:,1,1] = 2.0
        Hf3[:,1,2] = Hf3[:,2,1] = -4.0
        Hf3[:,2,2] = 8.0

        Hf4 = np.zeros((len(a), 4, 4))
        Hf4[:,0,0] = 2*np.sqrt(10)
        Hf4[:,0,3] = Hf4[:,3,0] = -2*np.sqrt(10)
        Hf4[:,3,3] = 2*np.sqrt(10)

        # f_k * Hf_k v
        Hv3 = np.einsum("bij,bj->bi", Hf3, vb) * f3[:,None]
        Hv4 = np.einsum("bij,bj->bi", Hf4, vb) * f4[:,None]

        # Sum contributions
        Hv_block = term1 + Hv3 + Hv4

        # Flatten back
        Hv = (2.0/n) * Hv_block.reshape(-1)
        return Hv

