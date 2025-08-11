import numpy as np
from Modified_Newton_Method import ModifiedNewtonMethod

def main():
    #np.random.seed(26)
    #x0 = np.random.rand(10**2)
    x0 = np.zeros(10**3)
    h = {'forward_difference': 1e-5, 'backward_difference': 1e-5, 'central_difference': 1e-6}
    NewtonBackTracking_ARG_f2= {'x0': x0,
                                'alpha0': 1,
                                'btmax': 50,
                                'rho': 0.5,
                                'c1': 1e-4,
                                'tolgrad': 1e-8,
                                'kmax': 1000,
                                'function': 'broyden_tridiagonal_function',
                                'solver_linear_system': 'cg',
                                'H_correction_factor': 3,
                                'precond': 'yes',
                                'derivatives': 'exact',
                                'derivative_method': 'central',
                                'perturbation': h['forward_difference']
                            }
    
    Nbk= ModifiedNewtonMethod.ModifiedNewton(**NewtonBackTracking_ARG_f2)
    xk2,fxk2,norm_gradfxk2,k2,x_seq2,bt_seq2= Nbk.Run()
    print('-'*50)
    print(f'Newton Method after {k2} iterations:')
    print(f'xk: {np.round(xk2, 4)}')
    print(f'f2_xk: {np.round(fxk2, 4)}')
    print(f'norma_gradf2_xk: {np.round(norm_gradfxk2, 4)}')
    print('-'*50)

main()