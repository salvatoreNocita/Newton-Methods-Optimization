import numpy as np
from Modified_Newton_Method import ModifiedNewtonMethod
from Truncated_Newton_Method import TruncatedNewtonMethod
from Testers import Test_settings

def main():
    np.random.seed(1)
    test = Test_settings()
    x0 = test.initialize_x0("extended_powell", 10**5 )
    #x0 = np.zeros(10**3)
    h = {'forward_difference': 1e-5, 'backward_difference': 1e-5, 'central_difference': 1e-6}
    NewtonBackTracking_ARG_f2= {'x0': x0,
                                'alpha0': 1,
                                'btmax': 50,
                                'rho': 0.5,
                                'c1': 1e-4,
                                'tolgrad': 1e-6,
                                'kmax': 500,
                                'eta': 0.5,
                                'function': 'extended_powell', 
                                #'solver_linear_system': 'cg',
                                #'H_correction_factor': 3,
                                #'precond': 'yes',
                                'rate_of_convergence': 'superlinear',
                                'derivatives': 'finite_differences',
                                'derivative_method': 'central',
                                'perturbation': h['central_difference']
                            }
    
    Nbk= TruncatedNewtonMethod.TruncatedNewtonMethod(**NewtonBackTracking_ARG_f2)
    _, xk2,fxk2,norm_gradfxk2_seq,k2,x_seq2, success, inner_iters, bt_seq2, tol_seq= Nbk.Run(timing = True)

    print('-'*50)
    print(f'Newton Method after {k2} iterations:')
    print(f'xk: {np.round(xk2, 4)}')
    print(f'f2_xk: {np.round(fxk2, 4)}')
    print(f'norma_gradf2_xk: {np.round(norm_gradfxk2_seq, 4)}')
    print('-'*50)

main()