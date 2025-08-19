import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from Testers import Test_settings
from Modified_Newton_Method import ModifiedNewtonMethod
from Truncated_Newton_Method import TruncatedNewtonMethod
import wandb
from colorama import Fore

def computation(comb,function, method):
    if method == 'modified':
        NM = ModifiedNewtonMethod.ModifiedNewton(**comb,function=function)
    else:
        NM = TruncatedNewtonMethod.TruncatedNewtonMethod(**comb,function=function)
    execution_times_, xk, fxk, norm_gradfx_seq, k, success, inner_iter, alphas, tol_seq= NM.Run(timing=True, print_every=0)
    
    return execution_times_, xk, fxk, norm_gradfx_seq, k, success, inner_iter, alphas, tol_seq

def make_test_name(successful_tests,num_tests,n,method,function):
    succ_rate = f"{successful_tests}-of-{num_tests}"
    name = f"{method}_{n}_{function}_{succ_rate}"
    
    return name

def make_checkpoint_name(n,method,function):
    name = f"{method}_{n}_{function}"

    return name

def save_results(data,name,method, checkpoint=False):
    data_matrix = np.array(data).T
    df = pd.DataFrame(data_matrix,
                      columns=['Experiment Type', 'Preconditioning/RateofConv', 'Derivative',
                               'Perturbation', 'Execution Time',
                               'Iterations','F(x)','Norm Gradient(x)',
                               'Experimental Order of Convergence','Converged'])

    if checkpoint:
        path_name = 'Modified_checkpoints' if method == 'modified' else 'Truncated_checkpoints'
    else:
        path_name = 'Results' + '_' + method + '_' + 'Newton Method'

    folder_path = os.path.join(os.getcwd(), path_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{name}.csv")
    df.to_csv(file_path, index=False)

def plot_results(execution_times_, norm_gradfx_seq, k, inner_iter, alphas, comb, der_method_label, tol_seq=None):
    for i in range(k):
        if method == "modified":
            wandb.log({
                "Norm Gradient": norm_gradfx_seq[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "preconditioner": comb['precond'],
                "derivative_method": der_method_label,
            },
                step = i)

            wandb.log({
                "Execution Time": execution_times_[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "preconditioner": comb['precond'],
                "derivative_method": der_method_label,
            },
                step = i)

            wandb.log({
                "Backtracking Step Size": alphas[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "preconditioner": comb['precond'],
                "derivative_method": der_method_label,
            },
                step = i)

            wandb.log({
                "Inner Iterations": inner_iter[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "preconditioner": comb['precond'],
                "derivative_method": der_method_label,
            },
                step = i)
        else:
            wandb.log({
                "Norm Gradient": norm_gradfx_seq[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "rate_of_convergence": comb['rate_of_convergence'],
                "derivative_method": der_method_label,
            },
                step = i)

            wandb.log({
                "Execution Time": execution_times_[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "rate_of_convergence": comb['rate_of_convergence'],
                "derivative_method": der_method_label,
            },
                step = i)

            wandb.log({
                "Backtracking Step Size": alphas[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "rate_of_convergence": comb['rate_of_convergence'],
                "derivative_method": der_method_label,
            },
                step = i)

            wandb.log({
                "Inner Iterations": inner_iter[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "rate_of_convergence": comb['rate_of_convergence'],
                "derivative_method": der_method_label,
            },
                step = i)
            
            wandb.log({
                "Adaptive Tolerance": tol_seq[i],
                "perturbation": comb['perturbation'] if comb['derivatives']=="finite_differences" else "exact",
                "rate_of_convergence": comb['rate_of_convergence'],
                "derivative_method": der_method_label,
            },
            step = i)

def Test(n,method,function,save_every):
    testers = Test_settings()

    seed = 346205
    np.random.seed(seed)

    match method:
        case 'modified':
            params = testers.get_modified_Newton_params()
        case 'truncated':
            params = testers.get_truncated_Newton_params()
        case _:
            raise ValueError(f"Unsupported method: {method}")

    combinations = testers.expand_param_grid(params)
    tot_comb = len(combinations)
    visited_comb = 0
    
    x0 = testers.initialize_x0(function,n)
    if params['x0'] == 'sampled':
        x0 = testers.sample_x0(x0,10,seed)
        x0_size = len(x0)

    #benchmarks
    num_tests = 0
    successful_tests = 0

    execution_times = []
    num_iterations = []
    solutions = []                      #Not included in csv for visibility, eventually for plot
    values = []
    final_norms = []
    EOC_seq = []
    succes_seq = []
    norm_grad_seq = []                  #For plots, not in csv

    x0_type_seq = []
    precond_seq = []
    rate_of_conv_seq = []
    tol_seq = []
    derivatives_method = []
    perturbation = []

    def run_test(comb, x0_value):
        fixed = True if comb['x0'] == 'fixed' else False
        x0_type_seq.append(comb['x0'])

        # Decide derivative_method label
        if comb['derivatives'] == "finite_differences":
            der_method_label = comb['derivative_method']
        elif comb['derivatives'] == "adaptive_finite_differences":
            der_method_label = "adaptive_" + comb['derivative_method']
        else:
            der_method_label = "exact"

        precond_seq.append(comb['precond']) if method == 'modified' else rate_of_conv_seq.append(comb['rate_of_convergence'])
        derivatives_method.append(der_method_label)
        perturbation.append(comb['perturbation'])


        # Group runs by function + preconditioner + derivative method
        run_group = f"{function}_Precond:{comb['precond']}_Diff:{der_method_label}_n:{n:.0e}" if method == "modified" else f"{function}_Rate:{comb['rate_of_convergence']}_Diff:{der_method_label}_n:{n:.0e}"

        # Run name distinguishes perturbation
        run_name = f"perturb: {comb['perturbation']}" if comb['derivatives']=="finite_differences" or comb['derivatives'] == "adaptive_finite_differences" else "exact"

        wandb.init(
            project=f"{method}_newton",
            group=run_group,
            name=run_name,
            config={"function": function, "method": method, "n": n}
        )

        comb['x0'] = x0_value
        execution_times_, xk, fxk, norm_gradfx_seq, k, success,\
              inner_iter, alphas, tol_sequence = computation(comb,function, method)
        
        tol_seq.append(tol_sequence) if method == 'truncated' else tol_seq.append(np.nan)

        print(f"Iterations: {k}, Success: {success}, Execution Time: {np.sum(execution_times_):.4f} seconds")
        
        if fixed:
            plot_results(np.cumsum(execution_times_), norm_gradfx_seq, k, inner_iter, alphas, comb, der_method_label, tol_seq)

        wandb.finish()
        
        execution_times.append(np.sum(execution_times_))
        num_iterations.append(k)
        solutions.append(xk)
        values.append(float(np.asarray(fxk).squeeze()))
        final_norms.append(norm_gradfx_seq[-1]) if len(norm_gradfx_seq) > 0 else np.nan

        nonlocal num_tests,successful_tests
        num_tests += 1
        if success:
            succes_seq.append(True)
            successful_tests += 1
        else:
            succes_seq.append(False)

        EOC = testers.experimental_convergence_rate(norm_gradfx_seq,tail=3) if success else np.nan
        EOC_seq.append(EOC)
        norm_grad_seq.append(norm_gradfx_seq)

    save_every = save_every
    name = make_checkpoint_name(n=n,method=method,function=function)

    if method == 'modified':
        data = [x0_type_seq,precond_seq,derivatives_method,perturbation,
            execution_times,num_iterations,values,final_norms,EOC_seq,succes_seq]
    else:
        data = [x0_type_seq,rate_of_conv_seq,derivatives_method,perturbation,
            execution_times,num_iterations,values,final_norms,EOC_seq,succes_seq]
    
    for i, comb in enumerate(combinations):
        if (i+1) % save_every == 0 and i > 0:
            save_results(data, name, method,checkpoint=True)
            print(Fore.GREEN +  f"Saved checkpoint for {name} at combo {i+1}." + Fore.RESET)
        if isinstance(x0, list):
            for i,x in enumerate(x0):
                run_test(comb,x)
                print(f'Initialization {i} of x0 out of {x0_size} of combination {visited_comb} completed')
            visited_comb += 1
            print()
            print(f'...Comb {visited_comb} out of {tot_comb} completed...')
        else:
            run_test(comb,x0)
            visited_comb += 1
            print()
            print(f'...Comb {visited_comb} out of {tot_comb} completed...')

    name = make_test_name(successful_tests,num_tests,n,method,function)
    save_results(data,name,method)

if __name__ == '__main__':
    """ Possible values:
        - n = [10**3,10**4,10**5]
        - method = [modified,truncated]
        - functions = [extended_rosenbrock,extended_powell,broyden_tridiagonal_function]
    """
    n = 10**4
    save_every = 15
    method = 'modified'
    function = 'extended_powell'
    Test(n,method,function,save_every)
