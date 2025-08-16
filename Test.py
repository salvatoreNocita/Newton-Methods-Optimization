import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from Testers import Test_settings
from Modified_Newton_Method import ModifiedNewtonMethod
import time
import wandb



def computation(comb,function):
    MNM = ModifiedNewtonMethod.ModifiedNewton(**comb,function=function)
    t0 = time.perf_counter()
    xk,fxk,norm_gradfx_seq,k,x_seq,success= MNM.Run()
    execution = time.perf_counter() - t0
    
    return execution,xk,fxk,norm_gradfx_seq,k,success

def make_test_name(successful_tests,num_tests,n,method,function):
    succ_rate = f"{successful_tests}-of-{num_tests}"
    name = f"{method}_{n}_{function}_{succ_rate}"
    
    return name

def save_results(data,name,method):
    data_matrix = np.array(data).T
    df = pd.DataFrame(data_matrix,
                      columns=['Execution Time','Number Iterations','F(x)','Norm Gradient(x)',
                               'Experimental Order of Convergence','Converged'])

    path_name = 'Results' + '_' + method + '_' + 'Newton Method'
    folder_path = os.path.join(os.getcwd(), path_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{name}.csv")
    df.to_csv(file_path, index=False)

def plot_results(execution, xk, fxk, norm_gradfx_seq, k, success):
    for i in range(k):
        norm_gradfx = norm_gradfx_seq[i]
        wandb.log({
            "Iteration": i,
            "Norm Gradient": norm_gradfx
        })


def Test(n,method,function):
    testers = Test_settings()

    seed = 346205
    np.random.seed(seed)

    match method:
        case 'modified':
            params = testers.get_modified_Newton_params()
        case 'truncated':
            pass
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
    derivatives_type = []
    derivatives_method = []
    perturbation = []

    def run_test(comb, x0_value):
        x0_type_seq.append(comb['x0'])
        precond_seq.append(comb['precond']) if method == 'modified' else rate_of_conv_seq.append(comb['rate_of_convergence'])
        derivatives_type.append(comb['derivatives'])
        derivatives_method.append(comb['derivative_method'])
        perturbation.append(comb['perturbation'])

        # Decide derivative_method label
        der_method_label = comb['derivative_method'] if comb['derivatives'] == "finite" else "exact"

        # Group runs by function + preconditioner + derivative method
        run_group = f"{function}_{comb['precond']}_{der_method_label}"

        # Run name distinguishes perturbation
        run_name = f"perturb: {comb['perturbation']}" if comb['derivatives']=="finite" else "exact"

        wandb.init(
            project=f"{method}_newton",
            group=run_group,
            name=run_name,
            config={
                "method": method,
                "test_function": function,
                "preconditioner": comb['precond'],
                "derivatives": comb['derivatives'],
                "derivative_method": comb['derivative_method'],
                "perturbation": comb['perturbation']
            }
        )

        comb['x0'] = x0_value
        execution, xk, fxk, norm_gradfx_seq, k, success = computation(comb,function)

        print(f"Iterations: {k}, Success: {success}, Execution Time: {execution:.4f} seconds")
        plot_results(execution, xk, fxk, norm_gradfx_seq, k, success)

        wandb.finish()
        
        execution_times.append(execution)
        num_iterations.append(k)
        solutions.append(xk)
        values.append(float(np.asarray(fxk).squeeze()))
        final_norms.append(norm_gradfx_seq[-1]) if len(norm_gradfx_seq) > 0 else np.nan

        nonlocal num_tests, successful_tests
        num_tests += 1
        if success:
            succes_seq.append(True)
            successful_tests += 1
        else:
            succes_seq.append(False)

        EOC = testers.experimental_convergence_rate(norm_gradfx_seq,tail=3) if success else np.nan
        EOC_seq.append(EOC)
        norm_grad_seq.append(norm_gradfx_seq)

    for comb in combinations:
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
    data = [x0_type_seq,precond_seq,derivatives_type,derivatives_method,perturbation,
            execution_times,num_iterations,values,final_norms,EOC_seq,succes_seq]
    save_results(data,name,method)


if __name__ == '__main__':
    n = 10**1
    method = 'modified'
    function = 'extended_rosenbrock'
    Test(n,method,function)
