from scipy.optimize import minimize
from Tools.Functions import FunctionDefinition
from Testers import Test_settings
import numpy as np

dbv_function = FunctionDefinition().get_objective_function('discrete_boundary_value_problem')

# Initialization (your code)
n = 50
h = 1.0 / (n + 1)
x0 = np.arange(1, n+1) * h * (1 - np.arange(1, n+1) * h)  # Your initial guess

# Minimize with BFGS (no gradient needed)
result = minimize(dbv_function, x0, method='BFGS')
print("Optimal x:", result.x)
print("Function value at solution:", result.fun)