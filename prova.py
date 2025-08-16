from scipy.optimize import minimize
from Tools.Functions import FunctionDefinition
from Testers import Test_settings
import numpy as np

tester = Test_settings()
params = tester.get_modified_Newton_params()
print(params)
print('*' * 50)
print(tester.expand_param_grid(params))