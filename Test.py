import numpy as np
from Testers import Test_settings

def Test():
    testers = Test_settings()

    seed = 346205
    np.random.seed(seed)
    n = 10**3
    function = 'extended_rosenbrock'

    x0 = testers.initialize_x0(function,n)
    sampled_x0 = testers.sample_x0(x0,10,seed)
    params = testers.get_params()
