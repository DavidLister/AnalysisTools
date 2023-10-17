# 01-basic_linear.py
#
# A test program for a basic linear fit

import numpy as np
import matplotlib.pyplot as plt
import analysis_tools

x = np.linspace(-100, 100, 100)
x = x + np.random.normal(0, 0.5, x.shape)

test_model = lambda x, m, b: x * m + b
y = test_model(x, 3.1415, 2.7) + np.random.normal(0, 1, x.shape)

fit_model = analysis_tools.models.generic.s_linear

error_model = analysis_tools.solver.error_l2norm

fit = analysis_tools.solver.fit_model(x, y, fit_model, error_model)

