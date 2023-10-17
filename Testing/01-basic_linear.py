# 01-basic_linear.py
#
# A test program for a basic linear fit

import numpy as np
import PhysicalFitting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

x = np.linspace(-100, 100, 100)
x = x + np.random.normal(0, 0.5, x.shape)

test_model = lambda x, m, b: x * m + b
y = test_model(x, 3.1415, 2.7) + np.random.normal(0, 1, x.shape)

fit_model_def = {"Linear": {PhysicalFitting.common.MODEL: PhysicalFitting.models.generic.s_linear,
                            PhysicalFitting.common.PARAMETERS: {'m': "linear_m",
                                                                'b': "linear_b"}},
                 PhysicalFitting.common.FIXED_PARAMETERS: {"linear_b": 2.7},
                 PhysicalFitting.common.FIT_PARAMETERS: {"linear_m": 0}}

# fit_model_def = {"Linear": {PhysicalFitting.common.MODEL: PhysicalFitting.models.generic.s_linear,
#                             PhysicalFitting.common.PARAMETERS: {'m': "linear_m",
#                                                                 'b': "linear_b"}},
#                  PhysicalFitting.common.FIT_PARAMETERS: {"linear_m": 0,
#                                                          "linear_b": 0}}


fit_model = PhysicalFitting.model_classes.CompositeModel(fit_model_def)

error_model = PhysicalFitting.solver.error_l2norm

parameters, fit = PhysicalFitting.solver.fit_model(x, y, fit_model, error_model)

print(parameters)
print(fit)

plt.plot(x, y, 'x', label="Input")
y_model = fit_model.run(x, parameters)
plt.plot(x, y_model, label="Model")
plt.legend()
plt.show()

