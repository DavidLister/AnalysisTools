# AnalysisTools
A package to make my data analysis easier and faster. Specifically curve fitting and exploring different models.

## Main Points
The main benefits of this package is:
 - Allows for the fitting of complex, composite models, with low marginal cost when modifying models
 - Model definition has only parameters, it's when building composite models that they become free or fixed
 - It's easy to change solvers and error functions, in case that helps with the model fits
 - Models are defined including their units, the code will throw errors if the units don't add up. While annoying, this is way better than unknowingly using the model wrong.
 - Models can be fit repeatably and the same model can be easily "stamped" onto other datasets, without specific tuning (which can accidentally introduce error)
 - Parameters are always passed by dictionary where possible, minimizing error due to parameter order changes.