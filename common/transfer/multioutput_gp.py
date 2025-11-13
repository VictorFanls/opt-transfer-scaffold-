import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor

def fit_multioutput_gp(X, Y):
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
    base = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X, Y)
    return model

def predict_multioutput_gp(model, X):
    return model.predict(X)
