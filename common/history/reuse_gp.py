import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

def fit_reuse_gp(X_hist, y_hist, X_new, y_new, alpha_hist=1.0, alpha_new=1.0):
    X = np.vstack([X_hist, X_new])
    y = np.hstack([alpha_hist * y_hist, alpha_new * y_new])
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    kernel = C(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42)
    gp.fit(Xn, y)
    return gp, scaler

def predict(gp, scaler, X):
    Xn = scaler.transform(X)
    mu, std = gp.predict(Xn, return_std=True)
    return mu, std
