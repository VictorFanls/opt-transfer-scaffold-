import numpy as np

def cp_like_features(x, n_panels=50):
    x = np.asarray(x)
    s = np.linspace(0, 1, n_panels)
    amps = np.sin(np.arange(1, min(6, len(x)+1)) * (x.mean()+0.5))
    profile = np.zeros_like(s)
    for k, a in enumerate(amps, start=1):
        profile += a * np.cos(2*np.pi*k*s + 0.3*x[(k-1) % len(x)])
    return (profile - profile.mean()) / (profile.std() + 1e-6)

def make_source_target(n_source=200, n_target=60, ds=10, dt=6, noise=0.01):
    Xs = np.random.uniform(-1, 1, size=(n_source, ds))
    Xt = np.random.uniform(-1, 1, size=(n_target, dt))
    Fs = np.array([cp_like_features(x) for x in Xs])
    Ft = np.array([cp_like_features(x) for x in Xt])
    w  = np.linspace(-1, 1, Fs.shape[1])
    ys = Fs @ w + noise*np.random.randn(n_source)
    yt = Ft @ w + noise*np.random.randn(n_target)
    return Xs, ys, Fs, Xt, yt, Ft
