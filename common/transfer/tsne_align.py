import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def align_and_embed(Fs, Ft, Xt, perplexity=30, random_state=42):
    scaler = StandardScaler()
    Z = np.vstack([Fs, Ft])
    Zs = scaler.fit_transform(Z)

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto",
                init="random", random_state=random_state)
    Z2 = tsne.fit_transform(Zs)
    Z2s, Z2t = Z2[:len(Fs)], Z2[len(Fs):]

    # Learn a map physics->target variables on target data, then apply to sources
    knn = KNeighborsRegressor(n_neighbors=min(10, len(Ft)//2))
    knn.fit(Z2t, Xt)
    Xs_hat = knn.predict(Z2s)
    return Xs_hat, Z2s, Z2t
