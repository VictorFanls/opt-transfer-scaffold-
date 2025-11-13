import os
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def scatter2d(P, title, path):
    ensure_dir(os.path.dirname(path))
    plt.figure()
    plt.scatter(P[:,0], P[:,1], s=10)
    plt.title(title); plt.xlabel("f1"); plt.ylabel("f2")
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()

