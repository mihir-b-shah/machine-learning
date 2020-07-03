import numpy as np
import em
import common

X = np.loadtxt("netflix_incomplete.txt")
#X_gold = np.loadtxt("test_complete.txt")

n, d = X.shape
K = [1,12]
seeds = [0,1,2,3,4]

for k in K:
    best_ll = -1000000000
    for seed in seeds:
        mixture, post = common.init(X, k, seed)
        ll = em.run(X, mixture, post)[2]
        best_ll = max(best_ll, ll)
        print("seed %d"%(seed))
    print("K: %d, LL: %f"%(k, best_ll))

