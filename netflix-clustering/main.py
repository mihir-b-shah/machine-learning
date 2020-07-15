import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
seeds = [0,1,2,3,4]
K = [1,2,3,4]

kbest = 1
bestbic = -100000000

for k in K:
    best = 100000000
    seed_best = 0
    for seed in seeds:
        mixtures, post = common.init(X, k, seed)
        tupl = naive_em.run(X, mixtures, None)
        if(best > tupl[2]):
            best = tupl[2]
            seed_best = seed
    mixtures, post = common.init(X, k, seed_best)
    tupl = naive_em.run(X, mixtures, None)
    bi = common.bic(X, mixtures, tupl[2])
    if(bi > bestbic):
        bestbic= bi
        kbest = k

print(kbest)
print(bestbic)
