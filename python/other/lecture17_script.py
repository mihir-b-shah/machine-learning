
import numpy as np

T = np.zeros((3,5,5))
T[0] = np.array([[0.5, 0.5, 0, 0, 0],
                 [0.25, 0.5, 0.25, 0, 0],
                 [0, 0.25, 0.5, 0.25, 0],
                 [0, 0, 0.25, 0.5, 0.25],
                 [0, 0, 0, 0.5, 0.5]])

T[1] = np.array([[0.5, 0.5, 0, 0, 0],
                 [0.333333, 0.666667, 0, 0, 0],
                 [0, 0.333333, 0.666667, 0, 0],
                 [0, 0, 0.333333, 0.666667, 0],
                 [0, 0, 0, 0.333333, 0.666667]])

T[2] = np.array([[0.666667, 0.333333, 0, 0, 0],
                 [0, 0.666667, 0.333333, 0, 0],
                 [0, 0, 0.666667, 0.333333, 0],
                 [0, 0, 0, 0.666667, 0.333333],
                 [0,0,0,0.5,0.5]])

R = np.array([0,0,0,0,1])

def get_best(s,V):
    maxval = 0
    gamma = 0.5
    for a in range(3):
        val = 0
        for spr in range(5):
            val += T[a][s][spr]*(R[spr]+gamma*V[spr])
        maxval = max(maxval, val)
    return maxval

def run(iter_count):
    V = np.zeros((5,1))
    for itr in range(iter_count):
        Vu = np.zeros((5,1))
        for s in range(5):
            Vu[s] = get_best(s,V)
        V = Vu
        #print(V)
    return V
