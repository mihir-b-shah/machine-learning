def norm(u,v,n):
    sm = 0
    for i in range(len(u)):
        sm += (abs(u[i]-v[i]))**n
    return sm**(1/n)

def k_means(points, start, norm):
    
    for point in points:
