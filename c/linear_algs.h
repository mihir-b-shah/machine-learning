
#ifndef LINEAR_ALGS
#define LINEAR_ALGS

struct POINT {
    float* coord;
    int label;
};

struct HYPERPLANE {
    float* vector;
    float offset;
};

void perceptron(const int T, const int N, const int DIM, 
        struct POINT* points, struct HYPERPLANE* plane);

void sgd(const int DIM, const int N, struct POINT* data, 
        struct HYPERPLANE* fill);

#endif