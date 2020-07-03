
#include "fixed_vector.h"

float dot(const int DIM, const float* v1, const float* v2) {
    float res = 0;
    for(int i = 0; i<DIM; ++i) {
        res += v1[i]*v2[i];
    }
    return res;
}

float add(const int DIM, float* v1, const float* v2, const float scl) {
    for(int i = 0; i<DIM; ++i) {
        v1[i] += scl*v2[i];
    }
}