
#include "linear_algs.h"
#include "fixed_vector.h"

void perceptron(const int T, const int N, const int DIM, 
        struct POINT* points, struct HYPERPLANE* plane) {
    
    struct POINT* ptr;
    float* vect = plane->vector;
    float offs = plane->offset;
    
    for(int i = 0; i<T; ++i) {
        ptr = points;
        for(int j = 0; j<N; ++j) {
            if(ptr->label*(dot(DIM, ptr->coord, vect)+offs) <= 0) {
                add(DIM, vect, ptr->coord, ptr->label);
                offs += ptr->label;
            }
            ++ptr;
        }
    }
    
    plane->offset = offs;
}