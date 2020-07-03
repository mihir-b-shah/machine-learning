#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fixed_vector.h"
#include "linear_algs.h"

int main()
{
    FILE* fio = fopen("classifiers.in", "r");
    const int N,DIM;
    fscanf(fio, "%d %d", &N, &DIM);

    int buffer;
    struct POINT points[N];
    float data[N*DIM];

    for(int i = 0; i<N; ++i) {
        struct POINT* pt = points + i;
        for(int j = 0; j<DIM; ++j) {
            fscanf(fio, "%f", data+DIM*i+j);
        }
        pt->coord = data+DIM*i;
        fscanf(fio, "%d", &buffer);
        pt->label = buffer;
    }
    fclose(fio);

    float vect[DIM];
    memset(vect, 0, DIM*sizeof(float));
    struct HYPERPLANE plane = {vect, 0};
    sgd(DIM, N, points, &plane);

    for(int i = 0; i<DIM; ++i) {
        printf("%f ", vect[i]);
    }
    printf("%f", plane.offset);
    return 0;   
}