
#include "linear_algs.h"
#include "fixed_vector.h"
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h> 

#define LAMBDA 1
#define BETA 5
#define NORM_BETA 0.2
#define EPS 0.01

bool init_rand = false;

int rand_index(int N) {
    return rand()*N/RAND_MAX;
}

float learning_rate(int iter) {
    return 1.0/(1+iter);
}

void sgd(const int DIM, const int N, struct POINT* data, struct HYPERPLANE* fill) {
    if(!init_rand) {
        srand(time(NULL));
    }
    float* theta = fill->vector;
    float* offset = &(fill->offset);
    float gradient[DIM+1];

    memset(theta, 0, DIM*sizeof(float));
    *offset = 0;
    int iter = 0;
    float loss_val;
    float learn_rate;

    do {
        int index;
        struct POINT* point;
        memset(gradient, 0, (DIM+1)*sizeof(float));

        for(int i = 0; i<BETA; ++i) {
            index = rand_index(N);
            point = data + index;
            loss_val = 0;
            for(int j = 0; j<DIM; ++j) {
                loss_val += point->coord[j]*theta[j];
            }
            loss_val += *offset;
            loss_val *= point->label;
            for(int j = 0; j<DIM; ++j) {
                gradient[j] += NORM_BETA*(LAMBDA*theta[j]+(loss_val > 0 ? 0: -(point->label)*(point->coord[j])));
            }
            gradient[DIM] -= point->label;
        }
        learn_rate = learning_rate(iter);
        for(int i = 0; i<DIM; ++i) {
            theta[i] -= learn_rate*gradient[i];
        }
        *offset = learn_rate*gradient[DIM];
        ++iter;
    } while(learn_rate > EPS);
}