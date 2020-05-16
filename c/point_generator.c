
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

float rand_flt() {
    return (float) rand()/RAND_MAX - 0.5f;
}

int main(int argc, char** argv) {
    FILE* fout = fopen("classifiers.in", "w");
    const int N = atoi(argv[1]);
    const int DIM = atoi(argv[2]);
    fprintf(fout, "%d %d\n", N, DIM);

    srand(time(NULL));
    float vect[DIM];
    int label = 0;

    for(int i = 0; i<DIM; ++i) {
        vect[i] = rand_flt();
    }
    float mgn = 0;
    for(int i = 0; i<DIM; ++i) {
        mgn += vect[i]*vect[i];
    }
    mgn = 1/sqrtf(mgn);
    for(int i = 0; i<DIM; ++i) {
        vect[i] *= mgn;
    }

    float offset = rand_flt();
    float points[DIM];
    float P0[DIM];
    float p0_sum = 0;
    for(int i = 0; i<DIM-1; ++i) {
        P0[i] = rand_flt();
        p0_sum += P0[i];
    }
    P0[DIM-1] = -(p0_sum+offset)/vect[DIM-1];
    for(int i = 0; i<N; ++i) {
        for(int j = 0; j<DIM; ++j) {
            points[j] = rand_flt();
        }
        float dot = 0;
        for(int j = 0; j<DIM; ++j) {
            dot += points[j]*vect[j];
        }
        label = (dot > 0) ? 1 : -1;
        for(int j = 0; j<DIM; ++j) {
            fprintf(fout, "%f ", points[j]);
        }
        fprintf(fout, "%d\n", label);
    }

    fflush(fout);
    fclose(fout);
    return 0;
}