
using namespace std;

#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>

class POINT {
    private:
        double x1,x2,x3,x4,x5;
        double y;
    public:
        void compress(int N, double* buffer);
};

void POINT::compress(int N, double* buffer) {
    if(N > 5) {
        // pca

    } else {
        switch(N) {
            case 5:
                x5 = buffer[4];
            case 4:
                x4 = buffer[3];
            case 3:
                x3 = buffer[2];
            case 2:
                x2 = buffer[1];
            case 1:
                x1 = buffer[0];
                break;
        }
    }
}

int main(int argc, char** argv) {
    if(argc != 2) {
        cerr << "Please enter a data file.\n";
    }

    std::ifstream fin(argv[1]);
    int buf; double buf2;
    fin >> buf;
    const int N = buf;

    fin >> buf;
    const int D = buf;

    std::vector<POINT> points(N);
    if(D > 5) {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix;
        matrix.resize(N,D);
        for(int i = 0; i<N; ++i) {
            for(int j = 0; j<D; ++j) {
                fin >> buf2;
                matrix(i,j) = buf2;
            }
        }
        Eigen::Matrix<double, 5,5> xTx;
        xTx.noalias() = matrix.transpose()*matrix;
        Eigen::EigenSolver<Eigen::MatrixXd> es(xTx);
        Eigen::MatrixXcd vects = es.eigenvectors();
        
    }
}
