#include "integrity.h"

int main(){
    
    IM im;

    // Example input 
    im.J = Eigen::MatrixXd::Zero(8,6); //8 is the measurement number, 6 is the size of estimation state.
    im.J << 1, 2, 3, 4, 5, 6,
        4, 5, 6, 7, 7, 7,
        7, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6,
        1, 2, 3, 4, 5, 6;
    im.lambda = 0.5;
    im.residual = {0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.2, 0.5};

    im.sig2pr_int = {0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2};
    im.sig2pr_acc =  {0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.3};
    im.nom_bias_int = {1, 2, 3, 2, 3, 2, 3, 2};
    im.nom_bias_acc = {1, 2, 3, 2, 3, 2, 3, 2};
    im.p_prior = {1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3};
    // im.p_prior_sys = {1e-3,1e-3};
    // im.num_system = {4,4};

    im.FE_option = true;

    im.integrityMonitor();

    return 0;
}