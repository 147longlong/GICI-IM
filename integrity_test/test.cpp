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

    // Example input for determineSubsets
    // subsets = {{1, 1, 1, 1, 1, 1, 1, 1}, 
    //                 {1, 1, 1, 1, 1, 1, 0, 0},
    //                 {1, 1, 1, 1, 0, 0, 1, 1},
    //                 {1, 1, 0, 0, 1, 1, 1, 1},
    //                 {0, 0, 1, 1, 1, 1, 1, 1}
    //             };
    // pap_subset = {0.21, 0.2, 0.3, 0.2, 0.3};                                         
    // p_not_monitored = 0.0000000001;

    im.sig2pr_int = {0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2};
    im.sig2pr_acc =  {0.1, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.3};
    im.nom_bias_int = {1, 2, 3, 2, 3, 2, 3, 2};
    im.nom_bias_acc = {1, 2, 3, 2, 3, 2, 3, 2};
    im.p_prior = {5e-5, 6e-5, 3e-5, 4e-5, 4e-5, 5e-5, 5e-5, 6e-5};
    im.p_prior_sys = {0.2,0.3};
    im.num_system = {4,4};
    im.FE_option = true;
    std::vector<double> pap_subset; 
    double p_not_monitored;
    std::vector<std::vector<int>> subsets_ex;    
    const double P_THRES = 9.0e-10;   
     const double Fc_THRES = 0.01;    
    im.determineSubsets(im.p_prior, P_THRES,Fc_THRES, subsets_ex, pap_subset, p_not_monitored);
    std::cout << "Info: Fault subsets and corresponding fault prior probability: \n";
    for (int i = 0; i < subsets_ex.size(); ++i ) {  
        for (int j = 0; j < subsets_ex[0].size(); ++j) {  
            std::cout << subsets_ex[i][j] << " ";  
        }  
        std::cout << " ------ " << pap_subset[i] * 100 << "%"<< std::endl; 
    }
    return 0;
}