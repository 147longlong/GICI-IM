#include "integrity.h"


IM::IM()
{
    std::cout << "The Integrity Monitoring Start! \n";
    HPL = 0.0;
    VPL = 0.0;
    IR = 0.0;
    chi2 = Eigen::MatrixXd::Constant(1,1,0);
}

IM::~IM()
{
    std::cout << "The Integrity Monitoring End! \n";
}


void IM::integrityMonitor()
{
   
    determineSubsets(p_prior, P_THRES, Fc_THRES, subsets, pap_subset, p_not_monitored, num_system, p_prior_sys);

    std::cout << "Info: Fault subsets and corresponding fault prior probability: \n";
    for (int i = 0; i < subsets.size(); ++i ) {  
        for (int j = 0; j < subsets[0].size(); ++j) {  
            std::cout << subsets[i][j] << " ";  
        }  
        std::cout << " ------ " << pap_subset[i] * 100 << "%"<< std::endl; 
    }

    computeSubsetSolution(J, lambda, residual, sig2pr_int, sig2pr_acc, nom_bias_int, nom_bias_acc,
                        subsets, 
                        sigma, bias, sigma_ss, bias_ss, s1vec, s2vec, s3vec, x, chi2);

    std::vector<int> idx = filteroutSubsets(sigma, bias, sigma_ss, bias_ss, s1vec, s2vec, s3vec, x, chi2,
                                            subsets, pap_subset, p_not_monitored);
    
    T = computeTestThresholds(sigma_ss, bias_ss, PFANE_VERT, PFANE_HOR);   
    

    //FD
    bool fault_exist = false; //True is the fault exist. 
    Eigen::MatrixXd TestStatistics = T;
    for (int i = 0; i < T.rows(); ++i)
    {
        for ( int q = 0; q < 3; ++q)
        {
            TestStatistics(i,q) = std::abs((x(i,q) - x(0,q)) / T(i,q));

            if (TestStatistics(i,q) <= 1.0)
            {
                //std::cout << "There is NO Fault occur in the fault mode i = " << i << " at direction q = " << q << std::endl;
            } 
            else 
            {
                std::cout << "There is a Fault occur in the fault mode i = " << i << " at direction q = " << q << std::endl;
                fault_exist = true;
            }
        }
    }
    if (IM::dataStorage)
    {
       
    }

    //IR computation of which x_0 is HMI((x-x_true) > AL), but no alert 1) in FD 2) in FDE 
    double IR_FD = computeIR(sigma, bias, T ,pap_subset,p_not_monitored, VAL, HAL);
    IR = IR_FD;

    //PL computation after FD
    if(!fault_exist)
    {
        std::cout << "Info: There is no fault in the system! " << std::endl;
        computePL(sigma, bias,T,pap_subset,p_not_monitored, PHMI_HOR, PHMI_VERT, PL_TOL, VPL, HPL);
        double IR = IR_FD + p_not_monitored;
        std::cout << "Info: The protection level(PL) and integrity risk(IR) is: HPL = " << HPL << ", VPL = " << VPL << ". IR = " << IR << std::endl; 
    }

    //deltect fault but only FD
    if(fault_exist&&(IM::FE_option == false))
    {
        std::cout << "Info: There are some fault in the measurements, but we don't use the FE option." << std::endl;
        std::cout << "Error: The system has Loss of Continuity(LOS)! Alert!!! The navigation service should been interrupted!!!!" << std::endl;
    }

    //use FE
    if(fault_exist&&(IM::FE_option == true))
    {   
        //initialize the bias, sigma and T in each direction. Every col of them is for a FE subset. The first col should be value in FD.  
        Eigen::MatrixXd bias_exc_sum_1 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd bias_exc_sum_2 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd bias_exc_sum_3 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd sigma_exc_sum_1 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd sigma_exc_sum_2 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd sigma_exc_sum_3 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd TestStatistics_exc_sum_1 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd TestStatistics_exc_sum_2 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd TestStatistics_exc_sum_3 = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        Eigen::MatrixXd p_fault_exc_sum = Eigen::MatrixXd::Constant(subsets.size(),subsets.size(),INFINITY);
        std::vector<int> subset_consistent;
        double IR_FE;
        
        bool consistent_exist = faultExclude(chi2, TestStatistics,  subsets, pap_subset,
                    p_prior , P_THRES, Fc_THRES,
                    J, lambda, residual, sig2pr_int, sig2pr_acc, nom_bias_int, nom_bias_acc,
                    PFDNE_VERT, PFDNE_HOR, 
                    subset_consistent,
                    bias_exc_sum_1, bias_exc_sum_2, bias_exc_sum_3,
                    sigma_exc_sum_1, sigma_exc_sum_2, sigma_exc_sum_3,
                    TestStatistics_exc_sum_1, TestStatistics_exc_sum_2, TestStatistics_exc_sum_3,
                    p_fault_exc_sum, IR_FE,
                    num_system, p_prior_sys);

        sigma_exc_sum_1.col(0) = sigma.col(0);
        sigma_exc_sum_2.col(0) = sigma.col(1);
        sigma_exc_sum_3.col(0) = sigma.col(2);
        bias_exc_sum_1.col(0) = bias.col(0);
        bias_exc_sum_2.col(0) = bias.col(1);
        bias_exc_sum_3.col(0) = bias.col(2);
        TestStatistics_exc_sum_1.col(0) = TestStatistics.col(0);
        TestStatistics_exc_sum_2.col(0) = TestStatistics.col(1);
        TestStatistics_exc_sum_3.col(0) = TestStatistics.col(2);
        for (int i = 0; i < pap_subset.size(); ++i)   p_fault_exc_sum(i,0) = pap_subset[i];
  
        if(consistent_exist)
        {
            std::cout << "Info: The FE finshed! " << std::endl;
            std::cout << "Info: The subset which don't have faults is(1 is used, 0 is fault): " ;
            for(int i = 0; i < subset_consistent.size(); ++i)  std::cout << subset_consistent[i] << " " ; 
            std::cout << std::endl;

            //PL computation after fault excluded.
            double HPL_exc = 0; double VPL_exc = 0;
            computePL_FDE(bias_exc_sum_1, bias_exc_sum_2, bias_exc_sum_3,
                    sigma_exc_sum_1, sigma_exc_sum_2, sigma_exc_sum_3,
                    TestStatistics_exc_sum_1, TestStatistics_exc_sum_2,TestStatistics_exc_sum_3,
                    p_fault_exc_sum,p_not_monitored,
                    PHMI_HOR, PHMI_VERT, PL_TOL, VPL_exc, HPL_exc);

            //IR computation after fault excluded.
            double IR_exc = IR_FD + IR_FE + p_not_monitored;
            IR = IR_exc;

            std::cout << "Info: The protection level(PL) and integrity risk(IR) after FDE is: HPL = " << HPL_exc << ", VPL = " << VPL_exc << ". IR = " << IR_exc << std::endl; 

        }
        
        //can't find a consistent. 
        if(!consistent_exist)
        {
            std::cout << "Info: Can't find a subset of measurements that is consistent! The fault exclusion failed." << std::endl;
            std::cout << "Error: The system has Loss of Continuity(LOS)! Alert!!! The navigation service should been interrupted!!!!" << std::endl;
        }


    }
     
}


void IM::determineSubsets(std::vector<double> p_prior,
                        double P_THRES,
                        double Fc_THRES,
                        std::vector<std::vector<int>>& subsets,
                        std::vector<double>& pap_subset,
                        double& p_not_monitored,
                        std::vector<int> num_system,
                        std::vector<double> p_prior_sys,
                        bool FE_flag)
{
    int N = p_prior.size();
    int N_sys  = p_prior_sys.size();
    
    if(p_prior.empty() && !FE_flag){
        std::cerr << "Error: The prior probality of ISM is empty! \n" << std::endl;
    }
    if(!p_prior_sys.empty() && !FE_flag){
        std::cout << "Info: The measurement has been divided into " << N_sys << " system! " << std::endl;
    }
    
    //Deterimine the maximum simultanous faults need to monitor.
    std::vector<double> p_sum;
    p_sum.insert(p_sum.end(),p_prior.begin(),p_prior.end());
    p_sum.insert(p_sum.end(),p_prior_sys.begin(),p_prior_sys.end());
    int N_fault_max = determineNfaultmax(p_sum,P_THRES);
    if(!FE_flag) std::cout << "Info: The maximum simultanous faults need to monitor = " << N_fault_max << std::endl;

    //Calculate the number of subsets.
    int N_used = N + N_sys;
    int subsetsize = 0;
    for(int j = 0; j <= N_fault_max;++j){
        subsetsize = subsetsize + nchoosek((N_used),j);
    }

    //Initialize the subsets_ex and pap_subset.
    std::vector<std::vector<int>> subsets_ex(subsetsize);
    for (auto& col:subsets_ex){
        col.resize(N+N_sys);
    }    
    std::fill(subsets_ex[0].begin(), subsets_ex[0].end(), 0);  //all-in-view
    pap_subset.resize(subsetsize);
    pap_subset[0] = 1;  


    //compute the probability of no fault occur
    double pnofault = 1.0; 
    for(int i = 0; i < N;++i){
        pnofault *= (1.0-p_prior[i]);
    }
    for(int i = 0; i < N_sys;++i){
        pnofault *= (1.0-p_prior_sys[i]);
    }  
    p_not_monitored = 1 - pnofault; 

    //Initialize k (number of simultaneous faults),p_not_monitored and subset index j   
    int k = 0;
    int j = 0;
    while ((k <= N_fault_max)&&(k <= N_used )&&(p_not_monitored > P_THRES)){
    
        //determine all the subsets of size k out of N_useds.
        std::vector<std::vector<int>> subsets_k_part = determine_k_subsets(N_used,k);

        //
        std::vector<std::vector<int>> subsets_k(subsets_k_part.size(), std::vector<int>(N + N_sys,1));
        std::vector<double> pap_subsets_k(subsets_k.size()); // evey row is the prior probability of fault mode k
        std::vector<double> p_diag((p_sum.size()));
        for(int i = 0; i < p_sum.size(); ++i){
            p_diag[i] = p_sum[i] / (1 - p_sum[i]);
            if(p_sum[i] == 0) p_diag[i] = 1.0;
        }
        for(int i = 0; i < subsets_k_part.size(); ++i)
        {
            double product = 1.0;
            int h_Col = 0;
            for(int jj = 0; jj < p_sum.size(); ++jj)
            {
                if(p_sum[jj] != 0 && h_Col < subsets_k_part[0].size())
                {
                    subsets_k[i][jj] = subsets_k_part[i][h_Col]; //If FE, the original subsets element position of useless measurements is 1(0 in final). And the new 1 is the new susbets.
                    ++h_Col;
                }
                if(subsets_k[i][jj]){
                    product *= p_diag[jj];
                }
            }
            product *= pnofault;
            pap_subsets_k[i] = product;
        }

        //sort subsets by decreasing probability
        std::vector<size_t> index(pap_subsets_k.size());
        std::iota(index.begin(),index.end(),0);
        std::sort(index.begin(),index.end(),[&](size_t i1, size_t i2){
            return pap_subsets_k[i1] > pap_subsets_k[i2];
        });
        std::vector<double> p_subsets_k_s(pap_subsets_k.size());
        for(int i = 0; i < pap_subsets_k.size(); ++i){
            p_subsets_k_s[i] = pap_subsets_k[index[i]];
        }
        std::vector<std::vector<int>> subsets_k_s(subsets_k.size());
        for(int i = 0; i < subsets_k.size(); ++i){
            subsets_k_s[i] = subsets_k[index[i]];
        }

        //k->all, and consider the system fault.
        int h = 0;
        while ((h < subsets_k_s.size())&&(p_not_monitored > P_THRES))
        {     
            if(p_subsets_k_s[h] > 0)
            {        
                subsets_ex[j] = subsets_k_s[h];
                pap_subset[j] = p_subsets_k_s[h]; 
                if( k !=0 ) p_not_monitored = p_not_monitored - pap_subset[j];
                ++j;
            }
            ++h;
        }
        ++k;
    }

    pap_subset.resize(j);    
    subsets_ex.resize(j);


    subsets = subsets_ex;
    //Constellation fault
    for(int h = 0; h < subsets.size(); ++h)
    {
        int j_system = 0;
        int sum_ = 0;
        while( j_system < p_prior_sys.size())
        {
            if(! p_prior_sys.size()|| !num_system.size() || num_system.size() != p_prior_sys.size()) std::cerr << "Error: You divide measurements into system, but I don't know the number of each system's measurements."<<std::endl;
            if(subsets[h][N + j_system]) //The first system occurs fault.
            {
                for(int i = 0; i < num_system[j_system];++i)
                {
                    subsets[h][sum_+i] = 1;
                }
            }
            sum_ += num_system[j_system];
            ++j_system;
        }
        if (subsets[h].size() > N) {
            subsets[h].resize(N);
        }
    }

    //System fault numbers, which will makes subset duplication.
    std::vector<int> sumsys(N_sys,0);
    std::vector<int> idsys;
    for (int jj = N; jj < subsets_ex[0].size(); ++jj) {
        for (int i = 0; i < subsets_ex.size(); ++i) {
            sumsys[jj - N] = sumsys[jj - N] + subsets_ex[i][jj];
        }
        if(sumsys[jj - N] > 0)   idsys.push_back(jj - N);

    }

    //Subset consolidation: above system fault will makes some subsets are included in the sys fault, 
    //for exmaple {1,0,0,0,  0,0,0,0,  0,0} and {1,1,1,1,  0,0,0,0,  1,0}, at this time , if former pap < 0.01 * latter pap, we can remove the former. 
    for(int jj = 0; jj < idsys.size(); ++jj)
    {
        // Find subset corresponding to constellation wide fault first
        std::vector<int> id_sys_c;
        for (int i = 0; i < subsets_ex.size(); ++i) {
            if (std::accumulate(subsets_ex[i].begin(), subsets_ex[i].begin() + N, 0) == 0 &&
                std::accumulate(subsets_ex[i].begin(), subsets_ex[i].end(), 0) == 1 &&
                subsets_ex[i][N + idsys[jj]] == 1) {
                id_sys_c.push_back(i);
            }
        }      
        // Find subsets that include a constellation fault and satellites whithin that constellation
        std::vector<int> index_cs;
        for (int i = 0; i < subsets.size(); ++i) {
            bool condition1 = false;
            bool condition2 = true;
            for (int j = 0; j < subsets[i].size(); ++j) {
                condition1 |= (subsets[i][j] * subsets[id_sys_c[0]][j]) > 0;
                condition2 &= (subsets[i][j] * (1 - subsets[id_sys_c[0]][j])) <= 0;
            }
            if (condition1 && condition2) {
                index_cs.push_back(i);
            }
        }

        std::vector<bool> idremove;
        for (int i = 0; i < index_cs.size(); ++i) {
            if (pap_subset[index_cs[i]] < Fc_THRES * pap_subset[id_sys_c[0]]) 
            { 
                idremove.push_back(true);
            }
            else
            {
                idremove.push_back(false);
            }
        }


        std::vector<int> idrmv;
        for (int i = 0; i < idremove.size(); ++i) {
            if (idremove[i]) {
                idrmv.push_back(index_cs[i]);
            }
        }

        //remove from the list and add probability to constellation wide fault
        pap_subset[id_sys_c[0]] += std::accumulate(idrmv.begin(), idrmv.end(), 0.0, [&](double sum, int i) { return sum + pap_subset[i]; });

        std::vector<int> idnew;
        for (int i = 0; i < pap_subset.size(); ++i) {
            if (std::find(idrmv.begin(), idrmv.end(), i) == idrmv.end()) {
                idnew.push_back(i); 
            }
        }

        std::vector<double> pap_subset_new(idnew.size());
        std::vector<std::vector<int>> subsets_new(idnew.size());
        for (int i = 0; i < idnew.size(); ++i) {
            pap_subset_new[i] = pap_subset[idnew[i]];
            subsets_new[i] = subsets[idnew[i]];
        }
        pap_subset = pap_subset_new;
        subsets = subsets_new;
    }



    for ( auto& col : subsets){
        for (auto& i : col){
            i = 1 - i;
        }
    }
                    
}

/*
Determine maximum simultanous faults need to monitor.
p: the probability of event including system
P_THRES: the probability to protect high probability to monitor
*/
int IM::determineNfaultmax(std::vector<double> p, double P_THRES)
{
    size_t n_p = p.size();
    size_t r = 0;
    double p_not_monitored = 1.0; 
    double pnofault_ = 1.0;
    std::vector<double> p_divisor;
    for(size_t i = 0; i < n_p;++i){
            p_divisor.push_back(p[i]/(1.0-p[i]));
            pnofault_ *= (1.0-p[i]);
    }

    while ((p_not_monitored > P_THRES)&&(r<=n_p))
    {
        r = r + 1;
 
        if(r <= 0){
            p_not_monitored = 1;
        }
        if(r == 1){
            p_not_monitored = 1 - pnofault_;
        }
        if(r == 2){
            double pmore = 0.0;
            for(size_t i = 0; i < n_p;++i){
                 pmore += p_divisor[i];
            }
            p_not_monitored = 1 - pnofault_ - pnofault_ * pmore;
        }
        if( r == 3){
            double pmore,pmore12= 0.0;
            for(size_t i = 0; i < n_p; ++i){
                pmore += p_divisor[i]; 
                for(size_t j = 0; j < i; ++j){
                    pmore12 += p_divisor[j] * p_divisor[i];
                }
            }
            p_not_monitored = 1 - pnofault_ - pnofault_ * pmore - pnofault_ * pmore12;
        }
        if(r >= 4){
            double sum_p = 0;
            for(size_t i = 0; i < n_p;++i){
                 sum_p += p[i];
            }
            double rr = 1;
            for(int i = 1;i<(r+1);++i){
                rr *= i;
            }
            p_not_monitored = std::pow(sum_p,r)/rr;
        }
    }

    int N_fault_max_ = r - 1;
    return N_fault_max_;

}

/*
determines all the subsets of size k out of n
The output is a matrix where each line corresponds to a subset.
If subsets_k(i,j) = 0, landmark j is in subset i, otherwise subsets_k(i,j)=1;
*/
std::vector<std::vector<int>> IM::determine_k_subsets(int n, int k) 
{

  std::vector<std::vector<int>> subsets;
  
  if (k == 0) {
    std::vector<int> empty(n, 0);
    subsets.push_back(empty);
    return subsets;
  }
  
  if (k == 1) {
    for (int i = 0; i < n; i++) {
      std::vector<int> single(n, 0);
      single[i] = 1;
      subsets.push_back(single);
    }
    return subsets;
  }

  for (int i = 0; i <= n - k; i++) {
    std::vector<std::vector<int>> prev = determine_k_subsets(n - i - 1, k - 1);
    for (auto subset : prev) {
      std::vector<int> newSubset(n, 0);
      for (int j = 0; j < i; j++) newSubset[j] = 0;
      newSubset[i] = 1;
      for (int j = i + 1; j < n; j++) newSubset[j] = subset[j - i - 1];
      subsets.push_back(newSubset);
    }
  }
  return subsets;
}

//A simple combination of C_n_k
int IM::nchoosek(int n, int k)
{   
    if(k > n - k){
        k = n - k;
    }
    int result = 1;
    for(int i = 0; i < k; ++i){
        result *=(n - i);
        result /=(i + 1);
    }
    return result;
}


void IM::computeSubsetSolution(Eigen::MatrixXd J,
                               double lambda,
                               std::vector<double> residual_,
                               std::vector<double> sig2pr_int_,
                               std::vector<double> sig2pr_acc_,
                               std::vector<double> nom_bias_int_,
                               std::vector<double> nom_bias_acc_,
                               std::vector<std::vector<int>> subsets_,
                               Eigen::MatrixXd& sigma,
                               Eigen::MatrixXd& bias,
                               Eigen::MatrixXd& sigma_ss,
                               Eigen::MatrixXd& bias_ss,
                               Eigen::MatrixXd& s1vec,
                               Eigen::MatrixXd& s2vec,
                               Eigen::MatrixXd& s3vec,
                               Eigen::MatrixXd& x,
                               Eigen::VectorXd& chi2

                            )
{

  
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> residual(residual_.data(),residual_.size());
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> sig2pr_int(sig2pr_int_.data(),sig2pr_int_.size());
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> sig2pr_acc(sig2pr_acc_.data(),sig2pr_acc_.size());
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> nom_bias_int(nom_bias_int_.data(),nom_bias_int_.size());
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> nom_bias_acc(nom_bias_acc_.data(),nom_bias_acc_.size());

    int N_geomatrix = J.rows(); //maybe equal to measurement number.
    int N_sets = subsets_.size();

    // TODO: 20230304 - the W is useful or not?
    // TODO: 20230426 - the form of (JtWJ + lambda * I)^-1 *JtW is correct or not?
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(sig2pr_int.size(),sig2pr_int.size());
    for (int i = 0; i < W.rows(); ++i){
        W(i,i) = 1/sig2pr_int(i);
    }
    Eigen::MatrixXd JtW = J.transpose() * W;

    Eigen::MatrixXd invcov_J = JtW * J;
    Eigen::MatrixXd lambda_matrix = lambda * Eigen::MatrixXd::Identity(invcov_J.rows(),invcov_J.cols());
    Eigen::MatrixXd invcov = invcov_J + lambda_matrix ;

    //Remove the zero column
    Eigen::VectorXd abs_sum = invcov.colwise().sum().array().abs();
    
    std::vector<int> not_zero;
    for (int i = 0; i < abs_sum.size(); ++i)
    {
        if (abs_sum[i] > 0){not_zero.push_back(i); }
    }
    Eigen::MatrixXd JtW_new = Eigen::MatrixXd::Zero(not_zero.size(),JtW.cols());
    Eigen::MatrixXd invcov_new = Eigen::MatrixXd::Zero(invcov.rows(),invcov.cols());
    Eigen::MatrixXd J_new = Eigen::MatrixXd::Zero(J.rows(),not_zero.size());
    int k = 0;
    for (int i = 0; i < not_zero.size(); ++i)
    {
        JtW_new.row(k) = JtW.row(not_zero[i]);
        J_new.col(k) = J.col(not_zero[i]);
        for (int j = 0; j < not_zero.size(); ++j)
        {
            invcov_new(i,j) = invcov(not_zero[i],not_zero[j]);
        }
        ++k;
    }
    J = J_new;
    JtW = JtW_new;
    invcov =invcov_new;
    //Remove over!

    Eigen::MatrixXd cov0 = invcov.inverse(); 

    s1vec = Eigen::MatrixXd::Constant(N_sets,N_geomatrix,INFINITY);
    s2vec = Eigen::MatrixXd::Constant(N_sets,N_geomatrix,INFINITY);
    s3vec = Eigen::MatrixXd::Constant(N_sets,N_geomatrix,INFINITY);
    x = Eigen::MatrixXd::Zero(N_sets,J.cols());

    compute_S_coefficients(J,W,JtW,lambda_matrix,subsets_,residual,s1vec,s2vec,s3vec,x);

    sigma = Eigen::MatrixXd::Constant(N_sets,3,INFINITY);
    bias = Eigen::MatrixXd::Constant(N_sets,3,INFINITY);
    sigma_ss = Eigen::MatrixXd::Constant(N_sets,3,INFINITY);
    bias_ss = Eigen::MatrixXd::Constant(N_sets,3,INFINITY);
    Eigen::MatrixXd s1vec_2 = s1vec.array().square();
    Eigen::MatrixXd s2vec_2 = s2vec.array().square();   
    Eigen::MatrixXd s3vec_2 = s3vec.array().square();
    Eigen::MatrixXd s1vec_abs = s1vec.array().abs();
    Eigen::MatrixXd s2vec_abs = s2vec.array().abs();   
    Eigen::MatrixXd s3vec_abs = s3vec.array().abs();  

    sigma.col(0) = (s1vec_2 * sig2pr_int).array().sqrt();
    sigma.col(1) = (s2vec_2 * sig2pr_int).array().sqrt();
    sigma.col(2) = (s3vec_2 * sig2pr_int).array().sqrt();
    bias.col(0) = s1vec_abs * nom_bias_int;
    bias.col(1) = s2vec_abs * nom_bias_int;
    bias.col(2) = s3vec_abs * nom_bias_int;

    
    Eigen::MatrixXd delta_s1vec = s1vec - Eigen::MatrixXd::Ones(N_sets,1) * s1vec.row(0);
    Eigen::MatrixXd delta_s2vec = s2vec - Eigen::MatrixXd::Ones(N_sets,1) * s2vec.row(0);
    Eigen::MatrixXd delta_s3vec = s3vec - Eigen::MatrixXd::Ones(N_sets,1) * s3vec.row(0);
    
    Eigen::MatrixXd delta_s1vec_2 = s1vec.array().square();
    Eigen::MatrixXd delta_s2vec_2 = s2vec.array().square();
    Eigen::MatrixXd delta_s3vec_2 = s3vec.array().square();
    Eigen::MatrixXd delta_s1vec_abs = s1vec.array().abs();
    Eigen::MatrixXd delta_s2vec_abs = s2vec.array().abs();   
    Eigen::MatrixXd delta_s3vec_abs = s3vec.array().abs();  

    sigma_ss.col(0) = (delta_s1vec_2 * sig2pr_acc).array().sqrt();
    sigma_ss.col(1) = (delta_s2vec_2 * sig2pr_acc).array().sqrt();
    sigma_ss.col(2) = (delta_s3vec_2 * sig2pr_acc).array().sqrt();
    bias_ss.col(0) = delta_s1vec_abs * nom_bias_acc;
    bias_ss.col(1) = delta_s2vec_abs * nom_bias_acc;
    bias_ss.col(2) = delta_s3vec_abs * nom_bias_acc;

    if(chi2(0)!= -1) //lower the computation.
    {
        //serve for FE.
        //convert the std::vector<std::vector<int>> to matrixxd;
        Eigen::MatrixXi subset_eigen = Eigen::MatrixXi::Ones(subsets_.size(),subsets_[0].size()) * 2;
        for(int i = 0; i < subsets_.size();++i)
        {
            for(int j = 0; j < subsets_[0].size();++j)
            {
                subset_eigen(i,j) = subsets_[i][j];
            }
        }
        Eigen::MatrixXd y_Gx2 = (residual * Eigen::MatrixXd::Ones(1,subset_eigen.rows()) - J * x.transpose()).array().square(); //(num_measurements * num_subsets)
        Eigen::MatrixXd sig_matrix = sig2pr_int * Eigen::MatrixXd::Ones(1,subset_eigen.rows()); //(num_sigma * num_subsets)
        Eigen::MatrixXi W_subsets = subset_eigen.transpose();//(num_measurements * num_subsets)
        Eigen::MatrixXd chi2_matrix = Eigen::MatrixXd::Zero(y_Gx2.rows(),y_Gx2.cols());
        //! Attention the size of each matrix!
        //TODO: The x matrix maybe inf, which may make chi2 also equal inf.
        for(int i = 0; i < y_Gx2.rows(); ++i)
        {
            for(int j = 0; j < y_Gx2.cols(); ++j)
            {
                chi2_matrix(i,j) = y_Gx2(i,j) * W_subsets(i,j) / sig_matrix(i,j);
            }
        }
        chi2 = chi2_matrix.colwise().sum().transpose();//(num_subsets * 1)

    }
}

//A function is used in computeSubsetSolution.
void IM::compute_S_coefficients(
    const Eigen::MatrixXd& J,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& JtW,
    const Eigen::MatrixXd& lambda_matrix,
    const std::vector<std::vector<int>>& subsets_,
    const Eigen::VectorXd& residual,
    Eigen::MatrixXd& s1vec,
    Eigen::MatrixXd& s2vec,
    Eigen::MatrixXd& s3vec,
    Eigen::MatrixXd& x)
{

    for (int i = 0; i < subsets_.size(); ++i)
    {
        Eigen::MatrixXd W_sub = W;
        int sum_i_nouse = 0;
        for (int j = 0; j < W.cols(); ++j)
        {
            if (subsets_[i][j] == 0)
            {
                W_sub(j, j) = 0; //Thie is different from MAAST "compute_s_coefficients.m" but i think  it's also right.
                ++sum_i_nouse;
            }
        }


        Eigen::MatrixXd JtW_sub = J.transpose() * W_sub;
        Eigen::MatrixXd invcov_J_sub = JtW_sub * J;

        Eigen::MatrixXd invcov_sub = invcov_J_sub + lambda_matrix;

        Eigen::MatrixXd S_sub = Eigen::MatrixXd::Constant(J.cols(), J.rows(), INFINITY);

        // Remove the zero column
        Eigen::VectorXd abs_sum = invcov_sub.colwise().sum().array().abs();
        std::vector<int> not_zero_idx;
        int n_unk = 0;
        for (int iii = 0; iii < abs_sum.size(); ++iii)
        {
            if (abs_sum(iii) > 0)
            {
                not_zero_idx.push_back(iii);
                if (iii < 3) ++n_unk;
            }
        }
        Eigen::MatrixXd JtW_sub_new(n_unk, JtW_sub.cols());
        Eigen::MatrixXd invcov_sub_new(n_unk, n_unk);
        Eigen::MatrixXd Jtw_all_new(n_unk, JtW.cols());
        int k = 0;
        for (int i_ = 0; i_ < n_unk; ++i_)
        {
            JtW_sub_new.row(k) = JtW_sub.row(not_zero_idx[i_]);
            Jtw_all_new.row(k) = JtW.row(not_zero_idx[i_]);
            for (int j = 0; j < n_unk; ++j)
            {
                invcov_sub_new(i_, j) = invcov_sub(not_zero_idx[i_], not_zero_idx[j]);
            }
            ++k;
        }
        if (n_unk == 0)
        {
            std::cerr << "The matrix invcov is empty! ";
            JtW_sub_new = JtW_sub;
            invcov_sub_new = invcov_sub;
            Jtw_all_new = JtW;
        }
        //Remove over.

        // TODO: 20240304 - the 3 is the state number. The value to modify depend on the stating method.
        if ((((J.rows() - sum_i_nouse) >= n_unk) && (J.rows() - sum_i_nouse) >= 3))
        {
            Eigen::MatrixXd covd_sub = invcov_sub_new.inverse();
            Eigen::MatrixXd Sred = covd_sub * (JtW_sub_new);
            for (int ii = 0; ii < n_unk; ++ii)
            {
                S_sub.row(not_zero_idx[ii]) = Sred.row(not_zero_idx[ii]);
            }
        }
        else
        {
            S_sub = Eigen::MatrixXd::Constant(J.cols(), J.rows(), INFINITY);
        }
        //If the number of measurements is too small to solve the solution.The s1(2,3)vec.row and x.row is INFINITY.
        s1vec.row(i) = S_sub.row(0);
        s2vec.row(i) = S_sub.row(1);
        s3vec.row(i) = S_sub.row(2);

        x.row(i) = (S_sub * residual).transpose();

        if (i == 0)
        {
            Eigen::MatrixXd S_all_in_view = S_sub;
        }

    }
}

//An approach to minimize the Protection Levels by adjusting the position
void IM::adjustPL()
{
    //TODO:Refer to  "Adjust all-in-view position coefficients" in MAAST or 4.11 in baseline. 
}

//Filter out modes that cannot be monitored (sigma == INF), adjust integrity budget and return the index of sigma we want.
std::vector<int> IM::filteroutSubsets(Eigen::MatrixXd& sigma,
                                    Eigen::MatrixXd& bias,
                                    Eigen::MatrixXd& sigma_ss,
                                    Eigen::MatrixXd& bias_ss,
                                    Eigen::MatrixXd& s1vec,
                                    Eigen::MatrixXd& s2vec,
                                    Eigen::MatrixXd& s3vec,
                                    Eigen::MatrixXd& x,
                                    Eigen::VectorXd& chi2,
                                    std::vector<std::vector<int>>& subsets,
                                    std::vector<double>& pap_subsets,
                                    double& p_not_monitored)
{
    std::vector<int> idx;
    for(int i = 0; i < sigma.rows(); ++i)
    {
        double sigma_min = sigma.row(i).minCoeff();
        if (sigma_min < INFINITY)
        {
            idx.push_back(i);
        }
    }

    Eigen::MatrixXd sigma_new(idx.size(), sigma.cols());
    Eigen::MatrixXd bias_new(idx.size(), bias.cols());
    Eigen::MatrixXd sigma_ss_new(idx.size(), sigma_ss.cols());
    Eigen::MatrixXd bias_ss_new(idx.size(), bias_ss.cols());
    Eigen::MatrixXd s1vec_new(idx.size(), s1vec.cols());
    Eigen::MatrixXd s2vec_new(idx.size(), s2vec.cols());
    Eigen::MatrixXd s3vec_new(idx.size(), s3vec.cols());
    Eigen::MatrixXd x_new(idx.size(), x.cols());
    Eigen::VectorXd chi2_new(idx.size());
    std::vector<std::vector<int>> subsets_new(idx.size());
    std::vector<double> pap_subsets_new(idx.size());
    
    for (int i = 0; i < idx.size(); ++i) {
        sigma_new.row(i) = sigma.row(idx[i]);
        bias_new.row(i) = bias.row(idx[i]);
        sigma_ss_new.row(i) = sigma_ss.row(idx[i]);
        bias_ss_new.row(i) = bias_ss.row(idx[i]);
        s1vec_new.row(i) = s1vec.row(idx[i]);
        s2vec_new.row(i) = s2vec.row(idx[i]);
        s3vec_new.row(i) = s3vec.row(idx[i]);
        x_new.row(i) = x.row(idx[i]);
        chi2_new(i) = chi2(idx[i]);
        subsets_new[i] = subsets[idx[i]];
        pap_subsets_new[i] = pap_subsets[idx[i]];
    }
    sigma = sigma_new;
    bias = bias_new;
    sigma_ss = sigma_ss_new;
    bias_ss = bias_ss_new;
    s1vec = s1vec_new;
    s2vec = s2vec_new;
    s3vec = s3vec_new;
    x = x_new;
    chi2 = chi2_new;
    subsets = subsets_new;
    p_not_monitored = p_not_monitored + std::accumulate(pap_subsets.begin(), pap_subsets.end(), 0.0)
                    - std::accumulate(pap_subsets_new.begin(), pap_subsets_new.end(), 0.0);
    pap_subsets = pap_subsets_new;

    return idx;
    
}


Eigen::MatrixXd IM::computeTestThresholds(Eigen::MatrixXd sigma_ss,
                               Eigen::MatrixXd bias_ss,
                               double pfa_vert,
                               double pfa_hor)
{
    int N_sets = sigma_ss.rows();

    //TODO: Whether \delta_x(xi-x) obeys the Gaussian distribution remains to be examined. 
    boost::math::normal_distribution<double> normal_d(0.0, 1.0); 
 
    double Kfa_1 = -boost::math::quantile(normal_d, 0.25 * pfa_hor / (N_sets - 1));
    double Kfa_2 = -boost::math::quantile(normal_d, 0.25 * pfa_hor / (N_sets - 1));
    double Kfa_3 = -boost::math::quantile(normal_d, 0.5 * pfa_vert / (N_sets - 1));


    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(sigma_ss.rows(),sigma_ss.cols());
    //Shizhuangwang: The \delta_x(xi-x) obeys the Gaussian distribution with 0 mean, so bias =0. 
    //but in the MAAST the bias keeps.
    T.col(0).array() = Kfa_1 * sigma_ss.col(0).array() + bias_ss.col(0).array();
    T.col(1).array() = Kfa_2 * sigma_ss.col(1).array() + bias_ss.col(1).array();
    T.col(2).array() = Kfa_3 * sigma_ss.col(2).array() + bias_ss.col(2).array();
    return T;

}

void IM::computePL( Eigen::MatrixXd sigma,
                    Eigen::MatrixXd bias,
                    Eigen::MatrixXd T,
                    std::vector<double> pap_subset,
                    double p_not_monitored,
                    const double PHMI_HOR,
                    const double PHMI_VERT,
                    const double PL_TOL,
                    double &VPL,
                    double &HPL
                   )
{
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> p_fault(pap_subset.data(),pap_subset.size());
    p_fault(0) = 2; //Server for IR and PL computation, because 2Q(***) +　Q(***)  
    double phmi_vert = PHMI_VERT * (1 - (p_not_monitored / (PHMI_HOR + PHMI_VERT)));
    double phmi_hor = PHMI_HOR * (1 - (p_not_monitored / (PHMI_HOR + PHMI_VERT))) / 2 ;
    Eigen::VectorXd sigma_col_1 = sigma.col(0); 
    Eigen::VectorXd sigma_col_2 = sigma.col(1); 
    Eigen::VectorXd sigma_col_3 = sigma.col(2); 
    Eigen::VectorXd bias_col_1 = bias.col(0); 
    Eigen::VectorXd bias_col_2 = bias.col(1); 
    Eigen::VectorXd bias_col_3 = bias.col(2); 
    Eigen::VectorXd T_col_1 = T.col(0); 
    Eigen::VectorXd T_col_2 = T.col(1); 
    Eigen::VectorXd T_col_3 = T.col(2); 

    VPL = computeVPL(sigma_col_3,bias_col_3,T_col_3,p_fault,phmi_vert,PL_TOL);
    double HPL_1 = computeVPL(sigma_col_1,bias_col_1,T_col_1,p_fault,phmi_hor,PL_TOL);
    double HPL_2 = computeVPL(sigma_col_2,bias_col_2,T_col_2,p_fault,phmi_hor,PL_TOL);
    HPL = sqrt(HPL_1 + HPL_2);


}


double IM::computeVPL(Eigen::VectorXd sigma,
                    Eigen::VectorXd bias,
                    Eigen::VectorXd T,
                    Eigen::VectorXd p_fault,
                    double phmi,
                    double PL_TOL
                   )
{
    const double MAX_ITERATION = 10;
    Eigen::VectorXd alloc_max = Eigen::VectorXd::Ones(sigma.rows()); //The maximum necessary allocation of corresponding allocations normcdf((threshold_plus_bias - vpl)./sigma)

    //Exclude sigmas that are inf and evaluate their integrity contribution.
    std::vector<int> index_Inf;
    std::vector<int> index_Fin;
    double p_not_monitorable = 0;
    for (int i = 0; i < sigma.rows(); ++i)
    {   
        if ( sigma(i) == INFINITY)
        {
            index_Inf.push_back(i);
            p_not_monitorable += p_fault(i);
            
        }
        else{
            index_Fin.push_back(i);
        }
    } 

    if (p_not_monitorable >= phmi)
    {
        double VPL = INFINITY;
        return VPL;
    }

    Eigen::VectorXd sigma_new = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    Eigen::VectorXd bias_new = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    Eigen::VectorXd T_new = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    Eigen::VectorXd p_fault_new = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    Eigen::VectorXd p = Eigen::VectorXd::Ones(index_Fin.size()) * INFINITY;
    for (int i = 0, k = 0; i < index_Fin.size(); ++i)
    {
        sigma_new(k) = sigma(index_Fin[i]);
        bias_new(k) = bias(index_Fin[i]);
        T_new(k) = T(index_Fin[i]);
        p_fault_new(k) = p_fault(index_Fin[i]); 
        ++k;       
    }
    sigma = sigma_new; bias = bias_new; T = T_new; p_fault = p_fault_new;
    phmi = phmi - p_not_monitorable;

    //determine the lower bound on VPL 
    Eigen::VectorXd phmi_right_low = p_fault;
    Eigen::VectorXd Klow = p_fault;
    boost::math::normal_distribution<double> normal_d(0.0, 1.0); 
    for (int i = 0; i < Klow.rows() - 1; ++i)
    {
        phmi_right_low(i) =((phmi / (p_fault(i)  * alloc_max(i))) > 1 )?  1 : (phmi / (p_fault(i) * alloc_max(i)));
        if(phmi_right_low(i) == 1)
        {
            Klow(i) = -INFINITY;
        }
        else
        {
            Klow(i) = - boost::math::quantile(normal_d, phmi_right_low(i));
        }
    }
    Klow.array() = T.array() + bias.array() + Klow.array() * sigma.array(); 
    double VPL_low = Klow.maxCoeff();

    //determine the upper bound on VPL 
    Eigen::VectorXd phmi_right_high = p_fault;
    phmi_right_high.array() = phmi / (sigma.rows() * p_fault.array());
    Eigen::VectorXd Khigh = p_fault;
    for (int i = 0; i < Khigh.rows();++i)
    {
        Khigh(i) = - boost::math::quantile(normal_d, phmi_right_high(i));
        if(Khigh(i) < 0) Khigh(i) = 0; 
    }
    Khigh.array() = T.array() + bias.array() + Khigh.array() * sigma.array(); 
    double VPL_high = Khigh.maxCoeff();

    //compute logarithm of phmi
    double log10phmi = std::log10(phmi);
    
    int count = 0;
    Eigen::VectorXd TbVs = Eigen::VectorXd::Zero(sigma.rows());
    while (((VPL_high - VPL_low) > PL_TOL) && (count < MAX_ITERATION))
    {
        ++count;
        double VPL_half = (VPL_high + VPL_low) / 2;

        double sum = 0;
        for (int i = 0; i < TbVs.rows(); ++i)
        {
            TbVs(i) = boost::math::cdf(normal_d, (T(i) + bias(i) - VPL_half) / sigma(i));
            if(TbVs(i) > 0.5) TbVs(i) = 1;
            if(TbVs(i) > alloc_max(i)) TbVs(i) = alloc_max(i);
            sum += p_fault(i) * TbVs(i);
        }
        double cdfhalf = std::log10(sum);
        if (cdfhalf > log10phmi) VPL_low = VPL_half;
        else VPL_high = VPL_half;
    }
    double VPL = VPL_high;
    return VPL;



}   


double IM::computeIR(Eigen::MatrixXd sigma,
                    Eigen::MatrixXd bias,
                    Eigen::MatrixXd T,
                    std::vector<double> pap_subset,
                    double p_not_monitored,
                    const double VAL,
                    const double HAL)
{
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned> p_fault(pap_subset.data(),pap_subset.size());
    p_fault(0) = 2; //Server for IR and PL computation, because 2Q(***) +　Q(***)  
    T.row(0) = Eigen::MatrixXd::Zero(1,T.cols());
    Eigen::Vector3d AL(HAL, HAL/std::sqrt(2), HAL/std::sqrt(2)); 
    Eigen::Vector3d IR = Eigen::Vector3d::Zero(IR.size());
    boost::math::normal_distribution<double> normal_d(0,1.0);
    
    for(int q = 0; q < 3; ++q)
    {
        for(int i = 0; i < sigma.rows(); ++i)
        {
            if(!std::isfinite(T(i,q)) || !std::isfinite(bias(i,q)) || !std::isfinite(bias(i,q))) continue;
            IR(q) += p_fault(i) * (1 - boost::math::cdf(normal_d, ((AL(q) - T(i,q) - bias(i,q)) / sigma(i,q))));
        }
    }
    double IR_w = IR(0) + IR(1) + IR(2);

    return (IR(0) + IR(1) + IR(2));
}


//TODO: Some ways to adjust PL, exculde fault, PL after FE and accuracy calculation. 
void IM::excludeDouble()
{
    //TODO:Refer to "Exclude modes that are double counted" in MAAST or 5.3 in baseline
}


bool IM::faultExclude(Eigen::VectorXd chi2, Eigen::MatrixXd TestStatistics, std::vector<std::vector<int>> subsets, std::vector<double> pap_subset,
                        std::vector<double> p_prior, double P_THRES,double Fc_THRES,
                        Eigen::MatrixXd J, double lambda, std::vector<double> residual_,std::vector<double> sig2pr_int_,std::vector<double> sig2pr_acc_,
                        std::vector<double> nom_bias_int_, std::vector<double> nom_bias_acc_, 
                        double PFDNE_VERT, double PFDNE_HOR,
                        std::vector<int>& subset_consistent,
                        Eigen::MatrixXd& bias_exc_sum_1, Eigen::MatrixXd& bias_exc_sum_2,Eigen::MatrixXd& bias_exc_sum_3,
                        Eigen::MatrixXd& sigma_exc_sum_1, Eigen::MatrixXd& sigma_exc_sum_2, Eigen::MatrixXd& sigma_exc_sum_3,
                        Eigen::MatrixXd& TestStatistics_exc_sum_1, Eigen::MatrixXd& TestStatistics_exc_sum_2,Eigen::MatrixXd& TestStatistics_exc_sum_3,
                        Eigen::MatrixXd& p_fault_exc_sum, double& IR_FE,
                        std::vector<int> num_system,
                        std::vector<double> p_prior_sys)
{
    int num_candidate = TestStatistics.rows();
    int num_allcandidate = TestStatistics.rows();
    bool consistent_exist = false;
    IR_FE = 0;
    bias_exc_sum_1 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    bias_exc_sum_2 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    bias_exc_sum_3 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    sigma_exc_sum_1 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    sigma_exc_sum_2 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    sigma_exc_sum_3 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    TestStatistics_exc_sum_1 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    TestStatistics_exc_sum_2 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    TestStatistics_exc_sum_3 = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);
    p_fault_exc_sum = Eigen::MatrixXd::Constant(subsets.size(),num_allcandidate,INFINITY);

    int PL_col = 1;
    while(num_candidate > 1)
    {
        //Determine the candidate for exclusion
        //Because the chi-square statistic is an upper bound of the solution separation test ratio. Minimize the chi-square statistic means maximizing the normalized solution separation.
        Eigen::MatrixXd::Index candidate_row, candidate_q;
        chi2.minCoeff(&candidate_row);
        if (candidate_row == 0) TestStatistics.maxCoeff(&candidate_row,&candidate_q); //NOW! The chi2 has some question because the x is inf, so we use T!
        
        if (candidate_row == 0) 
        {
            std::cout << "Info: There is no longer a maximum or minimum value for T or chi2.\n" <<std::endl;
            break;
        }
        
        std::vector<int> all_in_view_exc = subsets[candidate_row];

        std::vector<double> p_prior_exc(p_prior);
        std::vector<double> p_prior_sys_exc(p_prior_sys);
        std::vector<int> num_system_exc(num_system);

        //Construct the new p_prior, p_prior_sys, num_system.
        int h_measurements = 0; int h_sys = 0;
        for (int i = 0; i < subsets[candidate_row].size(); ++i)
        {
            if(all_in_view_exc[i] == 0 && i < p_prior.size())
            {
                p_prior_exc[i] = 0;
                ++h_measurements;
            }
            if(all_in_view_exc[i] == 0 && i >= p_prior.size() && i < (p_prior_sys.size() + p_prior.size()))
            {
                p_prior_sys_exc[i - p_prior.size()] = 0;
                num_system_exc[i - p_prior.size()] = 0;
                ++h_sys;
            }
        }

        std::cout << "Info: The candidate susbet in fault exclusion is(1 is used): " ;
        for(int i = 0; i < all_in_view_exc.size(); ++i)  std::cout << all_in_view_exc[i] << " " ; 
        std::cout << std::endl;        


        std::vector<double> pap_subset_exc; 
        double p_not_monitored_exc;
        std::vector<std::vector<int>> subsets_exc; 
        determineSubsets(p_prior_exc, P_THRES,Fc_THRES, subsets_exc, pap_subset_exc, p_not_monitored_exc, num_system_exc, p_prior_sys_exc, true);

        // std::cout << "Info: In FE!!!!!! Fault subsets and corresponding fault prior probability: \n";
        // for (int i = 0; i < subsets_exc.size(); ++i ) 
        // {  
        //     for (int j = 0; j < subsets_exc[0].size(); ++j) {  
        //         std::cout << subsets_exc[i][j] << " ";  
        //     }  
        //     std::cout << " ------ " << pap_subset_exc[i] * 100 << "%"<< std::endl; 
        // }

        Eigen::MatrixXd sigma_exc= Eigen::MatrixXd::Constant(subsets_exc.size(),3,INFINITY);
        Eigen::MatrixXd bias_exc= Eigen::MatrixXd::Constant(subsets_exc.size(),3,INFINITY);
        Eigen::MatrixXd sigma_ss_exc= Eigen::MatrixXd::Constant(subsets_exc.size(),3,INFINITY);
        Eigen::MatrixXd bias_ss_exc= Eigen::MatrixXd::Constant(subsets_exc.size(),3,INFINITY);
        Eigen::MatrixXd s1vec_exc= Eigen::MatrixXd::Constant(subsets_exc.size(),J.rows(),INFINITY);
        Eigen::MatrixXd s2vec_exc =  Eigen::MatrixXd::Constant(subsets_exc.size(),J.rows(),INFINITY);
        Eigen::MatrixXd s3vec_exc=  Eigen::MatrixXd::Constant(subsets_exc.size(),J.rows(),INFINITY);
        Eigen::MatrixXd x_exc=  Eigen::MatrixXd::Constant(subsets_exc.size(),J.rows(),INFINITY);
        Eigen::VectorXd chi2_exc = Eigen::MatrixXd::Constant(subsets_exc.size(),1,-1);


        computeSubsetSolution(J, lambda, residual_, sig2pr_int_, sig2pr_acc_, nom_bias_int_, nom_bias_acc_, subsets_exc, 
                            sigma_exc, bias_exc, sigma_ss_exc, bias_ss_exc, s1vec_exc, s2vec_exc, s3vec_exc, x_exc, chi2_exc);
        
        double PFDNE_VERT_exc = PFDNE_VERT / (num_allcandidate * pap_subset[candidate_row]);
        double PFDNE_HOR_exc = PFDNE_HOR / (num_allcandidate * pap_subset[candidate_row]);
        Eigen::MatrixXd T_exc = computeTestThresholds(sigma_ss_exc, bias_ss_exc,PFDNE_VERT_exc,PFDNE_HOR_exc); 

        bool fault_exist_exc = false; //True is the fault exist. 
        Eigen::MatrixXd TestStatistics_exc = T_exc;
        for (int i = 0; i < T_exc.rows(); ++i)
        {
            for ( int q = 0; q < 3; ++q)
            {
                TestStatistics_exc(i,q) = ( x_exc(i,q) - x_exc(0,q) ) / T_exc(i,q);
                if(TestStatistics_exc(i,q) < 0) TestStatistics_exc = -1.0 * TestStatistics_exc;

                if (TestStatistics_exc(i,q) > 1 || !std::isfinite(TestStatistics_exc(i,q)))
                {   
                    std::cout << "In FE, there is a first Fault in the fault mode i = " << i << "   !! The candidate is NOT consistent !! " <<std::endl;
                    fault_exist_exc = true;
                    break;
                }
            }
            if(fault_exist_exc) 
            {
                std::cout << std::endl;
                break;
            }
        }

        //find a consistent subset 
        if(fault_exist_exc == false && consistent_exist == false)
        {
            subset_consistent = all_in_view_exc; 
            consistent_exist = true;
        }

        //for next iteration.  
        chi2(candidate_row) = INFINITY;
        TestStatistics.row(candidate_row) = Eigen::MatrixXd::Constant(1,sigma_ss_exc.cols(),-1).row(0);
        --num_candidate;

        //serve for PL computation.
    
        sigma_exc_sum_1.col(PL_col).topRows(sigma_exc.rows()) = sigma_exc.col(0);
        sigma_exc_sum_2.col(PL_col).topRows(sigma_exc.rows())  = sigma_exc.col(1);
        sigma_exc_sum_3.col(PL_col).topRows(sigma_exc.rows())  = sigma_exc.col(2);
        bias_exc_sum_1.col(PL_col).topRows(bias_exc.rows())  = bias_exc.col(0);
        bias_exc_sum_2.col(PL_col).topRows(bias_exc.rows()) = bias_exc.col(1);
        bias_exc_sum_3.col(PL_col).topRows(bias_exc.rows()) = bias_exc.col(2);
        TestStatistics_exc_sum_1.col(PL_col).topRows(TestStatistics_exc.rows()) = TestStatistics_exc.col(0);
        TestStatistics_exc_sum_2.col(PL_col).topRows(TestStatistics_exc.rows()) = TestStatistics_exc.col(1);
        TestStatistics_exc_sum_3.col(PL_col).topRows(TestStatistics_exc.rows()) = TestStatistics_exc.col(2);
        for (int i = 0; i < pap_subset_exc.size(); ++i)   p_fault_exc_sum(i,PL_col) = pap_subset_exc[i];

        IR_FE += computeIR(sigma_exc, bias_exc, TestStatistics_exc, pap_subset_exc, p_not_monitored_exc, VAL, HAL);

        ++PL_col;

    }

    if(consistent_exist) return true;

    //tried all subsets but failed.
    return false;
}


void IM::computePL_FDE(Eigen::MatrixXd bias_sum_1, Eigen::MatrixXd bias_sum_2,Eigen::MatrixXd bias_sum_3,
                    Eigen::MatrixXd sigma_sum_1, Eigen::MatrixXd sigma_sum_2, Eigen::MatrixXd sigma_sum_3,
                    Eigen::MatrixXd TestStatistics_sum_1, Eigen::MatrixXd TestStatistics_sum_2,Eigen::MatrixXd TestStatistics_sum_3,
                    Eigen::MatrixXd p_fault_sum,
                    double p_not_monitored,
                    const double PHMI_HOR,
                    const double PHMI_VERT,
                    const double PL_TOL,
                    double &VPL,
                    double &HPL)
{
    p_fault_sum.row(0) = Eigen::MatrixXd::Constant(1,p_fault_sum.cols(),2).row(0); //Server for IR and PL computation, because 2Q(***) +　Q(***)  
    double phmi_vert = PHMI_VERT * (1 - (p_not_monitored / (PHMI_HOR + PHMI_VERT)));
    double phmi_hor = PHMI_HOR * (1 - (p_not_monitored / (PHMI_HOR + PHMI_VERT))) / 2 ;

    Eigen::VectorXd sigma_col_1(sigma_sum_1.size());
    Eigen::VectorXd sigma_col_2(sigma_sum_1.size());
    Eigen::VectorXd sigma_col_3(sigma_sum_1.size());
    Eigen::VectorXd bias_col_1(sigma_sum_1.size());
    Eigen::VectorXd bias_col_2(sigma_sum_1.size());
    Eigen::VectorXd bias_col_3(sigma_sum_1.size());
    Eigen::VectorXd T_col_1(sigma_sum_1.size());
    Eigen::VectorXd T_col_2(sigma_sum_1.size());
    Eigen::VectorXd T_col_3(sigma_sum_1.size());
    Eigen::VectorXd p_fault(sigma_sum_1.size());

    int count = 0;
    for(int i_subset = 0; i_subset < sigma_sum_1.cols(); ++i_subset )
    {
        for(int j = 0; j < sigma_sum_1.rows(); ++j)
        {
            if(std::isinf(sigma_sum_1(j,i_subset)) && std::isinf(sigma_sum_2(j,i_subset)) && std::isinf(sigma_sum_3(j,i_subset)) && std::isinf(bias_sum_1(j,i_subset)) && std::isinf(bias_sum_2(j,i_subset)) &&
            std::isinf(bias_sum_3(j,i_subset)) && std::isinf(TestStatistics_sum_1(j,i_subset)) && std::isinf(p_fault_sum(j,i_subset)) )
            {
                break; 
            }
            sigma_col_1(count) = sigma_sum_1(j,i_subset);
            sigma_col_2(count) = sigma_sum_2(j,i_subset);
            sigma_col_3(count) = sigma_sum_3(j,i_subset);
            bias_col_1(count) =  bias_sum_1(j,i_subset);
            bias_col_2(count) =  bias_sum_2(j,i_subset);
            bias_col_3(count) =  bias_sum_3(j,i_subset);
            T_col_1(count) =  TestStatistics_sum_1(j,i_subset);
            T_col_2(count) =  TestStatistics_sum_2(j,i_subset);
            T_col_3(count) =  TestStatistics_sum_3(j,i_subset);
            p_fault(count) =  p_fault_sum(j,i_subset);
            ++count;
        }

    }
    sigma_col_1.conservativeResize(count);sigma_col_2.conservativeResize(count);sigma_col_3.conservativeResize(count);
    bias_col_1.conservativeResize(count);bias_col_2.conservativeResize(count);bias_col_3.conservativeResize(count);
    T_col_1.conservativeResize(count);T_col_2.conservativeResize(count);T_col_3.conservativeResize(count);
    p_fault.conservativeResize(count);

    VPL = computeVPL(sigma_col_3,bias_col_3,T_col_3,p_fault,phmi_vert,PL_TOL);
    double HPL_1 = computeVPL(bias_col_1,bias_col_1,T_col_1,p_fault,phmi_hor,PL_TOL);
    double HPL_2 = computeVPL(sigma_col_2,bias_col_2,T_col_2,p_fault,phmi_hor,PL_TOL);
    HPL = sqrt(HPL_1 + HPL_2);
}