#ifndef INTEGRITY_H
#define INTEGRITY_H
/**
 * @file integrity.h
 * @brief Provides the base functionality to monitor the integrity of Vision/IMU navigation system.
 * 
 * This file is part of the project of 618 and SJTU.
 * 
 * @author SYL
 * @version 3.4.5
 * @date 2024.5.15
 */
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <vector>
#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>

class IM
{
    public:

    /*Input 1: The State Estimation Model*/
    Eigen::MatrixXd J;                                      //Geometry matrix in ENU. (N_measurement * state)
    std::vector<double> residual;                           //vector of measurements minus the expected ranging values based on the location of the objects and the position given by the last state iteration.
    double lambda;                                          //Damping factor of LM algorithm.

    /*Input 2: Integrity Support Message (ISM) */
    std::vector<double> sig2pr_int;                         //nominal variance of the measurement error for integrity. compared with acc, it is more conservative
    std::vector<double> sig2pr_acc;                         //nominal bias of the measurement error for accuracy. 
    std::vector<double> nom_bias_int;                       //nominal bias of the measurement error for integrity. compared with acc, it is more conservative
    std::vector<double> nom_bias_acc;                       //nominal bias of the measurement error for accuracy.
    std::vector<double> p_prior;                            //priori probability of fault in measurenment i.
    std::vector<double> p_prior_sys;                        //priori probability of some system for maybe using.When we divide the measurement into systems. 
    std::vector<int> num_system;                            //to notice the number of each system's measurement. It seems to the 4th or more cols in G of GNSS. 
    bool FE_option = false;                                 //when the vaule is false, algorithm does not execute fault exclusion.
    
    /*Input 3: Other user's option*/
    bool dataStorage = true;                      



    private: //TODO:　verify the reality of NRP.

    /*Input 4: Navigation Requrirement Parameters */
    const double PHMI = 1.0e-9;//(per 30s)                  //expected integrity risk. the probabilty of position error exceeds PL but no alert, lower than the expected P_HMI.            
    const double PHMI_HOR = 0.1e-9;                         //horizontal allocation of expected IR. used for computing HPL.
    const double PHMI_VERT = 0.9e-9;                        //vertical allocation of expected IR. used for computing VPL.
    const double PFA = 1.0e-7;//(per 30s)                   //expected probability of false alarm, in chinese "期望的虚警率"。
    const double PFA_HOR = 0.25e-7;                         //horizontal allocation of expacted PFA. used for FDE in subset to determine T.
    const double PFA_VERT = 0.75e-7;                        //horizontal allocation of expacted PFA. used for FDE in subset to determine T.
    double PFANE_HOR = PFA_HOR;                             //horizontal allocation of continuity impact, which is the probability of "alert in FD, under no fault exist(False alerm), but FE failed". used for FDE in subset to determine T.
    double PFANE_VERT = PFA_VERT;                           //vertical allocation of continuity impact, which is the probability of "alert in FD, under no fault exist(False alerm), but FE failed". used for FDE in subset to determine T.
    double PFDNE_HOR = 0.25e-7;                             //horizontal allocation of continuity impact, which is the probability of "alert in FD, under fault exist, but FE failed". used for FDE in subsubset to determine T.
    double PFDNE_VERT = 0.25e-7;                            //vertical allocation of continuity impact, which is the probability of "alert in FD, under fault exist, but FE failed". used for FDE in subsubset to determine T.
    const double HAL = 15.5;//(m)                           //horizontal alert limit. used for compute IR.
    const double VAL = 5.3;//(m)                            //vertical alert limit. used for compute IR.
    const double TTA = 1.5;//(s)                            //time to alert.    
    const double P_THRES = 8.0e-10;                         //adjustable //the threshold of determining the subsets to be monitored.
    const double Fc_THRES = 0.01;                           //adjustable //the threshold used for fault consolidation.
    const double PL_TOL = 1.0e-3;//(m)                      //adjustable //the threshold of computing PL, meaning the accuracy of PL.
    
    public:

    /*Output*/
    double HPL;
    double VPL;
    double IR;
    /*Intermediate variables*/
    std::vector<double> pap_subset;                        //the fault probability of each subset. 
    double p_not_monitored;                                //the probability which we don't monitor.
    std::vector<std::vector<int>> subsets;                 //the subsets determined by determineSubsets.   
    Eigen::MatrixXd sigma;                                 //sigma^2 = S{sig2pr_int}S^T The variance of x.(N_subsets * 3)
    Eigen::MatrixXd bias;                                  //(N_subsets * 3)
    Eigen::MatrixXd sigma_ss;                              //(Si-S){sig2pr_acc}(Si-S)^T the variance of (xi-x).(N_subsets * 3)
    Eigen::MatrixXd bias_ss;                               //(N_subsets * 3)
    Eigen::MatrixXd s1vec;                                 //The row of S which corresponding the first state of x.(N_subsets * J.rows)
    Eigen::MatrixXd s2vec;                                 
    Eigen::MatrixXd s3vec;                                                          
    Eigen::MatrixXd x;                                     //The solution. (N_subsets * J.rows)
    Eigen::VectorXd chi2;                                  //the chi-square statistic overbound for FE. (N_subsets * 1)
    Eigen::MatrixXd T;                                     //the test thresholds for each subset. (N_subsets * 3)



    public:
    IM();
    ~IM();

    /**
     * @brief The main function.
     * 
     * @param[out] IR The integrity risk, which is the "miss alarm" in Fault Detection (FD), and the false exclude in Fault Exclusion (FE). Similarly to "漏警率" in Chinese.
     * @param[out] HPL Horizontal Protection Level.
     * @param[out] VPL Vertical Protection Level.
     */
    void integrityMonitor();

    /**
     * @brief Constructs the fault subsets.
     * 
     * @param[in] p_prior The fault prior probability.
     * @param[in] P_THRES The THRESHOLD parameter.
     * @param[in] Fc_THRES The THRESHOLD parameter for fault consolidation.
     * @param[out] subsets The output subsets by [num_subset * num_prior]. If subsets_ex(i,j) = 1, 
     *                     landmark j is in subset i, otherwise subsets_k(i,j)=0.
     * @param[out] pap_subset The output fault probability, corresponds to each row of subsets_ex.
     * @param[out] p_not_monitored The output probability which we don't monitor.
     * @param[in] num_system (Optional) Similar to the number of satellites in each constellation.
     * @param[in] p_prior_sys (Optional) Similar to constellation fault probability.
     */
    void determineSubsets(std::vector<double> p_prior,
                            double P_THRES ,
                            double Fc_THRES,
                            std::vector<std::vector<int>>& subsets,
                            std::vector<double>& pap_subset,
                            double& p_not_monitored,
                            std::vector<int> num_system = std::vector<int>(),
                            std::vector<double> p_prior_sys = std::vector<double>(),
                            bool FE_flag = false);
    //Determine maximum simultanous faults need to monitor.
    int determineNfaultmax(std::vector<double> p, double P_THRES);
    //Determines all the subsets of size k out of n
    std::vector<std::vector<int>> determine_k_subsets(int n, int k);
    //A simple combination of C_n_k
    int nchoosek(int n, int k);
    
    /**
     * @brief Returns the standard deviation of the position error for each subset.
     * 
     * @param[in] J The geometry matrix in ENU (N_measurement * state)
     * @param[in] sig2pr_int Nominal variance of the measurement error for integrity (N_measurement * 1)
     * @param[in] sig2pr_acc Nominal variance of the measurement error for accuracy (N_measurement * 1)
     * @param[in] nom_bias_int Nominal bias of the measurement error for integrity
     * @param[in] nom_bias_acc Nominal bias of the measurement error for accuracy
     * @param[in] subsets A num_subset*num_prior matrix corresponding to all fault modes, from the function "determineSubsets"
     * @param[out] sigma sigma^2 = S{sig2pr_int}S^T, the variance of x (N_subsets * 3)
     * @param[out] bias (N_subsets * 3)
     * @param[out] sigma_ss (Si-S){sig2pr_int}(Si-S)^T, the variance of (xi-x) (N_subsets * 3)
     * @param[out] bias_ss (N_subsets * 3)
     * @param[out] s1vec The row of S corresponding to the first state of x (N_subsets * J.rows)
     * @param[out] s2vec The row of S corresponding to the second state of x (N_subsets * J.rows)
     * @param[out] s3vec The row of S corresponding to the third state of x (N_subsets * J.rows)
     * @param[out] x The solution (N_subsets * J.cols)
     */
    void computeSubsetSolution( Eigen::MatrixXd J,
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
                                );
    //A function is used in computeSubsetSolution.
    void compute_S_coefficients(
        const Eigen::MatrixXd& J,
        const Eigen::MatrixXd& W,
        const Eigen::MatrixXd& JtW,
        const Eigen::MatrixXd& lambda_matrix,
        const std::vector<std::vector<int>>& subsets_,
        const Eigen::VectorXd& residual,
        Eigen::MatrixXd& s1vec,
        Eigen::MatrixXd& s2vec,
        Eigen::MatrixXd& s3vec,
        Eigen::MatrixXd& x);
    
    //TODO: Some ways to adjust PL.
    //An approach to minimize the Protection Levels by adjusting the position
    void adjustPL(); 

    //Filter out modes that cannot be monitored (sigma == INF) and adjust integrity budget.
    std::vector<int> filteroutSubsets(Eigen::MatrixXd& sigma,
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
                                    double& p_not_monitored);

    /**
     * @brief Compute test thresholds for different fault modes.
     * 
     * @param[in] sigma_ss (Si-S){sig2pr_int}(Si-S)^T the variance of (xi-x).(N_subsets * 3)
     * @param[in] bias_ss (N_subsets * 3)
     * @param[in] pfa_vert False alert in vertical direction.
     * @param[in] pfa_hor False alert in horizontal direction.
     * @param[out] T Test thresholds (N_subsets * 3). The column corresponds to each axis.
     */
    Eigen::MatrixXd computeTestThresholds(Eigen::MatrixXd sigma_ss,
                                Eigen::MatrixXd bias_ss,
                                double pfa_vert,
                                double pfa_hor);


    /**
     * @brief Compute the protection level.
     * 
     * @param[in] sigma sigma^2 = S{sig2pr_int}S^T The variance of x.(N_subsets * 3)
     * @param[in] bias (N_subsets * 3)
     * @param[in] T The test thresholds from "computeTestThresholds".
     * @param[in] pap_subsets The fault probability, corresponds to each row of subsets_ex.
     * @param[in] p_not_monitored The probability which we don't monitor.
     * @param[out] HPL Horizontal Protection Level.
     * @param[out] VPL Vertical Protection Level.
     */
    void computePL( Eigen::MatrixXd sigma,
                        Eigen::MatrixXd bias,
                        Eigen::MatrixXd T,
                        std::vector<double> pap_subset,
                        double p_not_monitored,
                        const double PHMI_HOR,
                        const double PHMI_VERT,
                        const double PL_TOL,
                        double &VPL,
                        double &HPL
                    );
    /**
     * @brief Compute the vertical protection level.
     * 
     * @param[in] sigma sigma^2 = S{sig2pr_int}S^T The variance of x.(N_subsets * 3)
     * @param[in] bias (N_subsets * 3)
     * @param[in] T The test thresholds from "computeTestThresholds".
     * @param[in] pap_subsets The fault probability, corresponds to each row of subsets_ex.
     * @param[in] p_not_monitored The probability which we don't monitor.
     * @param[out] VPL Vertical Protection Level, also can be one part of HPL.
     */
    double computeVPL(Eigen::VectorXd sigma,
                        Eigen::VectorXd bias,
                        Eigen::VectorXd T,
                        Eigen::VectorXd p_fault,
                        double phmi,
                        double PL_TOL
                    );


    /**
     * @brief Compute the integrity risk.
     * 
     * @param[in] sigma sigma^2 = S{sig2pr_int}S^T The variance of x.(N_subsets * 3)
     * @param[in] bias (N_subsets * 3)
     * @param[in] T The test thresholds from "computeTestThresholds".
     * @param[in] pap_subset The fault probability, corresponds to each row of subsets_ex.
     * @param[in] p_not_monitored The probability which we don't monitor.
     * @param[in] VAL The vertical alert limit.
     * @param[in] HAV The horizontal alert limit.
     * @param[out] IR The integrity risk.
     */
    double computeIR(Eigen::MatrixXd sigma,
                    Eigen::MatrixXd bias,
                    Eigen::MatrixXd T,
                    std::vector<double> pap_subset,
                    double p_not_monitored,
                    const double VAL,
                    const double HAL);

    //TODO: Some ways to adjust PL, exculde fault, PL after FE and accuracy calculation. 
    //Exclude modes that are double counted
    void excludeDouble();

    /**
     * @brief Exclude fault, find a consistent subset, compute PL and IR in FE.
     * 
     * @param[in] chi2 The chi-square statistic of each subset from "computeSubsetsolution" in FD. (N_subsets * 1)
     * @param[in] TestStatistics The test thresholds. (N_subsets * 3)
     * @param[in] subsets The subsets determined by FD. (N_subsets * N_measurements)
     * @param[in] pap_subset The fault probability, corresponds to each row of subsets. (N_subsets * 1)
     * @param[in] p_prior,P_THRES,num_system,p_prior_sys,J,lambda,residual_,sig2pr_int_,sig2pr_acc_,nom_bias_int_,nom_bias_acc_ Some variables in IM.h.
     * @param[in] PFDNE_VERT, PFDNE_HOR Allocation of FDNE, which is the probability of "alert in FD, under fault exist, but FE failed".
     * @param[out] subset_consistent The consistent subset if we find.
     * @param[out] bias(sigma,T)_exc_sum_1(2,3) Every col of them is for a FE subset. The first col should be value in FD. 
     * @param[out] p_fault_exc_sum Every col of them is for a FE subset. The first col should be value in FD. 
     * @param[out] IR_FE The integrity risk in FE.
     */
    bool faultExclude(Eigen::VectorXd chi2, Eigen::MatrixXd TestStatistics, std::vector<std::vector<int>> subsets, std::vector<double> pap_subset,
                        std::vector<double> p_prior, double P_THRES,double Fc_THRES,
                        Eigen::MatrixXd J, double lambda, std::vector<double> residual_,std::vector<double> sig2pr_int_,std::vector<double> sig2pr_acc_,
                        std::vector<double> nom_bias_int_, std::vector<double> nom_bias_acc_, 
                        double PFDNE_VERT, double PFDNE_HOR,
                        std::vector<int>& subset_consistent,
                        Eigen::MatrixXd& bias_exc_sum_1, Eigen::MatrixXd& bias_exc_sum_2,Eigen::MatrixXd& bias_exc_sum_3,
                        Eigen::MatrixXd& sigma_exc_sum_1, Eigen::MatrixXd& sigma_exc_sum_2, Eigen::MatrixXd& sigma_exc_sum_3,
                        Eigen::MatrixXd& TestStatistics_exc_sum_1, Eigen::MatrixXd& TestStatistics_exc_sum_2,Eigen::MatrixXd& TestStatistics_exc_sum_3,
                        Eigen::MatrixXd& p_fault_exc_sum, double& IR_FE,
                        std::vector<int> num_system = std::vector<int>(),
                        std::vector<double> p_prior_sys = std::vector<double>());
    /**
     * @brief Compute the Protection Level (PL) of all Fault Detection and Exclusion (FDE) process.
     * 
     * @param[in] bias (sigma,T)_exc_sum_1(2,3) Each column corresponds to a FE subset. The first column should be value in FD.
     * @param[in] p_fault_exc_sum Each column corresponds to a FE subset. The first column should be value in FD.
     * @param[in] pap_subsets The fault probability, corresponds to each row of subsets_ex.
     * @param[in] p_not_monitored The probability which we don't monitor.
     * @param[out] HPL Horizontal Protection Level.
     * @param[out] VPL Vertical Protection Level.
     */
    void computePL_FDE(Eigen::MatrixXd bias_sum_1, Eigen::MatrixXd bias_sum_2,Eigen::MatrixXd bias_sum_3,
                        Eigen::MatrixXd sigma_sum_1, Eigen::MatrixXd sigma_sum_2, Eigen::MatrixXd sigma_sum_3,
                        Eigen::MatrixXd TestStatistics_sum_1, Eigen::MatrixXd TestStatistics_sum_2,Eigen::MatrixXd TestStatistics_sum_3,
                        Eigen::MatrixXd p_fault_sum,
                        double p_not_monitored,
                        const double PHMI_HOR,
                        const double PHMI_VERT,
                        const double PL_TOL,
                        double &VPL,
                        double &HPL);

};

#endif //INTEGRITY_H