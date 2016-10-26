/**
Copyright (c) 2016 Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

If the code is used in an article, the following paper shall be cited:
@techreport{qrsvd:2016,
  title={Implicit-shifted Symmetric QR Singular Value Decomposition of 3x3 Matrices},
  author={Gast, Theodore and Fu, Chuyuan and Jiang, Chenfanfu and Teran, Joseph},
  year={2016},
  institution={University of California Los Angeles}
}

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <cmath>
#include "Tools.h"
#include "ImplicitQRSVD.h"
#include <iostream>
using namespace std;

template <class T>
void testAccuracy(const std::vector<Eigen::Matrix<T, 3, 3> >& AA,
    const std::vector<Eigen::Matrix<T, 3, 3> >& UU,
    const std::vector<Eigen::Matrix<T, 3, 1> >& SS,
    const std::vector<Eigen::Matrix<T, 3, 3> >& VV)
{
    T max_UUt_error = 0, max_VVt_error = 0, max_detU_error = 0, max_detV_error = 0, max_reconstruction_error = 0;
    T ave_UUt_error = 0, ave_VVt_error = 0, ave_detU_error = 0, ave_detV_error = 0, ave_reconstruction_error = 0;
    for (size_t i = 0; i < AA.size(); i++) {
        Eigen::Matrix<T, 3, 3> M = AA[i];
        Eigen::Matrix<T, 3, 1> S = SS[i];
        Eigen::Matrix<T, 3, 3> U = UU[i];
        Eigen::Matrix<T, 3, 3> V = VV[i];
        T error;
        error = (U * U.transpose() - Eigen::Matrix<T, 3, 3>::Identity()).array().abs().maxCoeff();
        max_UUt_error = (error > max_UUt_error) ? error : max_UUt_error;
        ave_UUt_error += fabs(error);
        error = (V * V.transpose() - Eigen::Matrix<T, 3, 3>::Identity()).array().abs().maxCoeff();
        max_VVt_error = (error > max_VVt_error) ? error : max_VVt_error;
        ave_VVt_error += fabs(error);
        error = fabs(fabs(U.determinant()) - (T)1);
        max_detU_error = (error > max_detU_error) ? error : max_detU_error;
        ave_detU_error += fabs(error);
        error = fabs(fabs(V.determinant()) - (T)1);
        max_detV_error = (error > max_detV_error) ? error : max_detV_error;
        ave_detV_error += fabs(error);
        error = (U * S.asDiagonal() * V.transpose() - M).array().abs().maxCoeff();
        max_reconstruction_error = (error > max_reconstruction_error) ? error : max_reconstruction_error;
        ave_reconstruction_error += fabs(error);
    }
    ave_UUt_error /= (T)(AA.size());
    ave_VVt_error /= (T)(AA.size());
    ave_detU_error /= (T)(AA.size());
    ave_detV_error /= (T)(AA.size());
    ave_reconstruction_error /= (T)(AA.size());
    std::cout << std::setprecision(10) << " UUt max error: " << max_UUt_error
              << " VVt max error: " << max_VVt_error
              << " detU max error:" << max_detU_error
              << " detV max error:" << max_detV_error
              << " recons max error:" << max_reconstruction_error << std::endl;
    std::cout << std::setprecision(10) << " UUt ave error: " << ave_UUt_error
              << " VVt ave error: " << ave_VVt_error
              << " detU ave error:" << ave_detU_error
              << " detV ave error:" << ave_detV_error
              << " recons ave error:" << ave_reconstruction_error << std::endl;
}

template <class T>
void runImplicitQRSVD(const int repeat, const std::vector<Eigen::Matrix<T, 3, 3> >& tests, const bool accuracy_test)
{
    using namespace JIXIE;
    std::vector<Eigen::Matrix<T, 3, 3> > UU, VV;
    std::vector<Eigen::Matrix<T, 3, 1> > SS;
    JIXIE::Timer timer;
    timer.start();
    double total_time = 0;
    for (int test_iter = 0; test_iter < repeat; test_iter++) {
        timer.click();
        for (size_t i = 0; i < tests.size(); i++) {
            Eigen::Matrix<T, 3, 3> M = tests[i];
            Eigen::Matrix<T, 3, 1> S;
            Eigen::Matrix<T, 3, 3> U;
            Eigen::Matrix<T, 3, 3> V;
            singularValueDecomposition(M, U, S, V);
            if (accuracy_test && test_iter == 0) {
                UU.push_back(U);
                SS.push_back(S);
                VV.push_back(V);
            }
        }
        double this_time = timer.click();
        total_time += this_time;
        std::cout << std::setprecision(10) << "impQR time: " << this_time << std::endl;
    }
    std::cout << std::setprecision(10) << "impQR Average time: " << total_time / (double)(repeat) << std::endl;
    if (accuracy_test)
        testAccuracy(tests, UU, SS, VV);
}

template <class T>
void addRandomCases(std::vector<Eigen::Matrix<T, 3, 3> >& tests, const T random_range, const int N)
{
    using namespace JIXIE;
    int old_count = tests.size();
    std::cout << std::setprecision(10) << "Adding random test cases with range " << -random_range << " to " << random_range << std::endl;
    RandomNumber<T> random_gen(123);
    for (int t = 0; t < N; t++) {
        Eigen::Matrix<T, 3, 3> Z;
        random_gen.fill(Z, -random_range, random_range);
        tests.push_back(Z);
    }
    std::cout << std::setprecision(10) << tests.size() - old_count << " cases added." << std::endl;
    std::cout << std::setprecision(10) << "Total test cases: " << tests.size() << std::endl;
}

template <class T>
void addIntegerCases(std::vector<Eigen::Matrix<T, 3, 3> >& tests, const int int_range)
{
    using namespace JIXIE;
    int old_count = tests.size();
    std::cout << std::setprecision(10) << "Adding integer test cases with range " << -int_range << " to " << int_range << std::endl;
    Eigen::Matrix<T, 3, 3> Z;
    Z.fill(-int_range);
    typename Eigen::Matrix<T, 3, 3>::Index i = 0;
    tests.push_back(Z);
    while (i < Eigen::Matrix<T, 3, 3>::SizeAtCompileTime) {
        if (Z(i) < int_range) {
            Z(i)++;
            tests.push_back(Z);
            i = 0;
        }
        else {
            Z(i) = -int_range;
            i++;
        }
    }
    std::cout << std::setprecision(10) << tests.size() - old_count << " cases added." << std::endl;
    std::cout << std::setprecision(10) << "Total test cases: " << tests.size() << std::endl;
}

template <class T>
void addPerturbationFromIdentityCases(std::vector<Eigen::Matrix<T, 3, 3> >& tests, const int num_perturbations, const T perturb)
{
    using namespace JIXIE;
    int old_count = tests.size();
    std::vector<Eigen::Matrix<T, 3, 3> > tests_tmp;
    Eigen::Matrix<T, 3, 3> Z = Eigen::Matrix<T, 3, 3>::Identity();
    tests_tmp.push_back(Z);
    std::cout << std::setprecision(10) << "Adding perturbed identity test cases with perturbation " << perturb << std::endl;
    RandomNumber<T> random_gen(123);
    size_t special_cases = tests_tmp.size();
    for (size_t t = 0; t < special_cases; t++) {
        for (int i = 0; i < num_perturbations; i++) {
            random_gen.fill(Z, -perturb, perturb);
            tests.push_back(tests_tmp[t] + Z);
        }
    }
    std::cout << std::setprecision(10) << tests.size() - old_count << " cases added." << std::endl;
    std::cout << std::setprecision(10) << "Total test cases: " << tests.size() << std::endl;
}

template <class T>
void addPerturbationCases(std::vector<Eigen::Matrix<T, 3, 3> >& tests, const int int_range, const int num_perturbations, const T perturb)
{
    using namespace JIXIE;
    int old_count = tests.size();
    std::vector<Eigen::Matrix<T, 3, 3> > tests_tmp;
    Eigen::Matrix<T, 3, 3> Z;
    Z.fill(-int_range);
    typename Eigen::Matrix<T, 3, 3>::Index i = 0;
    tests_tmp.push_back(Z);
    while (i < Eigen::Matrix<T, 3, 3>::SizeAtCompileTime) {
        if (Z(i) < int_range) {
            Z(i)++;
            tests_tmp.push_back(Z);
            i = 0;
        }
        else {
            Z(i) = -int_range;
            i++;
        }
    }
    std::cout << std::setprecision(10) << "Adding perturbed integer test cases with perturbation " << perturb << " and range " << -int_range << " to " << int_range << std::endl;
    RandomNumber<T> random_gen(123);
    size_t special_cases = tests_tmp.size();
    for (size_t t = 0; t < special_cases; t++) {
        for (int i = 0; i < num_perturbations; i++) {
            random_gen.fill(Z, -perturb, perturb);
            tests.push_back(tests_tmp[t] + Z);
        }
    }
    std::cout << std::setprecision(10) << tests.size() - old_count << " cases added." << std::endl;
    std::cout << std::setprecision(10) << "Total test cases: " << tests.size() << std::endl;
}

void runBenchmark()
{
    using namespace JIXIE;
    using std::fabs;

    bool run_qr;

    bool test_float;
    bool test_double;
    bool accuracy_test;
    bool normalize_matrix;
    int number_of_repeated_experiments;
    bool test_random;
    int random_range;
    int number_of_random_cases;
    bool test_integer;
    int integer_range;
    bool test_perturbation;
    int perturbation_count;
    float float_perturbation;
    double double_perturbation;
    bool test_perturbation_from_identity;
    int perturbation_from_identity_count;
    float float_perturbation_identity;
    double double_perturbation_identity;
    std::string title;

    // Finalized options
    run_qr = true;

    test_float = true;
    test_double = true;
    normalize_matrix = false;
    int number_of_repeated_experiments_for_timing = 2;

    for (int test_number = 1; test_number <= 10; test_number++) {

        if (test_number == 1) {
            title = "random timing test";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = true, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, integer_range = 3, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 2) {
            title = "integer timing test";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = true; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 3) {
            title = "integer-perturbation timing test: 256 eps";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = true, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 4) {
            title = "identity-perturbation timing test: 1e-3";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = true, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 5) {
            title = "identity-perturbation timing test: 256 eps";
            number_of_repeated_experiments = number_of_repeated_experiments_for_timing;
            accuracy_test = false;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = true, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation_identity = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed itentity test
        }

        if (test_number == 6) {
            title = "random accuracy test";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = true, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, integer_range = 3, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 7) {
            title = "integer accuracy test";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = true; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 8) {
            title = "integer-perturbation accuracy test: 256 eps";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = true, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = false, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 9) {
            title = "identity-perturbation accuracy test: 1e-3";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = true, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = 1e-3, double_perturbation_identity = 1e-3; // perturbed itentity test
        }
        if (test_number == 10) {
            title = "identity-perturbation accuracy test: 256 eps";
            number_of_repeated_experiments = 1;
            accuracy_test = true;
            test_random = false, random_range = 3, number_of_random_cases = 1024 * 1024; // random test
            test_integer = false; // integer test
            integer_range = 2; // this variable is used by both integer test and perturbed integer test
            test_perturbation = false, perturbation_count = 4, float_perturbation = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed integer test
            test_perturbation_from_identity = true, perturbation_from_identity_count = 1024 * 1024, float_perturbation_identity = (float)256 * std::numeric_limits<float>::epsilon(), double_perturbation_identity = (double)256 * std::numeric_limits<double>::epsilon(); // perturbed itentity test
        }

        std::cout << " \n========== RUNNING BENCHMARK TEST == " << title << "=======" << std::endl;
        std::cout << " run_qr " << run_qr << std::endl;
        std::cout << " test_float " << test_float << std::endl;
        std::cout << " test_double " << test_double << std::endl;
        std::cout << " accuracy_test " << accuracy_test << std::endl;
        std::cout << " normalize_matrix " << normalize_matrix << std::endl;
        std::cout << " number_of_repeated_experiments " << number_of_repeated_experiments << std::endl;
        std::cout << " test_random " << test_random << std::endl;
        std::cout << " random_range " << random_range << std::endl;
        std::cout << " number_of_random_cases " << number_of_random_cases << std::endl;
        std::cout << " test_integer " << test_integer << std::endl;
        std::cout << " integer_range " << integer_range << std::endl;
        std::cout << " test_perturbation " << test_perturbation << std::endl;
        std::cout << " perturbation_count " << perturbation_count << std::endl;
        std::cout << " float_perturbation " << float_perturbation << std::endl;
        std::cout << " double_perturbation " << double_perturbation << std::endl;
        std::cout << " test_perturbation_from_identity " << test_perturbation_from_identity << std::endl;
        std::cout << " perturbation_from_identity_count " << perturbation_from_identity_count << std::endl;
        std::cout << " float_perturbation_identity " << float_perturbation_identity << std::endl;
        std::cout << " double_perturbation_identity " << double_perturbation_identity << std::endl;

        std::cout << std::setprecision(10) << "\n--- float test ---\n" << std::endl;
        if (test_float) {
            std::vector<Eigen::Matrix<float, 3, 3> > tests;
            if (test_integer)
                addIntegerCases(tests, integer_range);
            if (test_perturbation)
                addPerturbationCases(tests, integer_range, perturbation_count, float_perturbation);
            if (test_perturbation_from_identity)
                addPerturbationFromIdentityCases(tests, perturbation_from_identity_count, float_perturbation_identity);
            if (test_random)
                addRandomCases(tests, (float)random_range, number_of_random_cases);
            if (normalize_matrix) {
                for (size_t i = 0; i < tests.size(); i++) {
                    float norm = tests[i].norm();
                    if (norm > (float)8 * std::numeric_limits<float>::epsilon()) {
                        tests[i] /= norm;
                    }
                }
            }
            std::cout << std::setprecision(10) << "\n-----------" << std::endl;
            if (run_qr)
                runImplicitQRSVD(number_of_repeated_experiments, tests, accuracy_test);

        }

        std::cout << std::setprecision(10) << "\n--- double test ---\n" << std::endl;
        if (test_double) {
            std::vector<Eigen::Matrix<double, 3, 3> > tests;
            if (test_integer)
                addIntegerCases(tests, integer_range);
            if (test_perturbation)
                addPerturbationCases(tests, integer_range, perturbation_count, double_perturbation);
            if (test_perturbation_from_identity)
                addPerturbationFromIdentityCases(tests, perturbation_from_identity_count, double_perturbation_identity);
            if (test_random)
                addRandomCases(tests, (double)random_range, number_of_random_cases);
            if (normalize_matrix) {
                for (size_t i = 0; i < tests.size(); i++) {
                    double norm = tests[i].norm();
                    if (norm > (double)8 * std::numeric_limits<double>::epsilon()) {
                        tests[i] /= norm;
                    }
                }
            }
            std::cout << std::setprecision(10) << "\n-----------" << std::endl;
            if (run_qr)
                runImplicitQRSVD(number_of_repeated_experiments, tests, accuracy_test);

        }
    }
}


void My_SVD(const Eigen::Matrix2f& F,Eigen::Matrix2f& U,Eigen::Matrix2f& sigma,Eigen::Matrix2f& V){
    //
    //Compute the SVD of input F with sign conventions discussed in class and in assignment
    //
    //input: F
    //output: U,sigma,V with F=U*sigma*V.transpose() and U*U.transpose()=V*V.transpose()=I;

}

template<typename T>
void Algorithm_2_Test(){
    /** Pseudocode
    1) C = F^T F
    2) V_hat, Sigma_hat ^2 = Jacobi(C)
    3) sigma_i_hat = sqrt(sigma_i_hat^2)
    4) sort sigma_i_hats and swap rows of V_hat accordingly to get Sigma_bar, V_bar
    5) A = FV_bar
    6) Givens
    7) Flip signs / swap cols
    */

    Eigen::Matrix<T, 2, 2> F,C,V_hat,A,QT, U,temp;
    //float sigma_i_hats[2] = {};
    F<<    1.2500,    0.4330,  0.4330,    1.7500;
    //cout << F << endl;
    //1) C = F^T F
    C=F.transpose()*F;
    cout << "C" << C << endl;
    //Id << 1,0,0,1;

    //2) Compute V_hat, Sigma_hat ^2 = Jacobi(C) (using Jacobi rotation)
    T tau;
    T t;
    T c;
    T s;
    // Find c and s
    if(abs(C(1))==0){ // Do the Jacobi rotation (off-diag nonzero)
      V_hat << 1,0,0,1;
    }
    else{
      tau = (C(3)-C(0))/(2);//*abs(C(1)));
      if (tau > 0) { // Choose tau as small as possible
        t = C(1)/(tau+sqrt(tau*tau+C(1)*C(1)));//t = 1/(tau+sqrt(1+tau*tau));
      }
      else{
        t = C(1)/(tau-sqrt(tau*tau+C(1)*C(1)));//t = 1/(abs(tau) - sqrt(1+tau*tau));
      }
      c = 1/sqrt((1+t*t));
      s = -t*c;
      V_hat << c, -s,s,c;
      C = V_hat.transpose()*C*V_hat;
      cout << "test" << endl;
      cout << C << endl;
    }
    //float sigma1 = c*c*C(0)-2*c*s*C(1)+s*s*C(3); DOESN'T MAKE ANY SENSE HERE SINCE C REDEFINED
    //float sigma2 = s*s*C(0)+2*c*s*C(1)+c*c*C(3);

    //cout << "Sigma_hat^2 " << C << endl; // REMOVE LATER
    //cout << "c: " << c << " s: " << s << endl;
    //cout << "V_hat: " << V_hat << endl;

    /**Eigen::JacobiRotation<float> r(2,2); // Initialize Jacobi rotation
    // Write your own!!!!
    r.makeJacobi(C(0),C(1),C(3));
    C.applyOnTheLeft(0,1,r.transpose());
    C.applyOnTheRight(0,1,r); // Now C = Sigma_hat^2


    cout << "Sigma_hat^2 " << C << endl; // REMOVE LATER
    //Calculate V_hat: V_hat is given by the following lines
    V_hat << 1,0,0,1;
    V_hat.applyOnTheRight(0,1,r);
    */
    //cout << "FIrst V_hat " << V_hat << endl;


    //3/4) sigma_i_hat = sqrt(sigma_i_hat^2), sort and adjust V_hat!
    bool V_det_Neg = false;
    Eigen::Matrix<T, 2, 1> sigma_i_hats;
    if (C(0)>C(3)) { // Note: these are necessarily non-negative
      C << sqrt(C(0)),0,0, sqrt(C(3));
    } else { // We have a non-trivial sort and need to change V_hat
      T sig_temp = sqrt(C(0));
      C << sqrt(C(3)),0,0, sig_temp;
      V_det_Neg = true;
      V_hat<< -s,c,c,s;
    }
    //cout << "sigma_i_hats" << sigma_i_hats << endl;

    //cout << "V_hat " << V_hat << endl;

    //5) A = FV_bar
    A = F*V_hat;

    cout << "A" << endl;
    cout << A << endl;

    //6) Givens: A = QR
    //Eigen::JacobiRotation<float> r2(0, 1); // Initialize Jacobi rotation
    //r.makeGivens(A(0),A(2),&sigma_i_hats(0));
    //float t;
    T d = pow(A(0),2)+pow(A(1),2);
    c = 1;
    s = 0;
    if (d != 0) {
      t = sqrt(d);
      c = A(0)/t;
      s = -A(1)/t;
    }
    QT << c, -s, s, c;

    cout << "QT"  << endl;
    cout << QT << endl;
    //bool signCheck = ((Q*A)(3))>0;
    cout << "QT*A: "  << endl;
    cout <<  QT*A << endl;

    //6) Create U
    //float sign = pow(-1,((sigma2)>0)+1);
    temp = QT*A;
    bool U_det_Neg = (temp(3)<0);
    cout << "det U neg?" <<temp(3)<< " " << U_det_Neg <<endl;
    T sign = pow(-1,U_det_Neg);
    cout << sign <<endl;
    U << QT(0), QT(2), sign*(QT(1)), sign*(QT(3));
    // A = Q'R implies Q'A = R
    //U = (Q*A)*V_hat;

    cout << "U" << endl;
    cout << U << endl;

    //7) Flip signs / swap cols
    cout << "Pre Check: "<< endl;
    cout <<  U*C*V_hat.transpose() << endl;
    cout << "V_hat: "<< endl;
    cout <<  V_hat << endl;
    cout << "U: "<< endl;
    cout <<  U << endl;

    if (F(0)*F(3)-F(1)*F(2)<0) {
      C(3) = -C(3);
      if (U_det_Neg) {
        cout << "U col sign flip" << endl;
        U(2) = -U(2);
        U(3) = -U(3);
      } else {
        cout << "V col sign flip" << endl;
        V_hat(2) = -V_hat(2);
        V_hat(3) = -V_hat(3);
      }
    }
    else{// i.e. det(F)>=0
      if (U_det_Neg && V_det_Neg) {
        // "U col sign flip"
        U(2) = -U(2);
        U(3) = -U(3);
        // "V col sign flip"
        V_hat(2) = -V_hat(2);
        V_hat(3) = -V_hat(3);
      }
    }

    cout << "Check: "<< endl;
    cout <<  U*C*V_hat.transpose() << endl;
    cout << "V_hat: "<< endl;
    cout <<  V_hat << endl;
    cout << "U: "<< endl;
    cout <<  U << endl;
    cout << "F" << endl;
    cout << F << endl;
    // Seems to be working - Still need to clean up and test.
}



/** Computes the polar decomposition of F=RS F 3x3 real matrix, using algorithm
3.
*/
//template<typename T>
void My_Polar(const Eigen::Matrix3f& F,Eigen::Matrix3f& R,Eigen::Matrix3f& S){

  Eigen::Matrix<float, 3, 3> temp; // F,R,S
  Eigen::JacobiRotation<float> G; // Initialize Givens rotation
  //F << 4,3,4,56,2,3,7,3,8;
  S = F;
  R << 1,0,0,0,1,0,0,0,1; // Identity
  int it = 0;
  int max_it = 1000;
  float tol = .0001;
  //T denom;
  Eigen::Vector2f v;
  //Note max(|S21 − S12|, |S31 − S13|, |S32 − S23|) considered
  while (it<max_it&&max(abs(S(3)-S(1)),max(abs(S(6)-S(2)),abs(S(7)-S(5))))>tol){
    for (size_t j = 1; j < 3; j++) {
      for (size_t i = 0; i < j; i++) {
        v << S(3*i+i)+ S(3*j+j),S(3*i+j)-S(3*j+i);
        G.makeGivens(v.x(), v.y());
        R.applyOnTheRight(i,j, G);
        S.applyOnTheLeft(i,j,G.adjoint());
      }
    }
    it += 1;
  }
  if(it==max_it){
    cout << "MAXIMUM ITERATIONS REACHED BEFORE DESIRED TOLERANCE" << endl;
  }
  cout << "iter: " << it << endl;
  cout << "R" << endl;
  cout << R << endl;
  cout << "S" << endl;
  cout << S << endl;
  cout << "R*S" << endl;
  cout << R*S << endl;


}
/**
void Givens_test(){
  Eigen::JacobiRotation<float> G; // Initialize Givens rotation
  Eigen::Vector2f v = Eigen::Vector2f::Random();
  v << 1, 2;
  G.makeGivens(v.x(), v.y());
  cout << "Here is the vector v:" << endl << v << endl;
  v.applyOnTheLeft(0, 1, G.adjoint());
  cout << "Here is the vector J' * v:" << endl << v << endl;

}*/



//void My_Polar(const Eigen::Matrix3f& F,Eigen::Matrix3f& R,Eigen::Matrix3f& S){
  //
  //Compute the polar decomposition of input F (with det(R)=1)
  //
  //input: F
  //output: R,s with F=R*S and R*R.transpose()=I and S=S.transpose()

//}

// void Algorithm_2_Test(){
//
//   Eigen::Matrix2f F,C,U,V;
//   F<<1,2,3,4;
//   C=F*F.transpose();
//   Eigen::Vector2f s2;
//   JIXIE::Jacobi(C,s2,V);
//
// }

int main()
{
  Eigen::Matrix<float, 3, 3> F,R,S;
  F << 1,2,3,4,5,6,7,8,9;
  R << 0,0,0,0,0,0,0,0,0;
  S << 0,0,0,0,0,0,0,0,0;
  My_Polar(F,R,S);

  //Givens_test();

  bool run_benchmark = false;
  if (run_benchmark) runBenchmark();

  //Algorithm_3_Test<float>();


}
