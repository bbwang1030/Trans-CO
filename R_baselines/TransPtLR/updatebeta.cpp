// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
using namespace Rcpp;
#include <RcppEigen.h>
using namespace Eigen;
using namespace std;

// [[Rcpp::export]]
NumericVector updatebeta(MatrixXd x,VectorXd beta,VectorXd residual,VectorXd tauhat0,double lambda) {
  int p = x.cols();

  for (int j =0 ; j <p; j++) {
    residual = residual + x.col(j) * beta[j];
    
    // single variable OLS estimate
    //beta_ols_j = (residual.cwiseProduct(tauhat0).transpose().dot(x.col(j)))/(x.col(j).cwiseProduct(tauhat0).transpose().dot(x.col(j)));
    double denominator = x.col(j).cwiseProduct(tauhat0).transpose().dot(x.col(j));
    double beta_ols_j = (denominator != 0) ? (residual.cwiseProduct(tauhat0).transpose().dot(x.col(j))) / denominator : 0.0;
    
    // soft-threshold the result
    beta[j] = (beta_ols_j > 0) ? std::max(beta_ols_j - lambda, 0.0) : ((beta_ols_j < 0) ? std::min(beta_ols_j + lambda, 0.0) : 0.0);
    
    if (std::isnan(beta[j])) {
      beta[j] = 0.0;
    }
    
    // restore the residual
    residual = residual - x.col(j)*beta[j];
  }
  return wrap(beta);
}




