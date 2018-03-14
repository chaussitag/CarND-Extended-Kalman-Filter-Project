#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

// static variables
const double Tools::EPS = 0.0001;
const double Tools::PI  = 3.1416;


Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() < 1) {
    cerr << "the size of ground_truth and estimations should be same and non-zero" << endl;
    return rmse;
  }

  //accumulate squared residuals
  const int n = estimations.size();
  for(int i = 0; i < n; ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / n;

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
    * Calculate a Jacobian here.
  */
  MatrixXd Hj(3,4);
  // recover state parameters
  const auto px = x_state(0);
  const auto py = x_state(1);
  const auto vx = x_state(2);
  const auto vy = x_state(3);

  // prevent division by zero
  const auto d_square = std::max(px * px + py * py, Tools::EPS);
  //compute the Jacobian matrix
  const auto d = sqrt(d_square);
  const auto d3 = d * d_square;
  Hj << px / d, py / d, 0, 0,
      -py / d_square, px / d_square, 0, 0,
      py * (vx * py - vy * px) / d3, px * (vy * px - vx * py) / d3, px / d, py / d;

  return Hj;
}
