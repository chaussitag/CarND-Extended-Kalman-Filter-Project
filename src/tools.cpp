#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

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
  int n = estimations.size();
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
  auto px = x_state(0);
  auto py = x_state(1);
  auto vx = x_state(2);
  auto vy = x_state(3);

  // check division by zero
  auto d_square = px * px + py * py;
  if (d_square < Tools::EPS) {
    cerr << "invalid state, resuling division by zero!!" << endl;
    return Hj;
  }

  //compute the Jacobian matrix
  auto d = sqrt(d_square);
  auto d3 = d * d_square;
  Hj << px / d, py / d, 0, 0,
      -py / d_square, px / d_square, 0, 0,
      py * (vx * py - vy * px) / d3, px * (vy * px - vx * py) / d3, px / d, py / d;

  return Hj;
}
