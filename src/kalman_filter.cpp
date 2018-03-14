#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

// convert the passed in angle to an equal one between [-PI, PI)
static double normalizeAngle(double phi) {
  if (phi >= -Tools::PI && phi < Tools::PI) {
    return phi;
  }

  static const auto two_pi = 2.0 * Tools::PI;

  const auto delta = (phi < -Tools::PI) ? two_pi : -two_pi;
  while (phi < -Tools::PI || phi >= Tools::PI) {
    phi += delta;
  }
  return phi;
}

} // end of anonymous namespace

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  const auto py = x_(1);
  const auto vx = x_(2);
  const auto px = x_(0);
  const auto vy = x_(3);
  const auto d = sqrt(px * px + py * py);

  VectorXd z_pred(z.size());
  // prevent division by zero
  const auto rodot = (d < Tools::EPS) ? 0.0 : (px * vx + py * vy) / d;
  z_pred << d, atan2(py, px), rodot;
  VectorXd y = z - z_pred;

  // adjust the angle phi to [-pi, pi)
  y(1) = normalizeAngle(y(1));

  UpdateCommon(y);

}

void KalmanFilter::UpdateCommon(const Eigen::VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  P_ -= K * H_ * P_;
}
