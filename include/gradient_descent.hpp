/**
 * @file gradient_descent.hpp
 * @brief Gradient-descent arm optimizer for STRIDER morphing quadrotor.
 *
 * Extracted from Sim_STRIDER/compare branch.
 * Maximises energy-efficiency η and controllability C of the 4-arm
 * configuration using projected gradient ascent (eq.10 of the paper).
 *
 * USAGE
 * -----
 *   1. Add the required parameters to your params.hpp (see below).
 * 
        // ===== gradient descent parameters =====
        inline constexpr double ARM_OPT_BETA1    = 0.01;                   // η ascent rate
        inline constexpr double ARM_OPT_BETA2    = 0.01;                  // C ascent rate
        inline constexpr double ARM_OPT_EPS      = 1e-4;                   // finite difference step

        inline constexpr double POWER_GAMMA      = 10.000;                 // loss factor
        inline constexpr double AIR_DENSITY      = 1.225;                  // kg/m³ 
        inline constexpr double PROP_DISK_AREA   = M_PI * 0.1524 * 0.1524; // 12-inch prop radius = 0.1524m 

 *   2. #include "gradient_descent.hpp" after params.hpp and utils.hpp.
 *   3. Call  GD::arm_cmd(s, cmd, tilt_des, thrust_des) !! in "main.cpp" 416~419 line
 *      Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (f_1234 > 0)
 *      Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
 *      Sequential_Allocation(f_sum, tau_des, cmd.tauz_bar, delayed_s.arm_q, s.r_com, thrust_des, tilt_ang_des);
 *      if (AUTO_PHASE ==  Phase::GAC_ONLY && elapsed_double < param::BUILD_TIME) GD::arm_cmd(s, cmd, tilt_ang_des, thrust_des); <--------this part!!
 *
 *
 */

#ifndef GRADIENT_DESCENT_HPP
#define GRADIENT_DESCENT_HPP

#include <cmath>
#include <cstdio>
#include <array>
#include <Eigen/Dense>

namespace GD {

static inline Eigen::Vector2d polar_to_cart(const Eigen::Vector2d& polar, int arm_idx) {

  const double rho   = polar(0);
  const double alpha = polar(1);

  return Eigen::Vector2d(param::B2BASE_X[arm_idx] + rho * std::cos(alpha), param::B2BASE_Y[arm_idx] + rho * std::sin(alpha));
}

static inline Eigen::Vector2d cart_to_polar(const Eigen::Vector2d& cart, int arm_idx) {

  constexpr double eps = 1e-12;

  const double dx = cart(0) - param::B2BASE_X[arm_idx];
  const double dy = cart(1) - param::B2BASE_Y[arm_idx];

  const double rho = std::sqrt(dx * dx + dy * dy);
  double alpha = (rho > eps) ? std::atan2(dy, dx) : 0.0;

  alpha = spin_360(alpha, param::ALPHA_MIN[arm_idx], param::ALPHA_MAX[arm_idx]);

  return Eigen::Vector2d(rho, alpha);
}

static inline bool build_A1_matrix(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2,       const Eigen::Vector2d& p3,    const Eigen::Vector2d& p4,
                                   const Eigen::Vector3d& Pc, const Eigen::Vector4d& tilt_des, Eigen::Matrix4d&       A1_out) {

  const double pcx = Pc(0), pcy = Pc(1);
  const double s1 = std::sin(tilt_des(0)), c1 = std::cos(tilt_des(0));
  const double s2 = std::sin(tilt_des(1)), c2 = std::cos(tilt_des(1));
  const double s3 = std::sin(tilt_des(2)), c3 = std::cos(tilt_des(2));
  const double s4 = std::sin(tilt_des(3)), c4 = std::cos(tilt_des(3));

  A1_out(0,0) = -inv_sqrt2*param::PWM_ZETA*s1 + (pcy - p1(1))*c1;
  A1_out(0,1) = -inv_sqrt2*param::PWM_ZETA*s2 + (pcy - p2(1))*c2;
  A1_out(0,2) = -inv_sqrt2*param::PWM_ZETA*s3 + (pcy - p3(1))*c3;
  A1_out(0,3) = -inv_sqrt2*param::PWM_ZETA*s4 + (pcy - p4(1))*c4;

  A1_out(1,0) = -inv_sqrt2*param::PWM_ZETA*s1 + (p1(0) - pcx)*c1;
  A1_out(1,1) = -inv_sqrt2*param::PWM_ZETA*s2 + (p2(0) - pcx)*c2;
  A1_out(1,2) = -inv_sqrt2*param::PWM_ZETA*s3 + (p3(0) - pcx)*c3;
  A1_out(1,3) = -inv_sqrt2*param::PWM_ZETA*s4 + (p4(0) - pcx)*c4;

  A1_out(2,0) = -param::PWM_ZETA * c1;
  A1_out(2,1) =  param::PWM_ZETA * c2;
  A1_out(2,2) = -param::PWM_ZETA * c3;
  A1_out(2,3) =  param::PWM_ZETA * c4;

  A1_out(3,0) = -c1;
  A1_out(3,1) = -c2;
  A1_out(3,2) = -c3;
  A1_out(3,3) = -c4;

  return true;
}

static inline double eta(const Eigen::Matrix4d& A1) {

  //  η (energy efficiency)  — eq.(7)
  Eigen::FullPivLU<Eigen::Matrix4d> lu(A1);
  if (!lu.isInvertible()) return 0.0;

  Eigen::Vector4d T_hover(0.0, 0.0, 0.0, -param::TOTAL_MASS * param::G);
  Eigen::Vector4d f = lu.solve(T_hover);

  double sum_f = f.sum();
  double sum_power = 0.0;

  for (int i = 0; i < 4; ++i) {
    if (f(i) > 0) {
      double power_i = param::POWER_GAMMA * std::sqrt(f(i)*f(i)*f(i) / (2.0 * param::AIR_DENSITY * param::PROP_DISK_AREA));
      sum_power += power_i;
    }
  }

  return (sum_power > 0.0) ? sum_f / sum_power : 0.0;
}

static inline double controllability(const Eigen::Matrix4d& A1) {

  //  C (controllability)  — eq.(8)
  Eigen::FullPivLU<Eigen::Matrix4d> lu(A1);
  if (!lu.isInvertible()) return 0.0;

  Eigen::Matrix4d A1inv = lu.inverse();

  const Eigen::Map<const Eigen::Matrix3d> J_full(param::J);
  Eigen::Matrix<double, 3, 2> J_sub;
  J_sub << J_full(0,0), -J_full(0,1),
           J_full(1,0),  J_full(1,1),
           J_full(2,0), -J_full(2,1);

  double max_norm = 0.0;
  for (int i = 0; i < 4; ++i) {
    Eigen::RowVector3d Gi = A1inv.row(i).segment(1, 3);
    Eigen::RowVector2d Si = Gi * J_sub;
    max_norm = std::max(max_norm, Si.norm());
  }

  return (max_norm > 0.0) ? 1.0 / max_norm : 0.0;
}

static inline Eigen::Vector2d estimate_CoM_Hover(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& p3, const Eigen::Vector2d& p4, const Eigen::Vector4d& thrust_des) {
  const double sum_f = thrust_des.sum();
  if (std::abs(sum_f) < 1e-6) return (p1 + p2 + p3 + p4) * 0.25;

  Eigen::Vector2d CoM;
  CoM(0) = (p1(0)*thrust_des(0) + p2(0)*thrust_des(1) + p3(0)*thrust_des(2) + p4(0)*thrust_des(3)) / sum_f;
  CoM(1) = (p1(1)*thrust_des(0) + p2(1)*thrust_des(1) + p3(1)*thrust_des(2) + p4(1)*thrust_des(3)) / sum_f;

  return CoM;
}

static inline void gradients(Eigen::Vector2d&             p1,           Eigen::Vector2d&             p2,       Eigen::Vector2d& p3, Eigen::Vector2d& p4,
                             const Eigen::Vector3d&       Pc,           const Eigen::Vector4d&       tilt_des,
                             Eigen::Matrix<double, 4, 2>& grad_eta_out, Eigen::Matrix<double, 4, 2>& grad_C_out) {

  Eigen::Matrix4d A1_base;
  build_A1_matrix(p1, p2, p3, p4, Pc, tilt_des, A1_base);
  const double eta_base = eta(A1_base);
  const double C_base   = controllability(A1_base);

  Eigen::Vector2d* p[4] = {&p1, &p2, &p3, &p4};
  for (int i = 0; i < 4; ++i) {
    for (int ax = 0; ax < 2; ++ax) {
      (*p[i])(ax) += param::ARM_OPT_EPS;

      Eigen::Matrix4d A1_perturbed;
      build_A1_matrix(p1, p2, p3, p4, Pc, tilt_des, A1_perturbed);

      const double eta_pert = eta(A1_perturbed);
      const double C_pert   = controllability(A1_perturbed);

      (*p[i])(ax) -= param::ARM_OPT_EPS;
      //  Numerical gradients  ∇η, ∇C  (finite differences)
      grad_eta_out(i, ax) = (eta_pert - eta_base) / param::ARM_OPT_EPS;
      grad_C_out(i, ax)   = (C_pert   - C_base)   / param::ARM_OPT_EPS;
    }
  }
}

static inline void arm_cmd(const State& s, Command& cmd, const Eigen::Vector4d& tilt_des, const Eigen::Vector4d& thrust_des) {

  static Eigen::Vector2d p1 = polar_to_cart(cmd.r1, 0);
  static Eigen::Vector2d p2 = polar_to_cart(cmd.r2, 1);
  static Eigen::Vector2d p3 = polar_to_cart(cmd.r3, 2);
  static Eigen::Vector2d p4 = polar_to_cart(cmd.r4, 3);

  const Eigen::Vector2d r_com_xy = estimate_CoM_Hover(p1, p2, p3, p4, thrust_des);
  const Eigen::Vector3d Pc_hover(r_com_xy(0), r_com_xy(1), s.r_com(2));

  Eigen::Matrix<double, 4, 2> grad_eta, grad_C;
  gradients(p1, p2, p3, p4, Pc_hover, tilt_des, grad_eta, grad_C);

  // --- project ∇C onto orthogonal complement of ∇η ---
  double dot_product = 0.0, norm_eta_sq = 0.0;
  for (int i = 0; i < 4; ++i) {
    for (int ax = 0; ax < 2; ++ax) {
      dot_product += grad_C(i, ax) * grad_eta(i, ax);
      norm_eta_sq += grad_eta(i, ax) * grad_eta(i, ax);
    }
  }
  const double proj_scalar = (norm_eta_sq > 1e-12) ? dot_product / norm_eta_sq : 0.0;
  const Eigen::Matrix<double, 4, 2> grad_C_orth = grad_C - proj_scalar * grad_eta;

  //  θ_{k+1} = θ_k + β₁·∇η + β₂·(∇C − proj_{∇η}∇C)       — eq.(10)
  p1 += param::ARM_OPT_BETA1 * grad_eta.row(0).transpose() + param::ARM_OPT_BETA2 * grad_C_orth.row(0).transpose();
  p2 += param::ARM_OPT_BETA1 * grad_eta.row(1).transpose() + param::ARM_OPT_BETA2 * grad_C_orth.row(1).transpose();
  p3 += param::ARM_OPT_BETA1 * grad_eta.row(2).transpose() + param::ARM_OPT_BETA2 * grad_C_orth.row(2).transpose();
  p4 += param::ARM_OPT_BETA1 * grad_eta.row(3).transpose() + param::ARM_OPT_BETA2 * grad_C_orth.row(3).transpose();

  // --- convert back to polar & enforce feasibility ---
  std::array<Eigen::Vector2d, 4> feas = {cart_to_polar(p1, 0), cart_to_polar(p2, 1), cart_to_polar(p3, 2), cart_to_polar(p4, 3)};

  if (make_feasible(feas)) {
    cmd.r1 = feas[0];
    cmd.r2 = feas[1];
    cmd.r3 = feas[2];
    cmd.r4 = feas[3];
  }
}

}

#endif // GRADIENT_DESCENT_HPP