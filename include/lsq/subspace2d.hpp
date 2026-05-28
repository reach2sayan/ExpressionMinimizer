#pragma once

#include "../minimizer/trustregionbase.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace exprmin {

// NLS 2D subspace trust-region method for minimizing 1/2 ||R(theta)||^2,
// where R : R^N -> R^M (M >= N) is given as diff::Equation<R1,...,RM>.
//
// The shared trust-region loop and NLS state management live in
// NLSTrustRegionBase. This class only computes the constrained subspace step.
template <typename System> struct Subspace2D;

template <diff::CExpression... RExprs>
struct Subspace2D<diff::Equation<RExprs...>>
    : NLSTrustRegionBase<Subspace2D<diff::Equation<RExprs...>>, RExprs...> {
  using Base =
      NLSTrustRegionBase<Subspace2D<diff::Equation<RExprs...>>, RExprs...>;
  using value_type = typename Base::value_type;
  static constexpr int N = Base::N;
  using ParamVec = typename Base::ParamVec;
  using NMat = typename Base::NMat;
  using RVec = typename Base::RVec;
  using JMat = typename Base::JMat;
  using WMat = Eigen::Matrix<value_type, N, 2>;

  using Base::Base;
  using Base::get_optimal_value;
  using Base::iter;

  constexpr ParamVec compute_step(const ParamVec &g, const NMat &B,
                                  value_type delta) const;
};

template <diff::CExpression... RExprs>
constexpr typename Subspace2D<diff::Equation<RExprs...>>::ParamVec
Subspace2D<diff::Equation<RExprs...>>::compute_step(const ParamVec &g,
                                                    const NMat &B,
                                                    value_type delta) const {
  using std::abs, std::sqrt;
  constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();
  const JMat &J = this->current_J();

  // Cauchy step: dx_sd = -alpha * g.
  const RVec Jg = J * g;
  const auto Jg_sq = Jg.squaredNorm();
  const auto alpha =
      (Jg_sq > value_type{0}) ? g.squaredNorm() / Jg_sq : value_type{1};
  const auto dx_sd = -alpha * g;

  // Gauss-Newton step.
  const auto dx_gn = -(B.ldlt().solve(g));
  const auto norm_gn = dx_gn.norm();

  if (norm_gn <= delta) {
    return dx_gn;
  }

  const auto norm_sd = dx_sd.norm();
  const ParamVec q0 = dx_sd / norm_sd;

  ParamVec q1_raw = dx_gn - q0.dot(dx_gn) * q0;
  const auto q1_norm = q1_raw.norm();
  const int rank = (q1_norm < EPS * norm_gn) ? 1 : 2;

  if (rank == 1) {
    return (delta / norm_sd) * dx_sd;
  }

  const auto q1 = q1_raw / q1_norm;
  WMat W;
  W << q0, q1;

  const auto subg = W.transpose() * g;
  const auto subB = W.transpose() * B * W;

  const auto trB = subB.trace();
  const auto detB = subB.determinant();
  const auto d2 = delta * delta;

  Eigen::Matrix<value_type, 2, 2> adjB;
  adjB << subB(1, 1), -subB(0, 1), -subB(1, 0), subB(0, 0);

  const auto adjBg = adjB * subg;
  const auto term0 = adjBg.squaredNorm();
  const auto term1 = subg.dot(adjBg);
  const auto subg2 = subg.squaredNorm();

  const auto a3 = value_type{2} * trB;
  const auto a2 = trB * trB + value_type{2} * detB - subg2 / d2;
  const auto a1 = value_type{2} * (detB * trB - term1 / d2);
  const auto a0 = detB * detB - term0 / d2;

  // clang-format off
  Eigen::Matrix4d C;
  C << 0.0, 0.0, 0.0, -static_cast<double>(a0),
       1.0, 0.0, 0.0, -static_cast<double>(a1),
       0.0, 1.0, 0.0, -static_cast<double>(a2),
       0.0, 0.0, 1.0, -static_cast<double>(a3);
  // clang-format on

  Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> es(C, false);
  const auto eigs = es.eigenvalues();

  value_type best_obj = std::numeric_limits<value_type>::max();
  Eigen::Vector<value_type, 2> y_best = Eigen::Vector<value_type, 2>::Zero();
  bool found = false;

  for (int k = 0; k < 4; ++k) {
    if (abs(eigs[k].imag()) > value_type{1e-6} * abs(eigs[k].real())) {
      continue;
    }
    const auto lam = static_cast<value_type>(eigs[k].real());
    Eigen::Matrix<value_type, 2, 2> A2 = subB;
    A2.diagonal().array() += lam;
    const value_type det2 = A2.determinant();
    if (abs(det2) < EPS) {
      continue;
    }

    const auto y_raw = -(A2.inverse() * subg);
    const value_type yn = y_raw.norm();
    if (yn < EPS) {
      continue;
    }
    const auto y = (delta / yn) * y_raw;
    const value_type obj = subg.dot(y) + value_type{0.5} * y.dot(subB * y);
    if (obj < best_obj) {
      best_obj = obj;
      y_best = y;
      found = true;
    }
  }
  if (found) {
    return W * y_best;
  }
  return (delta / norm_sd) * dx_sd;
}

template <diff::CExpression... RExprs>
Subspace2D(diff::Equation<RExprs...>) -> Subspace2D<diff::Equation<RExprs...>>;

template <diff::CExpression... RExprs, typename T>
Subspace2D(diff::Equation<RExprs...>, T)
    -> Subspace2D<diff::Equation<RExprs...>>;

} // namespace exprmin
