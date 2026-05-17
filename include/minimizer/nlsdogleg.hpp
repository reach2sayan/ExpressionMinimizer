#pragma once

#include "trustregionbase.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace exprmin {

enum class DoglegVariant { Standard, Double };

// NLS Powell dogleg trust-region method for minimizing 1/2 ||R(theta)||^2,
// where R : R^N -> R^M (M >= N) is given as diff::Equation<R1,...,RM>.
//
// The shared trust-region loop and NLS state management live in
// NLSTrustRegionBase. This class only chooses the dogleg step geometry.
template <typename System, DoglegVariant DV = DoglegVariant::Standard>
struct NLSDogleg;

template <diff::CExpression... RExprs, DoglegVariant DV>
struct NLSDogleg<diff::Equation<RExprs...>, DV>
    : NLSTrustRegionBase<NLSDogleg<diff::Equation<RExprs...>, DV>, RExprs...> {
  using Base =
      NLSTrustRegionBase<NLSDogleg<diff::Equation<RExprs...>, DV>, RExprs...>;
  using value_type = typename Base::value_type;
  using ParamVec = typename Base::ParamVec;
  using NMat = typename Base::NMat;
  using RVec = typename Base::RVec;
  using JMat = typename Base::JMat;

  using Base::Base;
  using Base::get_optimal_value;
  using Base::iter;

  constexpr ParamVec compute_step(const ParamVec &g, const NMat &B,
                                  value_type delta) const;
};

template <diff::CExpression... RExprs, DoglegVariant DV>
constexpr typename NLSDogleg<diff::Equation<RExprs...>, DV>::ParamVec
NLSDogleg<diff::Equation<RExprs...>, DV>::compute_step(const ParamVec &g,
                                                       const NMat &B,
                                                       value_type delta) const {
  using std::abs, std::min, std::sqrt;
  const JMat &J = this->current_J();

  // Cauchy step: dx_sd = -alpha * g, alpha = ||g||^2 / ||Jg||^2.
  const RVec Jg = J * g;
  const auto Jg_sq = Jg.squaredNorm();
  const auto alpha =
      (Jg_sq > value_type{0}) ? g.squaredNorm() / Jg_sq : value_type{1};
  const ParamVec dx_sd = -alpha * g;

  // Gauss-Newton step: (J^T J) dx_gn = -g.
  const ParamVec dx_gn = -(B.ldlt().solve(g));
  const auto norm_sd = dx_sd.norm();
  const auto norm_gn = dx_gn.norm();

  if (norm_sd >= delta) {
    return (delta / norm_sd) * dx_sd;
  }

  if (norm_gn <= delta) {
    return dx_gn;
  }

  auto t = value_type{1};
  if constexpr (DV == DoglegVariant::Double) {
    const auto gBg = g.dot(B * g);
    const auto gdx = abs(g.dot(dx_gn));
    if (gBg > value_type{0} && gdx > value_type{0}) {
      const auto g2 = g.squaredNorm();
      const auto c = min(value_type{1}, (g2 * g2) / (gBg * gdx));
      t = value_type{1} - value_type{0.8} * (value_type{1} - c);
    }
    if (t * norm_gn <= delta) {
      return (delta / norm_gn) * dx_gn;
    }
  }

  const auto d = t * dx_gn - dx_sd;
  const auto a_q = d.squaredNorm();
  const auto b_q = value_type{2} * dx_sd.dot(d);
  const auto c_q = dx_sd.squaredNorm() - delta * delta;
  auto disc = b_q * b_q - value_type{4} * a_q * c_q;
  if (disc < value_type{0}) {
    disc = value_type{0};
  }
  const auto beta = (-b_q + sqrt(disc)) / (value_type{2} * a_q);
  return dx_sd + beta * d;
}

template <diff::CExpression... RExprs>
NLSDogleg(diff::Equation<RExprs...>) -> NLSDogleg<diff::Equation<RExprs...>>;

template <diff::CExpression... RExprs, typename T>
NLSDogleg(diff::Equation<RExprs...>, T) -> NLSDogleg<diff::Equation<RExprs...>>;

} // namespace exprmin
