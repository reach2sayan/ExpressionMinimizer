#pragma once

#include "equation.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>

namespace exprmin {

namespace mp = boost::mp11;

enum class DoglegVariant { Standard, Double };

// NLS Powell dogleg trust-region method for minimizing ½‖R(θ)‖²,
// where R : ℝᴺ → ℝᴹ (M ≥ N) is given as diff::Equation<R1,...,RM>.
//
// The Gauss-Newton Hessian B = JᵀJ is recomputed from the exact
// AD Jacobian at each accepted step (no BFGS approximation).
//
// DoglegVariant::Standard — classical Powell dogleg
// DoglegVariant::Double   — Dennis & Mei (1979) double dogleg
template <typename System, DoglegVariant DV = DoglegVariant::Standard>
struct NLSDogleg;

template <diff::CExpression... RExprs, DoglegVariant DV>
struct NLSDogleg<diff::Equation<RExprs...>, DV> {
  using Sys = diff::Equation<RExprs...>;
  static_assert(Sys::input_dim <= Sys::output_dim,
                "NLSDogleg requires a system with M >= N residuals");

  using value_type = typename Sys::value_type;
  using Syms = typename Sys::symbols;
  static constexpr int M = static_cast<int>(Sys::output_dim);
  static constexpr int N = static_cast<int>(Sys::input_dim);
  using ParamVec = Eigen::Vector<value_type, N>;
  using RVec = Eigen::Vector<value_type, M>;
  using JMat = Eigen::Matrix<value_type, M, N>;
  using NMat = Eigen::Matrix<value_type, N, N>;

  int iter{};

private:
  Sys system;
  value_type tol;
  int itmax;
  value_type trustregion0;
  value_type trustregion_min;
  value_type fret{};

  static constexpr value_type TR_DOWN_FACTOR = value_type{0.1};
  static constexpr value_type TR_DOWN_THRESHOLD = value_type{0.25};
  static constexpr value_type TR_UP_FACTOR = value_type{2.0};
  static constexpr value_type TR_UP_THRESHOLD = value_type{0.75};

  constexpr auto to_arr(const ParamVec &p) const noexcept {
    std::array<value_type, static_cast<std::size_t>(N)> arr;
    for (int i = 0; i < N; ++i)
      arr[static_cast<std::size_t>(i)] = p[i];
    return arr;
  }

  constexpr std::pair<RVec, JMat> eval_rJ(const ParamVec &p);

  // Select dogleg step given gradient g, B = J^T J, Jacobian J, radius delta.
  constexpr ParamVec compute_step(const ParamVec &g, const NMat &B,
                                  const JMat &J, value_type delta) const;

public:
  constexpr value_type get_optimal_value() const { return fret; }
  constexpr explicit NLSDogleg(diff::Equation<RExprs...> sys,
                               value_type tol_ = value_type{1e-8},
                               int itmax_ = 200,
                               value_type tr0_ = value_type{1e3},
                               value_type trmin_ = value_type{1e-12})
      : system{std::move(sys)}, tol{tol_}, itmax{itmax_}, trustregion0{tr0_},
        trustregion_min{trmin_} {}

  constexpr ParamVec minimize(ParamVec p);
};

template <diff::CExpression... RExprs, DoglegVariant DV>
constexpr std::pair<typename NLSDogleg<diff::Equation<RExprs...>, DV>::RVec,
                    typename NLSDogleg<diff::Equation<RExprs...>, DV>::JMat>
NLSDogleg<diff::Equation<RExprs...>, DV>::eval_rJ(const ParamVec &p) {
  system.update(Syms{}, to_arr(p));
  RVec r;
  if constexpr (M == 1) {
    r[0] = system.evaluate();
  } else {
    const auto r_arr = system.evaluate();
    r = Eigen::Map<const RVec>(r_arr.data());
  }
  const auto J_arr = system.template jacobian<diff::DiffMode::Reverse>();
  JMat J = Eigen::Map<
    const Eigen::Matrix<value_type, M, N, Eigen::RowMajor>
>(&J_arr[0][0]);

  return {r, J};
}

template <diff::CExpression... RExprs, DoglegVariant DV>
constexpr typename NLSDogleg<diff::Equation<RExprs...>, DV>::ParamVec
NLSDogleg<diff::Equation<RExprs...>, DV>::compute_step(const ParamVec &g,
                                                       const NMat &B,
                                                       const JMat &J,
                                                       value_type delta) const {
  using std::sqrt, std::abs, std::min;

  // Cauchy step: dx_sd = -alpha * g,  alpha = ‖g‖² / ‖Jg‖²
  const RVec Jg = J * g;
  const value_type Jg_sq = Jg.squaredNorm();
  const value_type alpha =
      (Jg_sq > value_type{0}) ? g.squaredNorm() / Jg_sq : value_type{1};
  const ParamVec dx_sd = -alpha * g;

  // Gauss-Newton step: (J^T J) dx_gn = -g
  const ParamVec dx_gn = -(B.ldlt().solve(g));

  const value_type norm_sd = dx_sd.norm();
  const value_type norm_gn = dx_gn.norm();

  if (norm_sd >= delta) {
    return (delta / norm_sd) * dx_sd; // Cauchy outside TR → truncate
  }

  if (norm_gn <= delta) {
    return dx_gn; // GN inside TR → use as-is (global minimizer of quadratic)
  }

  // GN outside, Cauchy inside: compute interpolation factor t
  value_type t = value_type{1};
  if constexpr (DV == DoglegVariant::Double) {
    // t = 1 - 0.8*(1 - c),  c = ‖g‖⁴ / (g^T B g · |g^T dx_gn|)
    const value_type gBg = g.dot(B * g);
    const value_type gdx = abs(g.dot(dx_gn));
    if (gBg > value_type{0} && gdx > value_type{0}) {
      const value_type g2 = g.squaredNorm();
      const value_type c = min(value_type{1}, (g2 * g2) / (gBg * gdx));
      t = value_type{1} - value_type{0.8} * (value_type{1} - c);
    }
    // Double dogleg extra case: t·‖dx_gn‖ ≤ delta → scale full GN to boundary
    if (t * norm_gn <= delta) {
      return (delta / norm_gn) * dx_gn;
    }
  }

  // Interpolate: dx_sd + beta*(t·dx_gn − dx_sd) to land on boundary
  const ParamVec d = t * dx_gn - dx_sd;
  const value_type a_q = d.squaredNorm();
  const value_type b_q = value_type{2} * dx_sd.dot(d);
  const value_type c_q = dx_sd.squaredNorm() - delta * delta;
  value_type disc = b_q * b_q - value_type{4} * a_q * c_q;
  if (disc < value_type{0}) {
    disc = value_type{0};
  }
  const value_type beta = (-b_q + sqrt(disc)) / (value_type{2} * a_q);
  return dx_sd + beta * d;
}

template <diff::CExpression... RExprs, DoglegVariant DV>
constexpr typename NLSDogleg<diff::Equation<RExprs...>, DV>::ParamVec
NLSDogleg<diff::Equation<RExprs...>, DV>::minimize(ParamVec p) {
  using std::abs, std::max;

  auto [r, J] = eval_rJ(p);
  NMat B = J.transpose() * J;
  ParamVec g = J.transpose() * r;
  value_type phi = value_type{0.5} * r.squaredNorm();
  value_type delta = trustregion0;

  for (iter = 0; iter < itmax; ++iter) {
    const value_type den = max(abs(phi), value_type{1});
    const value_type gnorm =
        (g.cwiseAbs().array() * p.cwiseAbs().cwiseMax(value_type{1}).array())
            .maxCoeff();
    if (gnorm / den < tol) {
      break;
    }

    const ParamVec step = compute_step(g, B, J, delta);
    const bool at_boundary = step.norm() >= value_type{0.999} * delta;

    const value_type pred =
        -(g.dot(step) + value_type{0.5} * step.dot(B * step));

    const ParamVec p_new = p + step;
    auto [r_new, J_new] = eval_rJ(p_new);
    const value_type phi_new = value_type{0.5} * r_new.squaredNorm();

    const value_type rho =
        (pred > value_type{0}) ? (phi - phi_new) / pred : value_type{0};

    if (rho < TR_DOWN_THRESHOLD) {
      delta *= TR_DOWN_FACTOR;
    } else if (rho > TR_UP_THRESHOLD && at_boundary) {
      delta *= TR_UP_FACTOR;
    }
    if (delta < trustregion_min)
      break;

    if (rho > value_type{0}) {
      if (step.cwiseAbs().maxCoeff() < tol) {
        break;
      }
      p = p_new;
      r = std::move(r_new);
      J = std::move(J_new);
      B = J.transpose() * J;
      g = J.transpose() * r;
      phi = phi_new;
    }
  }

  fret = phi;
  return p;
}

template <diff::CExpression... RExprs>
NLSDogleg(diff::Equation<RExprs...>) -> NLSDogleg<diff::Equation<RExprs...>>;

template <diff::CExpression... RExprs, typename T>
NLSDogleg(diff::Equation<RExprs...>, T) -> NLSDogleg<diff::Equation<RExprs...>>;

} // namespace exprmin
