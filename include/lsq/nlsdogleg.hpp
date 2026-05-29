#pragma once

#include "../minimizer/trustregionbase.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace exprmin {

/**
 * @brief Selects the dogleg step geometry used by NLSDogleg.
 *
 * - @c Standard — classic two-segment path: Cauchy → Gauss-Newton
 *                 (Powell 1970; N&W §10.1).
 * - @c Double   — Byrd–Schnabel–Shultz scaled variant: scales the GN leg
 *                 by a damping factor t ∈ [0.2, 1] derived from a Powell
 *                 condition, softening aggressive GN steps on ill-scaled
 * problems.
 */
enum class DoglegVariant { Standard, Double };

/**
 * @brief Powell dogleg trust-region solver for nonlinear least squares
 *        (Nocedal & Wright §10.1; Dennis & Schnabel §10.3).
 *
 * Minimises φ(θ) = ½ ‖R(θ)‖² where R : ℝᴺ → ℝᴹ (M ≥ N) is supplied as a
 * diff::Equation<R₁,…,Rₘ>.  The outer trust-region loop and all NLS state
 * management (residuals, Jacobian, Gauss-Newton B = JᵀJ) are inherited from
 * NLSTrustRegionBase.  This class only provides the dogleg step geometry.
 *
 * Two variants are available via @p DV:
 *   - Standard — classic two-segment path: steepest descent → Gauss-Newton
 *                (Powell 1970; N&W §10.1 eq. 10.13).
 *   - Double   — Byrd–Schnabel–Shultz scaled variant: scales the Gauss-Newton
 *                leg by a damping factor t ∈ [0.2, 1] derived from a Powell
 *                condition, softening aggressive GN steps on badly scaled
 * problems.
 *
 * @tparam System  diff::Equation<RExprs...> specialisation.
 * @tparam DV      DoglegVariant::Standard or DoglegVariant::Double.
 */
template <typename System, DoglegVariant DV = DoglegVariant::Standard,
          typename Callbacks = callback::NoCallbacks>
struct NLSDogleg;

template <diff::CExpression... RExprs, DoglegVariant DV, typename Callbacks>
struct NLSDogleg<diff::Equation<RExprs...>, DV, Callbacks>
    : NLSTrustRegionBase<NLSDogleg<diff::Equation<RExprs...>, DV, Callbacks>, RExprs...> {
  using Base =
      NLSTrustRegionBase<NLSDogleg<diff::Equation<RExprs...>, DV, Callbacks>, RExprs...>;
  using value_type = typename Base::value_type;
  using ParamVec = typename Base::ParamVec;
  using NMat = typename Base::NMat;
  using RVec = typename Base::RVec;
  using JMat = typename Base::JMat;

  using Base::get_optimal_value;
  using Base::iter;

  constexpr explicit NLSDogleg(typename Base::Sys sys,
                                value_type tol_ = value_type{1e-8},
                                int itmax_ = 200,
                                value_type tr0_ = value_type{1e3},
                                value_type trmin_ = value_type{1e-12},
                                Callbacks cbs = {})
      : Base{std::move(sys), tol_, itmax_, tr0_, trmin_}, cbs_(cbs) {}

  constexpr void on_tr_iter(int itr, value_type phi, value_type gnorm,
                            value_type delta, value_type rho, bool accepted) noexcept {
    cbs_.on_tr_iter(itr, phi, gnorm, delta, rho, accepted);
  }

  /**
   * @brief Compute the dogleg step for the NLS trust-region subproblem.
   *
   * Approximates min m(p) = φ + gᵀp + ½pᵀBp s.t. ‖p‖ ≤ Δ along the
   * two-segment (Standard) or scaled two-segment (Double) dogleg path:
   *
   *   Standard:  p^SD → p^GN               (N&W §10.1, eq. 10.13)
   *   Double:    p^SD → t·p^GN,  t ∈ [0.2,1]  (Byrd–Schnabel–Shultz scaling)
   *
   * Here p^SD = −α g is the Cauchy (steepest-descent) step with α chosen to
   * minimise m along −g, and p^GN = −B⁻¹g is the Gauss-Newton step.
   * g = Jᵀr and B = JᵀJ are passed in from NLSTrustRegionBase.
   *
   * @param g      Gradient Jᵀr at the current point.
   * @param B      Gauss-Newton Hessian approximation JᵀJ.
   * @param delta  Current trust-region radius Δ.
   * @return       The dogleg step p ∈ ℝᴺ with ‖p‖ ≤ Δ.
   */
  constexpr ParamVec compute_step(const ParamVec &g, const NMat &B,
                                  value_type delta) const;

private:
  [[no_unique_address]] Callbacks cbs_{};
};

template <diff::CExpression... RExprs, DoglegVariant DV, typename Callbacks>
constexpr typename NLSDogleg<diff::Equation<RExprs...>, DV, Callbacks>::ParamVec
NLSDogleg<diff::Equation<RExprs...>, DV, Callbacks>::compute_step(const ParamVec &g,
                                                                   const NMat &B,
                                                                   value_type delta) const {
  using std::abs, std::min, std::sqrt;
  const JMat &J = this->current_J();

  // Step 1: Cauchy (steepest-descent) step p^SD = −α g, where
  //   α = ‖g‖² / ‖Jg‖²  minimises m along −g  (N&W §10.1, eq. 10.8).
  // Falls back to α = 1 when ‖Jg‖ = 0 (g in the null space of J).
  const RVec Jg = J * g;
  const auto Jg_sq = Jg.squaredNorm();
  const auto alpha =
      (Jg_sq > value_type{0}) ? g.squaredNorm() / Jg_sq : value_type{1};
  const ParamVec dx_sd = -alpha * g;

  // Step 2: Gauss-Newton step p^GN = −(JᵀJ)⁻¹ g = −B⁻¹g (eq. 10.10).
  const ParamVec dx_gn = -(B.ldlt().solve(g));
  const auto norm_sd = dx_sd.norm();
  const auto norm_gn = dx_gn.norm();

  // Step 3: Case 1 — Cauchy step already hits or exceeds the boundary: clip it.
  if (norm_sd >= delta) {
    return (delta / norm_sd) * dx_sd;
  }

  // Step 4: Case 2 — Gauss-Newton step fits inside the region: return it
  // directly.
  if (norm_gn <= delta) {
    return dx_gn;
  }

  // Step 5 (Double variant only): compute Byrd–Schnabel–Shultz damping factor
  // t. c measures how well the GN direction is aligned with −g via a Powell
  // condition; t ∈ [0.2, 1] scales p^GN to avoid over-aggressive steps on
  // ill-scaled problems.
  auto t = value_type{1};
  if constexpr (DV == DoglegVariant::Double) {
    const auto gBg = g.dot(B * g);
    const auto gdx = abs(g.dot(dx_gn));
    if (gBg > value_type{0} && gdx > value_type{0}) {
      const auto g2 = g.squaredNorm();
      const auto c = min(value_type{1}, (g2 * g2) / (gBg * gdx));
      t = value_type{1} - value_type{0.8} * (value_type{1} - c); // t ∈ [0.2, 1]
    }
    // If the scaled GN step still fits, return it clipped to the boundary.
    if (t * norm_gn <= delta) {
      return (delta / norm_gn) * dx_gn;
    }
  }

  // Step 6: Case 3 — interpolate on the segment [p^SD, t·p^GN] to hit ‖p‖ = Δ.
  // Solve ‖dx_sd + β·d‖² = Δ² with d = t·dx_gn − dx_sd (quadratic in β).
  const auto d = t * dx_gn - dx_sd;
  const auto a_q = d.squaredNorm();
  const auto b_q = value_type{2} * dx_sd.dot(d);
  const auto c_q = dx_sd.squaredNorm() - delta * delta;
  auto disc = b_q * b_q - value_type{4} * a_q * c_q;
  if (disc < value_type{0}) {
    disc = value_type{0}; // guard against floating-point noise
  }
  const auto beta = (-b_q + sqrt(disc)) / (value_type{2} * a_q);
  return dx_sd + beta * d;
}

template <diff::CExpression... RExprs>
NLSDogleg(diff::Equation<RExprs...>) -> NLSDogleg<diff::Equation<RExprs...>>;

template <diff::CExpression... RExprs, typename T>
NLSDogleg(diff::Equation<RExprs...>, T) -> NLSDogleg<diff::Equation<RExprs...>>;

template <diff::CExpression... RExprs, typename T>
NLSDogleg(diff::Equation<RExprs...>, T, int) -> NLSDogleg<diff::Equation<RExprs...>>;

} // namespace exprmin
