#pragma once

#include "equation.hpp"
#include "gradient.hpp"
#include "../callback/callback.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>

namespace exprmin {

/**
 * @brief CRTP base implementing the outer trust-region loop
 *        (Nocedal & Wright §4.1, Algorithm 4.1).
 *
 * Each iteration minimises the quadratic model
 *
 *   m(p) = f + gᵀp + ½pᵀBp   s.t. ‖p‖ ≤ Δ                  (N&W eq. 4.3)
 *
 * via Derived::compute_step, then computes the actual-to-predicted reduction
 * ratio
 *
 *   ρ = (f(x) − f(x+p)) / (m(0) − m(p))                      (N&W eq. 4.4)
 *
 * to decide whether to accept the step and how to update Δ.
 *
 * Derived must supply (or inherit from NLSTrustRegionBase):
 *   eval_state(p)                  → tuple<T, ParamVec, NMat>  [f, g, B]
 *   eval_trial(p+step)             → T   (caches new state for commit_state)
 *   compute_step(g, B, Δ)          → ParamVec
 *   adjust_tr(Δ&, ρ, at_boundary)  → void
 *   commit_state(step, g_old, B)   → pair<ParamVec, NMat>  [g_new, B_new]
 *   refresh_hessian(p, B&)         → void  [default: no-op]
 *
 * @tparam Derived  CRTP subclass.
 * @tparam T        Floating-point scalar type.
 * @tparam N        Number of parameters (compile-time).
 */
template <typename Derived, typename T, int N> struct TrustRegionBase {
  using value_type = T;
  using ParamVec = Eigen::Vector<T, N>;
  using NMat = Eigen::Matrix<T, N, N>;

  int iter{};
  constexpr T get_optimal_value() const { return fret; }

  // Default CRTP hook — overridden by derived classes that hold a Callbacks member.
  constexpr void on_tr_iter(int, T, T, T, T, bool) const noexcept {}

  /**
   * @brief Execute the trust-region outer loop (N&W Alg. 4.1) from x₀ = @p p.
   *
   * @param p  Starting point x₀ ∈ ℝᴺ.
   * @return   Approximate minimiser xₖ after at most @c itmax iterations.
   *
   * Convergence criterion: scaled ∞-norm of the gradient
   *   max_i |gᵢ| · max(|xᵢ|, 1) / max(|f|, 1) < tol.
   * The loop also terminates early if Δ drops below @c trustregion_min.
   */
  constexpr ParamVec minimize(ParamVec p);

protected:
  static constexpr T TR_DOWN_FACTOR = T{0.1};
  static constexpr T TR_DOWN_THRESHOLD = T{0.25};
  static constexpr T TR_UP_FACTOR = T{2.0};
  static constexpr T TR_UP_THRESHOLD = T{0.75};

  T tol;
  int itmax;
  T trustregion0;
  T trustregion_min;
  T fret{};

  constexpr explicit TrustRegionBase(T tol_, int itmax_, T tr0_, T trmin_)
      : tol{tol_}, itmax{itmax_}, trustregion0{tr0_}, trustregion_min{trmin_} {}
  constexpr void refresh_hessian(const ParamVec &, NMat & /*B*/) {}

private:
  constexpr Derived &self() { return static_cast<Derived &>(*this); }
  constexpr const Derived &self() const {
    return static_cast<const Derived &>(*this);
  }
};
template <typename Derived, typename T, int N>
constexpr typename TrustRegionBase<Derived, T, N>::ParamVec
TrustRegionBase<Derived, T, N>::minimize(ParamVec p) {
  using std::abs, std::max;

  // Step 1: evaluate f(p₀), gradient g, and initial Hessian approximation B.
  auto [phi, g, B] = self().eval_state(p);
  T delta = trustregion0; // initial trust-region radius Δ₀

  for (iter = 0; iter < itmax; ++iter) {
    // Step 2: convergence check — scaled ∞-norm of g (N&W §4.1 stopping test).
    // den = max(|f|, 1) avoids division by zero when f ≈ 0.
    const T den = max(abs(phi), T{1});
    const T gnorm =
        (g.cwiseAbs().array() * p.cwiseAbs().cwiseMax(T{1}).array()).maxCoeff();
    if (gnorm / den < tol) {
      break;
    }

    // Step 3: optionally refresh B (exact AD path); no-op for BFGS/GN paths.
    self().refresh_hessian(p, B);

    // Step 4: solve the TR subproblem min m(p) s.t. ‖p‖ ≤ Δ (N&W eq. 4.3).
    const ParamVec step = self().compute_step(g, B, delta);
    // at_boundary: ‖p‖ ≈ Δ — needed by adjust_tr to gate radius expansion.
    const bool at_boundary = step.norm() >= T{0.999} * delta;

    // Step 5: predicted reduction from the quadratic model: pred = −(gᵀp +
    // ½pᵀBp).
    const T pred = -(g.dot(step) + T{0.5} * step.dot(B * step));

    // Step 6: evaluate f at the trial point; compute ρ = actual/predicted
    // (eq. 4.4).
    const T phi_new = self().eval_trial(p + step);
    const T rho = (pred > T{0}) ? (phi - phi_new) / pred : T{0};

    // Step 7: adapt Δ based on ρ (Alg. 4.1 policy).
    self().on_tr_iter(iter, phi, gnorm, delta, rho, rho > T{0});
    self().adjust_tr(delta, rho, at_boundary);
    if (delta < trustregion_min) {
      break; // radius collapsed to machine noise; further progress impossible
    }

    // Step 8: accept step when ρ > 0; update p, f, g, B.
    if (rho > T{0}) {
      if (step.cwiseAbs().maxCoeff() < tol) {
        break; // step negligibly small despite ρ > 0
      }
      p += step;
      phi = phi_new;
      std::tie(g, B) = self().commit_state(step, g, B);
    }
    // else: step rejected; Δ was already shrunk by adjust_tr — retry.
  }
  fret = phi;
  return p;
}

/**
 * @brief Trust-region base for nonlinear least squares over a diff::Equation
 * system.
 *
 * Specialises TrustRegionBase for systems of M ≥ N residual expressions.
 * Implements all eval/commit hooks using the Gauss-Newton approximation
 *
 *   B ≈ JᵀJ,   g = Jᵀr,   φ = ½‖r‖²                        (N&W §10.2)
 *
 * so second derivatives are never required.  Derived supplies only
 * compute_step() and may inspect the current Jacobian via current_J().
 *
 * @tparam Derived  CRTP subclass (e.g. NLSDogleg).
 * @tparam RExprs   Pack of residual expressions forming the diff::Equation.
 */
template <typename Derived, diff::CExpression... RExprs>
struct NLSTrustRegionBase
    : TrustRegionBase<Derived, typename diff::Equation<RExprs...>::value_type,
                      static_cast<int>(diff::Equation<RExprs...>::input_dim)> {

  using Sys = diff::Equation<RExprs...>;
  using value_type = typename Sys::value_type;
  using Syms = typename Sys::symbols;
  static constexpr int M = static_cast<int>(Sys::output_dim);
  static constexpr int N = static_cast<int>(Sys::input_dim);
  using Base = TrustRegionBase<Derived, value_type, N>;
  using ParamVec = typename Base::ParamVec;
  using NMat = typename Base::NMat;
  using RVec = Eigen::Vector<value_type, M>;
  using JMat = Eigen::Matrix<value_type, M, N>;

  static_assert(Sys::input_dim <= Sys::output_dim,
                "NLS trust-region requires M >= N residuals");

protected:
  Sys system;

  constexpr auto to_arr(const ParamVec &p) const noexcept {
    std::array<value_type, static_cast<std::size_t>(N)> arr;
    std::copy_n(p.data(), N, arr.begin());
    return arr;
  }

  /**
   * @brief Evaluate all M residuals and the M×N Jacobian at @p p.
   *
   * Calls system.update then reads r and J from the AD backend.
   * Caches nothing; caller decides what to store.
   */
  constexpr std::pair<RVec, JMat> eval_rJ(const ParamVec &p);
  constexpr const JMat &current_J() const { return J_; }

public:
  /**
   * @brief Seed the TR loop: returns φ = ½‖r‖², g = Jᵀr, B = JᵀJ at @p p
   *        and caches r, J for the first iteration (N&W §10.2).
   */
  constexpr std::tuple<value_type, ParamVec, NMat>
  eval_state(const ParamVec &p);

  /**
   * @brief Evaluate φ_new = ½‖r_new‖² at the trial point without committing.
   *
   * Caches r_new and J_new so commit_state can promote them if the step is
   * accepted.
   */
  constexpr value_type eval_trial(const ParamVec &p_new);

  /**
   * @brief Standard trust-region radius update (N&W Alg. 4.1 policy).
   *
   * Shrinks Δ by TR_DOWN_FACTOR when ρ < ¼; expands by TR_UP_FACTOR when
   * ρ > ¾ and the step hit the boundary.
   */
  constexpr void adjust_tr(value_type &delta, value_type rho, bool at_boundary);

  /**
   * @brief Promote cached trial state to current state after step acceptance.
   *
   * Moves r_new → r and J_new → J, then returns the new gradient g = Jᵀr
   * and Gauss-Newton Hessian B = JᵀJ.  @p step and @p g_old are unused
   * because the NLS update is entirely driven by the cached Jacobian.
   */
  constexpr std::pair<ParamVec, NMat> commit_state(const ParamVec & /*step*/,
                                                   const ParamVec & /*g_old*/,
                                                   const NMat & /*B_cur*/);

  constexpr explicit NLSTrustRegionBase(Sys sys,
                                        value_type tol_ = value_type{1e-8},
                                        int itmax_ = 200,
                                        value_type tr0_ = value_type{1e3},
                                        value_type trmin_ = value_type{1e-12})
      : Base{tol_, itmax_, tr0_, trmin_}, system{std::move(sys)} {}

private:
  RVec r_{}, r_new_{};
  JMat J_{}, J_new_{};
};

template <typename Derived, diff::CExpression... RExprs>
constexpr std::pair<typename NLSTrustRegionBase<Derived, RExprs...>::RVec,
                    typename NLSTrustRegionBase<Derived, RExprs...>::JMat>
NLSTrustRegionBase<Derived, RExprs...>::eval_rJ(const ParamVec &p) {
  // Push p into the expression system so evaluate() and jacobian() are
  // consistent.
  system.update(Syms{}, to_arr(p));

  // Read residual vector r (M×1).
  RVec r;
  if constexpr (M == 1) {
    r[0] = system.evaluate();
  } else {
    const auto r_arr = system.evaluate();
    r = Eigen::Map<const RVec>(r_arr.data());
  }

  // Read Jacobian J (M×N) from reverse-mode AD; stored row-major by the
  // backend.
  const auto J_arr = system.template jacobian<diff::DiffMode::Reverse>();
  JMat J = Eigen::Map<const Eigen::Matrix<value_type, M, N, Eigen::RowMajor>>(
      &J_arr[0][0]);

  return {r, J};
}

template <typename Derived, diff::CExpression... RExprs>
constexpr std::tuple<
    typename NLSTrustRegionBase<Derived, RExprs...>::value_type,
    typename NLSTrustRegionBase<Derived, RExprs...>::ParamVec,
    typename NLSTrustRegionBase<Derived, RExprs...>::NMat>
NLSTrustRegionBase<Derived, RExprs...>::eval_state(const ParamVec &p) {
  // Evaluate r and J; cache both for use by the first eval_trial/commit_state.
  auto [r, J] = eval_rJ(p);
  r_ = std::move(r);
  J_ = std::move(J);
  // Return φ = ½‖r‖², g = Jᵀr, B = JᵀJ (Gauss-Newton Hessian approx., N&W
  // §10.2).
  return {value_type{0.5} * r_.squaredNorm(), J_.transpose() * r_,
          J_.transpose() * J_};
}

template <typename Derived, diff::CExpression... RExprs>
constexpr typename NLSTrustRegionBase<Derived, RExprs...>::value_type
NLSTrustRegionBase<Derived, RExprs...>::eval_trial(const ParamVec &p_new) {
  // Evaluate at the trial point and stash (r_new_, J_new_) for commit_state.
  auto [r, J] = eval_rJ(p_new);
  r_new_ = std::move(r);
  J_new_ = std::move(J);
  return value_type{0.5} * r_new_.squaredNorm(); // φ_new = ½‖r_new‖²
}

template <typename Derived, diff::CExpression... RExprs>
constexpr void NLSTrustRegionBase<Derived, RExprs...>::adjust_tr(
    value_type &delta, value_type rho, bool at_boundary) {
  if (rho < Base::TR_DOWN_THRESHOLD) {
    delta *= Base::TR_DOWN_FACTOR; // ρ < ¼: poor agreement — shrink Δ
  } else if (rho > Base::TR_UP_THRESHOLD && at_boundary) {
    delta *= Base::TR_UP_FACTOR; // ρ > ¾ and step hit boundary — expand Δ
  }
  // ¼ ≤ ρ ≤ ¾: acceptable agreement — keep Δ unchanged.
}
template <typename Derived, diff::CExpression... RExprs>
constexpr std::pair<typename NLSTrustRegionBase<Derived, RExprs...>::ParamVec,
                    typename NLSTrustRegionBase<Derived, RExprs...>::NMat>
NLSTrustRegionBase<Derived, RExprs...>::commit_state(const ParamVec &,
                                                     const ParamVec &,
                                                     const NMat &) {
  // Promote trial state: r_new_ → r_, J_new_ → J_.
  r_ = std::move(r_new_);
  J_ = std::move(J_new_);
  // Return updated gradient g = Jᵀr and Gauss-Newton Hessian B = JᵀJ.
  return {J_.transpose() * r_, J_.transpose() * J_};
}

} // namespace exprmin
