#pragma once

#include "equation.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>

namespace exprmin {

// ── Level 1: generic trust-region loop ───────────────────────────────────────
//
// CRTP base owning the minimize() loop, TR constants, and shared state.
// Derived must implement (or inherit from NLSTrustRegionBase which provides them):
//
//   eval_state(p)                      → tuple<T, ParamVec, NMat>  [phi, g, B]
//   eval_trial(p_new)                  → T   (caches new state internally in Derived)
//   compute_step(g, B, delta)          → ParamVec
//   adjust_tr(delta&, rho, at_bnd)     → void
//   commit_state(step, g_old, B_cur)   → pair<ParamVec, NMat>  [g_new, B_new]
//   refresh_hessian(p, B&)             → void  [default: no-op; Dogleg::ExactAD overrides]
//
template <typename Derived, typename T, int N>
struct TrustRegionBase {
  using value_type = T;
  using ParamVec   = Eigen::Vector<T, N>;
  using NMat       = Eigen::Matrix<T, N, N>;

  int iter{};

  constexpr T get_optimal_value() const { return fret; }

  constexpr ParamVec minimize(ParamVec p) {
    using std::abs, std::max;

    auto [phi, g, B] = self().eval_state(p);
    T delta = trustregion0;

    for (iter = 0; iter < itmax; ++iter) {
      const T den   = max(abs(phi), T{1});
      const T gnorm = (g.cwiseAbs().array() *
                       p.cwiseAbs().cwiseMax(T{1}).array())
                          .maxCoeff();
      if (gnorm / den < tol) break;

      self().refresh_hessian(p, B);

      const ParamVec step       = self().compute_step(g, B, delta);
      const bool     at_boundary = step.norm() >= T{0.999} * delta;
      const T        pred        = -(g.dot(step) + T{0.5} * step.dot(B * step));

      const T phi_new = self().eval_trial(p + step);
      const T rho     = (pred > T{0}) ? (phi - phi_new) / pred : T{0};

      self().adjust_tr(delta, rho, at_boundary);
      if (delta < trustregion_min) break;

      if (rho > T{0}) {
        if (step.cwiseAbs().maxCoeff() < tol) break;
        p   += step;
        phi  = phi_new;
        std::tie(g, B) = self().commit_state(step, g, B);
      }
    }
    fret = phi;
    return p;
  }

protected:
  static constexpr T TR_DOWN_FACTOR    = T{0.1};
  static constexpr T TR_DOWN_THRESHOLD = T{0.25};
  static constexpr T TR_UP_FACTOR      = T{2.0};
  static constexpr T TR_UP_THRESHOLD   = T{0.75};

  T   tol;
  int itmax;
  T   trustregion0;
  T   trustregion_min;
  T   fret{};

  constexpr explicit TrustRegionBase(T tol_, int itmax_, T tr0_, T trmin_)
      : tol{tol_}, itmax{itmax_}, trustregion0{tr0_}, trustregion_min{trmin_} {}

  // Default no-op; Dogleg<Expr, HessianMode::ExactAD> overrides this.
  constexpr void refresh_hessian(const ParamVec& /*p*/, NMat& /*B*/) {}

private:
  constexpr Derived&       self()       { return static_cast<Derived&>(*this); }
  constexpr const Derived& self() const { return static_cast<const Derived&>(*this); }
};


// ── Level 2: NLS-specific shared layer (diff::Equation) ──────────────────────
//
// Inherits TrustRegionBase and implements the four eval/update hooks for any
// diff::Equation<RExprs...> system with M >= N residuals.
// Derived still provides compute_step; it may read the cached Jacobian via
// current_J().
//
template <typename Derived, diff::CExpression... RExprs>
struct NLSTrustRegionBase
    : TrustRegionBase<Derived,
                      typename diff::Equation<RExprs...>::value_type,
                      static_cast<int>(diff::Equation<RExprs...>::input_dim)> {

  using Sys        = diff::Equation<RExprs...>;
  using value_type = typename Sys::value_type;
  using Syms       = typename Sys::symbols;
  static constexpr int M = static_cast<int>(Sys::output_dim);
  static constexpr int N = static_cast<int>(Sys::input_dim);
  using Base     = TrustRegionBase<Derived, value_type, N>;
  using ParamVec = typename Base::ParamVec;
  using NMat     = typename Base::NMat;
  using RVec     = Eigen::Vector<value_type, M>;
  using JMat     = Eigen::Matrix<value_type, M, N>;

  static_assert(Sys::input_dim <= Sys::output_dim,
                "NLS trust-region requires M >= N residuals");

protected:
  Sys system;

  constexpr auto to_arr(const ParamVec& p) const noexcept {
    std::array<value_type, static_cast<std::size_t>(N)> arr;
    std::copy_n(p.data(), N, arr.begin());
    return arr;
  }

  constexpr std::pair<RVec, JMat> eval_rJ(const ParamVec& p) {
    system.update(Syms{}, to_arr(p));
    RVec r;
    if constexpr (M == 1) {
      r[0] = system.evaluate();
    } else {
      const auto r_arr = system.evaluate();
      for (int i = 0; i < M; ++i)
        r[i] = r_arr[static_cast<std::size_t>(i)];
    }
    const auto J_arr = system.template jacobian<diff::DiffMode::Reverse>();
    JMat J;
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        J(i, j) = J_arr[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return {r, J};
  }

  // Derived compute_step implementations can read the current Jacobian here.
  constexpr const JMat& current_J() const { return J_; }

public:
  // ── CRTP hook implementations ─────────────────────────────────────────────

  constexpr std::tuple<value_type, ParamVec, NMat> eval_state(const ParamVec& p) {
    auto [r, J] = eval_rJ(p);
    r_ = std::move(r);
    J_ = std::move(J);
    return {value_type{0.5} * r_.squaredNorm(),
            J_.transpose() * r_,
            J_.transpose() * J_};
  }

  constexpr value_type eval_trial(const ParamVec& p_new) {
    auto [r, J] = eval_rJ(p_new);
    r_new_ = std::move(r);
    J_new_ = std::move(J);
    return value_type{0.5} * r_new_.squaredNorm();
  }

  constexpr void adjust_tr(value_type& delta, value_type rho, bool at_boundary) {
    if (rho < Base::TR_DOWN_THRESHOLD)
      delta *= Base::TR_DOWN_FACTOR;
    else if (rho > Base::TR_UP_THRESHOLD && at_boundary)
      delta *= Base::TR_UP_FACTOR;
  }

  // step and g_old are unused; the NLS update comes from cached r_new_, J_new_.
  constexpr std::pair<ParamVec, NMat>
  commit_state(const ParamVec& /*step*/, const ParamVec& /*g_old*/,
               const NMat& /*B_cur*/) {
    r_ = std::move(r_new_);
    J_ = std::move(J_new_);
    return {J_.transpose() * r_, J_.transpose() * J_};
  }

  constexpr explicit NLSTrustRegionBase(Sys          sys,
                                         value_type   tol_   = value_type{1e-8},
                                         int          itmax_ = 200,
                                         value_type   tr0_   = value_type{1e3},
                                         value_type   trmin_ = value_type{1e-12})
      : Base{tol_, itmax_, tr0_, trmin_}, system{std::move(sys)} {}

private:
  RVec r_{}, r_new_{};
  JMat J_{}, J_new_{};
};

} // namespace exprmin
