#pragma once

#include "trustregionbase.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>

namespace exprmin {

namespace mp = boost::mp11;

enum class HessianMode { BFGS, ExactAD };

/**
 * @brief Powell dogleg trust-region minimizer (Nocedal & Wright §4.1).
 *
 * Minimizes a scalar f(x) by iterating the trust-region loop (Algorithm 4.1)
 * whose inner subproblem is solved by the dogleg approximation.  Each
 * iteration minimizes the quadratic model
 *
 *   m(p) = f + gᵀp + ½pᵀBp   s.t. ‖p‖ ≤ Δ                (eq. 4.5)
 *
 * along the two-segment path p̃(τ) (eq. 4.16): first along the steepest-
 * descent direction toward the Cauchy point p^U, then from p^U toward the
 * Newton (full) step p^B = −B⁻¹g.  The outer loop adapts Δ from the ratio
 *
 *   ρ = (f(x) − f(x+p)) / (m(0) − m(p))                   (eq. 4.4)
 *
 * and accepts the step only when ρ > 0.
 *
 * @tparam Expr  Expression satisfying diff::CExpression; defines f and ∇f.
 * @tparam HM    Hessian approximation strategy:
 *               - BFGS    — rank-2 symmetric update applied to B directly
 *                           (eq. 6.17); cheaper but inexact.
 *               - ExactAD — exact Hessian recomputed every iteration via
 *                           diff::derivative_tensor<2>; required when B
 *                           must be exact (e.g. near the solution).
 */
template <diff::CExpression Expr, HessianMode HM = HessianMode::BFGS>
struct Dogleg
    : TrustRegionBase<
          Dogleg<Expr, HM>, typename Expr::value_type,
          static_cast<int>(
              mp::mp_size<diff::extract_symbols_from_expr_t<Expr>>::value)> {

  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr int N = static_cast<int>(mp::mp_size<Syms>::value);
  using Base = TrustRegionBase<Dogleg<Expr, HM>, value_type, N>;
  using Point = Eigen::Vector<value_type, N>;
  using Matrix = Eigen::Matrix<value_type, N, N>;

  using ParamVec = Point;
  using NMat = Matrix;

  using Base::get_optimal_value;
  using Base::iter;

  constexpr value_type operator()(const Point &p) { return eval_at(p); }
  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(expr);
    return {expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  constexpr explicit Dogleg(Expr e, value_type tol_ = value_type{1e-8},
                            int itmax_ = 200,
                            value_type trustregion0_ = value_type{1e3},
                            value_type trustregion_min_ = value_type{1e-12})
      : Base{tol_, itmax_, trustregion0_, trustregion_min_},
        expr{std::move(e)} {}

  // B₀ = I so the first dogleg step is a scaled Cauchy step; BFGS or
  // ExactAD will populate B after the first successful commit/refresh.
  constexpr std::tuple<value_type, Point, Matrix> eval_state(const Point &p) {
    auto [f, g] = eval_grad(p);
    return {f, g, Matrix::Identity()};
  }

  constexpr value_type eval_trial(const Point &p_new) {
    auto [f, g] = eval_grad(p_new);
    g_new_ = std::move(g);
    return f;
  }

  // ExactAD path: B = ∇²f(xₖ) recomputed exactly at the start of each outer
  // iteration.  BFGS path is a no-op (base-class default handles it).
  constexpr void refresh_hessian(const Point &p, Matrix &B) {
    if constexpr (HM == HessianMode::ExactAD) {
      expr.update(Syms{}, p);
      const auto H = diff::derivative_tensor<2>(expr);
      B = Eigen::Map<const Eigen::Matrix<value_type, N, N, Eigen::RowMajor>>(
          &H[0][0]);
    }
  }

  /**
   * @brief Three-way dogleg step (Nocedal §4.1, eqs. 4.11–4.16).
   *
   * Approximates the solution of min m(p) s.t. ‖p‖ ≤ Δ by restricting the
   * search to the two-segment dogleg path p̃(τ) (eq. 4.16):
   *
   *   p̃(τ) = τ p^U,                       0 ≤ τ ≤ 1
   *           p^U + (τ−1)(p^B − p^U),      1 ≤ τ ≤ 2
   *
   * where p^U is the unconstrained minimizer of m along −g (Cauchy step,
   * eq. 4.15) and p^B = −B⁻¹g is the full Newton step.  By Lemma 4.2, ‖p̃‖
   * is non-decreasing and m(p̃) is non-increasing in τ, so the minimum on the
   * path subject to ‖p‖ ≤ Δ is taken as far along the path as Δ allows:
   *
   *   Case 1: ‖p^B‖ ≤ Δ  →  return p^B (full Newton step inside region).
   *   Case 2: ‖p^U‖ ≥ Δ  →  return (Δ/‖p^U‖) p^U (scaled Cauchy step).
   *   Case 3: otherwise  →  solve ‖p^U + τ(p^B − p^U)‖² = Δ² for τ and
   *                          return the interpolated point.
   *
   * When B is indefinite (gᵀBg ≤ 0) the Cauchy scaling κ defaults to −1 so
   * that p^U = −g; case 2 then clips it to the trust-region boundary.
   *
   * nn_ = ‖p^B‖ is cached here so that adjust_tr can apply the libdogleg
   * Newton-size floor before shrinking Δ.
   */
  constexpr Point compute_step(const Point &g, const Matrix &B,
                               value_type delta) {
    using std::sqrt;

    // Cauchy step: p^U = −(gᵀg / gᵀBg) g  (eq. 4.15).
    // κ = −‖g‖²/(gᵀBg); falls back to −1 when B is indefinite.
    const auto gBg = g.dot(B * g);
    const auto kc =
        (gBg > value_type{0}) ? (-g.squaredNorm() / gBg) : value_type{-1};
    const Point p_c = kc * g;
    const auto nc = p_c.norm();

    // Full Newton step p^B = −B⁻¹g (eq. 4.3 unconstrained minimizer).
    const Point p_n = -(B.ldlt().solve(g));
    nn_ = p_n.norm();

    if (nn_ <= delta) {
      return p_n;                          // case 1: Newton inside region
    } else if (nc >= delta) {
      return (delta / nc) * p_c;           // case 2: clip Cauchy to boundary
    } else {
      // Case 3: solve ‖p^U + τ(p^B − p^U)‖² = Δ² for τ ∈ [0,1].
      // Expanding and collecting: l2·τ² + 2c·τ + (nc²−Δ²) = 0.
      const Point d = p_n - p_c;
      const auto l2 = d.squaredNorm();
      const auto c = d.dot(p_c);
      value_type disc = c * c - l2 * (nc * nc - delta * delta);
      if (disc < value_type{0}) {
        disc = value_type{0};              // guard against floating-point noise
      }
      return p_c + ((-c + sqrt(disc)) / l2) * d;
    }
  }

  /**
   * @brief Adapts the trust-region radius (Alg. 4.1 policy, libdogleg variant).
   *
   * Standard Alg. 4.1 shrinks Δ by TR_DOWN_FACTOR whenever ρ < ¼.  The
   * libdogleg variant first collapses Δ to ‖p^B‖ (the Newton-step length,
   * cached in nn_) when the accepted step was strictly interior (at_boundary
   * false), then applies the factor.  This avoids over-aggressive shrinking
   * when Δ is already larger than the Newton step needs.
   *
   * Expansion by TR_UP_FACTOR occurs only when ρ > ¾ and the step hit the
   * boundary, matching the condition in Alg. 4.1.
   */
  constexpr void adjust_tr(value_type &delta, value_type rho, bool at_boundary);

  /**
   * @brief Updates gradient and Hessian approximation after an accepted step.
   *
   * BFGS mode applies the symmetric rank-2 update to B (the Hessian approx.,
   * not its inverse H), derived from eq. 6.17 by duality:
   *
   *   B_new = B − (Bs)(Bs)ᵀ/(sᵀBs) + yyᵀ/(yᵀs)
   *
   * where s = step and y = g_new − g_old.  The update is skipped unless
   * yᵀs > 0 and sᵀBs > 0; both conditions together ensure B_new is positive
   * definite whenever B was (Dennis & Schnabel, §9.2).
   *
   * ExactAD mode: B was already refreshed by refresh_hessian at the top of
   * the current iteration; only the cached gradient is updated here.
   */
  constexpr std::pair<Point, Matrix>
  commit_state(const Point &step, const Point &g_old, const Matrix &B_cur);

private:
  Expr expr;
  Point g_new_{};
  value_type nn_{};
};

template <diff::CExpression Expr, HessianMode HM>
constexpr void Dogleg<Expr, HM>::adjust_tr(value_type &delta, value_type rho,
                                           bool at_boundary) {
  if (rho < Base::TR_DOWN_THRESHOLD) {
    if (!at_boundary) {
      delta = nn_;
    }
    delta *= Base::TR_DOWN_FACTOR;
  } else if (rho > Base::TR_UP_THRESHOLD && at_boundary) {
    delta *= Base::TR_UP_FACTOR;
  }
}
template <diff::CExpression Expr, HessianMode HM>
constexpr std::pair<typename Dogleg<Expr, HM>::Point,
                    typename Dogleg<Expr, HM>::Matrix>
Dogleg<Expr, HM>::commit_state(const Point &step, const Point &g_old,
                               const Matrix &B_cur) {
  Matrix B_new = B_cur;
  if constexpr (HM == HessianMode::BFGS) {
    const Point Bs = B_new * step;
    const value_type sBs = step.dot(Bs);
    const Point dg = g_new_ - g_old;
    const value_type ys = dg.dot(step);
    if (ys > value_type{0} && sBs > value_type{0}) {
      B_new += (dg * dg.transpose()) / ys - (Bs * Bs.transpose()) / sBs;
    }
  }
  // else if (HM == HessianMode::ExactAD)
  return {g_new_, B_new};
}

template <diff::CExpression Expr> Dogleg(Expr) -> Dogleg<Expr>;
template <diff::CExpression Expr, typename T> Dogleg(Expr, T) -> Dogleg<Expr>;

} // namespace exprmin
