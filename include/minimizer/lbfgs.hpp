#pragma once

#include "bfgs.hpp"
#include <utility>

namespace exprmin {

/**
 * @brief Nocedal & Wright §7.2 — Limited-memory BFGS (Algorithm 7.5).
 *
 * Replaces the dense N×N inverse-Hessian approximation Hₖ of full BFGS with
 * the M most-recent curvature pairs {(sᵢ, yᵢ)} and the two-loop recursion
 * (Algorithm 7.4).  Each iterate follows
 *
 *   xₖ₊₁ = xₖ + αₖ pₖ,   pₖ = −Hₖ ∇fₖ                  (eq. 7.15)
 *
 * where αₖ satisfies the Wolfe conditions and Hₖ is built implicitly from
 * the M pairs via eq. 7.19.  Storage is O(M·N) instead of O(N²), making this
 * the preferred choice for large N when the true Hessian is dense or expensive.
 *
 * The initial Hessian scaling H⁰ₖ = γₖI uses γₖ = sᵀₖ₋₁yₖ₋₁ / yᵀₖ₋₁yₖ₋₁
 * (eq. 7.20) to keep the search direction well-scaled so that αₖ = 1 is
 * accepted in most iterations.
 *
 * Convergence is linear in general; the method performs well on large smooth
 * problems but can be slow on ill-conditioned ones with a wide eigenvalue
 * spread (§7.2, "Limited-Memory BFGS" discussion).
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 * @tparam LS1D  Line-search policy (Brent default; Armijo also supported).
 * @tparam M     History depth — number of (s, y) pairs retained (default 10).
 *               Practical values: 3–20; larger M improves curvature info at
 *               higher per-iteration cost (see Table 7.1).
 */
template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = Brent, int M = 10>
struct LBFGS : QuasiNewtonBase<Expr, LS1D> {
  using Base = QuasiNewtonBase<Expr, LS1D>;
  using Base::eval_at;
  using Base::eval_grad;
  using Base::fret;
  using Base::make_line_search_fn;
  using Base::quasi_newton_impl;
  using typename Base::Point;
  using typename Base::value_type;
  static constexpr int ITMAX = 200;

  /**
   * @brief Constructs an LBFGS minimizer wrapping the given expression.
   * @param e      Expression to minimize.
   * @param gtol_  Scaled-gradient convergence tolerance (default 10⁻⁸).
   */
  constexpr explicit LBFGS(Expr e,
                           value_type gtol_ = static_cast<value_type>(1e-8))
      : Base(std::move(e), gtol_) {}

  /**
   * @brief Minimizes from initial point @p p.
   * @param p  Initial point.
   * @return Approximate minimizer.
   */
  constexpr Point minimize(Point p) {
    const auto eg = [this](const Point &q) { return eval_grad(q); };
    auto ls_fn = make_line_search_fn();
    DirState ds;
    p = quasi_newton_impl(eg, std::move(p), this->gtol, ITMAX, ls_fn, ds,
                          this->iter);
    fret = eval_at(p);
    return p;
  }

private:
  static constexpr int NI = static_cast<int>(Base::N);

  /**
   * @brief Direction state: circular buffer of the M most-recent (sᵢ, yᵢ)
   * pairs.
   *
   * Implements Algorithm 7.4 (two-loop recursion) to compute pₖ = −Hₖ∇fₖ in
   * O(M·N) time without ever forming the N×N matrix Hₖ explicitly.  The
   * buffer is written in FIFO order with buf_head pointing to the next write
   * slot; buf_size counts how many valid pairs are held (capped at M).
   */
  struct DirState {
    using Point = Eigen::Vector<value_type, NI>;
    std::array<Point, M> s_buf, y_buf;
    std::array<value_type, M> rho_buf{};
    int buf_size = 0, buf_head = 0;

    /**
     * @brief Computes the L-BFGS search direction (Algorithm 7.4).
     *
     * Executes the two-loop recursion that implicitly applies Hₖ to ∇fₖ:
     *
     *   Loop 1 (newest→oldest): αᵢ ← ρᵢ sᵢᵀ q;  q ← q − αᵢ yᵢ
     *   Scale:                  r  ← H⁰ₖ q = γₖ q  (eq. 7.20)
     *   Loop 2 (oldest→newest): βᵢ ← ρᵢ yᵢᵀ r;  r ← r + sᵢ(αᵢ − βᵢ)
     *
     * where ρᵢ = 1/(yᵢᵀsᵢ) (eq. 7.17) and the result pₖ = −r.
     *
     * Falls back to pₖ = −g (steepest descent) on the first call when the
     * buffer is empty (equivalent to H⁰ₖ = I, m = 0).
     *
     * @param g  Current gradient ∇fₖ.
     * @return   Search direction pₖ = −Hₖ∇fₖ.
     */
    constexpr Point compute(const Point &g) const {
      Point q = g; // q ← ∇fₖ
      std::array<value_type, M> al{};
      // First loop: i = k−1, k−2, …, k−m  (newest pair first)
      for (int j = 0; j < buf_size; ++j) {
        const int idx = (buf_head - 1 - j + M) % M;
        al[j] = rho_buf[idx] * s_buf[idx].dot(q); // αᵢ ← ρᵢ sᵢᵀ q
        q -= al[j] * y_buf[idx];                  // q  ← q − αᵢ yᵢ
      }
      // Initial scaling H⁰ₖ = γₖI where γₖ = sᵀy/yᵀy (eq. 7.20).
      // Isolates H⁰ₖ from the correction pairs so it can vary each iteration,
      // keeping the search direction well-scaled for unit-step acceptance.
      Point r;
      if (buf_size > 0) {
        const int last = (buf_head - 1 + M) % M;
        r = (s_buf[last].dot(y_buf[last]) / y_buf[last].squaredNorm()) * q;
      } else {
        r = q; // H⁰ₖ = I on cold start
      }
      // Second loop: i = k−m, k−m+1, …, k−1  (oldest pair first)
      for (int j = buf_size - 1; j >= 0; --j) {
        const int idx = (buf_head - 1 - j + M) % M;
        // βᵢ ← ρᵢ yᵢᵀ r;  r ← r + sᵢ(αᵢ − βᵢ)  (inlined to avoid a temp)
        r += s_buf[idx] * (al[j] - rho_buf[idx] * y_buf[idx].dot(r));
      }
      return -r; // pₖ = −Hₖ∇fₖ
    }

    /**
     * @brief Records a new (s, y) pair, evicting the oldest when the buffer is
     * full.
     *
     * Stores sₖ = xₖ₊₁ − xₖ (dx) and yₖ = ∇fₖ₊₁ − ∇fₖ (dg) together with
     * ρₖ = 1/(yₖᵀsₖ) (eq. 7.17).  Pairs that violate the curvature condition
     *
     *   yᵀs > ε ‖y‖²
     *
     * are silently skipped: accepting them would make ρₖ ≤ 0 and could
     * render the implicit Hₖ indefinite, destabilising the line search.
     * This stricter check (vs. yᵀs > 0) guards against near-zero ys that
     * could cause numerical blow-up in ρ.
     *
     * @param dx  Step vector sₖ = x_new − x_old.
     * @param dg  Gradient difference yₖ = g_new − g_old.
     */
    constexpr void update(const Point &dx, const Point &dg) {
      constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();
      const value_type ys = dg.dot(dx);
      if (ys > EPS * dg.squaredNorm()) {
        s_buf[buf_head] = dx;
        y_buf[buf_head] = dg;
        rho_buf[buf_head] = value_type{1} / ys;
        buf_head = (buf_head + 1) % M;
        if (buf_size < M) {
          ++buf_size;
        }
      }
    }

    /// @brief Clears the curvature-pair buffer (called after a direction
    /// failure).
    constexpr void reset() {
      buf_size = 0;
      buf_head = 0;
    }
  };
};

template <diff::CExpression Expr> LBFGS(Expr) -> LBFGS<Expr>;
template <diff::CExpression Expr, diff::Numeric T>
LBFGS(Expr, T) -> LBFGS<Expr>;

} // namespace exprmin
