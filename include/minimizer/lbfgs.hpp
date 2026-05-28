#pragma once

#include "bfgs.hpp"
#include "detail.hpp"
#include <utility>

namespace exprmin {

/**
 * @brief Nocedal §7.2 — Limited-memory BFGS minimizer.
 *
 * Replaces the N×N inverse-Hessian with the M most-recent (sₖ, yₖ) curvature
 * pairs and the two-loop recursion.  Storage O(M·N) vs O(N²) for full BFGS.
 * Line search selected by the same LS1D template parameter as BFGS.
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 * @tparam LS1D  Line-search policy (Brent default).
 * @tparam M     History size — number of (s, y) pairs retained (default 10).
 */
template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = Brent, int M = 10>
struct LBFGS : QuasiNewtonBase<Expr, LS1D> {
  using Base = QuasiNewtonBase<Expr, LS1D>;
  using Base::eval_at;
  using Base::fret;
  using Base::make_line_search_fn;
  using typename Base::Point;
  using typename Base::value_type;
  static constexpr int ITMAX = 200;

  constexpr explicit LBFGS(Expr e,
                           value_type gtol_ = static_cast<value_type>(1e-8))
      : Base(std::move(e), gtol_) {}

  /**
   * @brief Minimizes from initial point @p p.
   * @param p  Initial point.
   * @return Approximate minimizer.
   */
  constexpr Point minimize(Point p) {
    const auto eg = [this](const Point &q) { return this->eval_grad(q); };
    auto ls_fn = make_line_search_fn();
    DirState ds;
    p = this->quasi_newton_impl(eg, std::move(p), this->gtol, ITMAX, ls_fn, ds, this->iter);
    fret = eval_at(p);
    return p;
  }

private:
  static constexpr int NI = static_cast<int>(Base::N);

  /**
   * @brief Direction state: circular buffer of M most-recent (s, y) pairs.
   *
   * Applies the Nocedal two-loop recursion to compute −H_k·g in O(M·N)
   * without ever forming H_k explicitly.
   */
  struct DirState {
    using Point = Eigen::Vector<value_type, NI>;
    std::array<Point, M> s_buf, y_buf;
    std::array<value_type, M> rho_buf{};
    int buf_size = 0, buf_head = 0;

    /**
     * @brief Computes the L-BFGS search direction via the two-loop recursion.
     *
     * Returns −H_k·g built from the stored (s, y) pairs.  Falls back to −g
     * on the first iteration when the buffer is empty.
     *
     * @param g  Current gradient.
     * @return   Search direction vector.
     */
    constexpr Point compute(const Point &g) const {
      Point q = g;
      std::array<value_type, M> al{};
      for (int j = 0; j < buf_size; ++j) {
        const int idx = (buf_head - 1 - j + M) % M;
        al[j] = rho_buf[idx] * s_buf[idx].dot(q);
        q -= al[j] * y_buf[idx];
      }
      Point r;
      if (buf_size > 0) {
        const int last = (buf_head - 1 + M) % M;
        r = (s_buf[last].dot(y_buf[last]) / y_buf[last].squaredNorm()) * q;
      } else {
        r = q;
      }
      for (int j = buf_size - 1; j >= 0; --j) {
        const int idx = (buf_head - 1 - j + M) % M;
        r += s_buf[idx] * (al[j] - rho_buf[idx] * y_buf[idx].dot(r));
      }
      return -r;
    }

    /**
     * @brief Stores a new (s, y) pair, evicting the oldest when full.
     *
     * Skips pairs that violate the curvature condition y·s > ε·‖y‖² to keep
     * the implicit inverse-Hessian positive-definite.
     *
     * @param dx  Step s = x_new − x_old.
     * @param dg  Gradient difference y = g_new − g_old.
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

    /// @brief Clears the curvature-pair buffer (called after a direction failure).
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
