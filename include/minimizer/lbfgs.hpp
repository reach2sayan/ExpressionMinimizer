#pragma once

#include "bfgs.hpp"
#include "detail.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <array>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>
#include <utility>

namespace exprmin {

namespace mp = boost::mp11;

// Nocedal §7.2 — Limited-memory BFGS.
//
// Minimizes f(x) using the M most-recent (sₖ,yₖ) curvature pairs instead of
// an explicit N×N inverse Hessian.  Storage O(M·N) vs O(N²).
// Line search selected at compile time via LS (same enum as BFGS).
//   sₖ = xₖ₊₁ − xₖ,  yₖ = ∇fₖ₊₁ − ∇fₖ,  ρₖ = 1/(yₖᵀsₖ)
template <diff::CExpression Expr, LineSearch LS = LineSearch::Brent, int M = 10>
struct LBFGS {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

  static constexpr int ITMAX = 200;

  Expr expr;
  value_type fret{};
  int iter{};
  const value_type gtol;

  constexpr explicit LBFGS(Expr e,
                            value_type gtol_ = static_cast<value_type>(1.0e-8))
      : expr(std::move(e)), gtol(gtol_) {}

  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(expr);
    Point g = Eigen::Map<const Point>(g_arr.data());
    return {expr.eval(), std::move(g)};
  }

  constexpr Point minimize(Point p) {
    using std::abs, std::max;
    constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();
    constexpr value_type ZEPS = EPS * value_type{1e-3};
    constexpr value_type C1{1e-4};

    const auto eg = [this](const Point &q) -> std::pair<value_type, Point> {
      return eval_grad(q);
    };

    // Circular buffers for the M most recent (s, y) pairs
    std::array<Point, M> s_buf, y_buf;
    std::array<value_type, M> rho_buf{};
    int buf_size = 0, buf_head = 0;

    auto [fp, g] = eg(p);

    for (iter = 0; iter < ITMAX; ++iter) {
      // Scaled-gradient convergence criterion
      const value_type den = max(abs(fp), value_type{1});
      if ((g.cwiseAbs().array() *
           p.cwiseAbs().cwiseMax(value_type{1}).array())
              .maxCoeff() /
          den <
          gtol)
        break;

      // ── Nocedal two-loop recursion ────────────────────────────────────────
      // Loop 1: newest → oldest — compute α and update q
      Point q = g;
      std::array<value_type, M> al{};
      for (int j = 0; j < buf_size; ++j) {
        const int idx = (buf_head - 1 - j + M) % M;
        al[j] = rho_buf[idx] * s_buf[idx].dot(q);
        q -= al[j] * y_buf[idx];
      }
      // Initial scaling: γ = (sₖᵀyₖ) / (yₖᵀyₖ) from most-recent pair
      Point r;
      if (buf_size > 0) {
        const int last = (buf_head - 1 + M) % M;
        const value_type gamma =
            s_buf[last].dot(y_buf[last]) / y_buf[last].squaredNorm();
        r = gamma * q;
      } else {
        r = q;
      }
      // Loop 2: oldest → newest — apply second correction
      for (int j = buf_size - 1; j >= 0; --j) {
        const int idx = (buf_head - 1 - j + M) % M;
        const value_type beta = rho_buf[idx] * y_buf[idx].dot(r);
        r += s_buf[idx] * (al[j] - beta);
      }
      Point xi = -r; // search direction ≈ −H⁻¹·g

      // If not a descent direction, reset and fall back to steepest descent
      if (g.dot(xi) >= value_type{}) {
        buf_size = 0;
        buf_head = 0;
        xi = -g;
      }

      // ── Line search: compute dx = α·xi ───────────────────────────────────
      const value_type slope = g.dot(xi);
      Point dx;

      if constexpr (LS == LineSearch::Armijo) {
        value_type alpha{1};
        for (int k = 0; k < 40 && alpha > EPS; ++k, alpha *= value_type{0.5})
          if (eg(p + alpha * xi).first <= fp + C1 * alpha * slope)
            break;
        dx = alpha * xi;
      } else if constexpr (LS == LineSearch::Dbrent) {
        struct Funcd {
          decltype(eg) eg_;
          const Point &p_, &xi_;
          value_type operator()(value_type t) const {
            return eg_(p_ + t * xi_).first;
          }
          value_type df(value_type t) const {
            return eg_(p_ + t * xi_).second.dot(xi_);
          }
        } fc{eg, p, xi};
        value_type ax{0}, bx{1}, cx, fa = fc(ax), fb = fc(bx), fc_val;
        detail::bracket(fc, ax, bx, cx, fa, fb, fc_val);
        dx = detail::dbrent(fc, ax, bx, cx, gtol, ZEPS) * xi;
      } else {
        auto f1d = [&](value_type t) { return eg(p + t * xi).first; };
        value_type ax{0}, bx{1}, cx, fa = f1d(ax), fb = f1d(bx), fc_val;
        detail::bracket(f1d, ax, bx, cx, fa, fb, fc_val);
        dx = detail::brent(f1d, ax, bx, cx, gtol, ZEPS) * xi;
      }

      const Point p_new = p + dx;
      auto [fn, g_new] = eg(p_new);

      // Store pair only when curvature condition holds: yᵀs > ε·‖y‖²
      const Point y = g_new - g;
      const value_type ys = y.dot(dx);
      if (ys > EPS * y.squaredNorm()) {
        s_buf[buf_head] = dx;
        y_buf[buf_head] = y;
        rho_buf[buf_head] = value_type{1} / ys;
        buf_head = (buf_head + 1) % M;
        if (buf_size < M)
          ++buf_size;
      }

      p = p_new;
      g = std::move(g_new);
      fp = fn;
    }

    fret = eval_at(p);
    return p;
  }
};

template <diff::CExpression Expr> LBFGS(Expr) -> LBFGS<Expr>;
template <diff::CExpression Expr, typename T> LBFGS(Expr, T) -> LBFGS<Expr>;

} // namespace exprmin
