#pragma once

#include "bracketmethod.hpp"
#include "detail.hpp"
#include "gradient.hpp"
#include <limits>

namespace exprmin {

/**
 * @brief NR §10.3 — Brent's method for derivative-free 1-D minimization.
 *
 * Combines golden-section search with parabolic interpolation inside a
 * pre-computed bracket [ax, cx] with interior point bx.  Converges
 * super-linearly on smooth unimodal functions.
 *
 * @tparam T      Numeric scalar type.
 * @tparam F      Callable with signature T(T).
 * @param f       Objective function.
 * @param ax      Left bracket endpoint.
 * @param bx      Interior point (initial best-guess abscissa).
 * @param cx      Right bracket endpoint.
 * @param tol     Fractional tolerance on the abscissa.
 * @param zeps    Small additive offset preventing tol from collapsing to zero.
 * @param itmax   Maximum number of iterations (default 100).
 * @return Abscissa of the minimum.
 */
struct BrentFn {
  template <diff::Numeric T, std::invocable<T> F>
  constexpr T operator()(F &f, const T &ax, const T &bx, const T &cx,
                         const T &tol, const T &zeps,
                         const int itmax = 100) const {
    using std::abs;
    using std::max;
    using std::min;
    constexpr T CGOLD = static_cast<T>(1.0 - 1.0 / std::numbers::phi_v<double>);

    // Three points x (best), w (2nd), v (3rd) are kept ranked as fx≤fw≤fv.
    // The parabolic interpolation below fits a curve through all three, so the
    // housekeeping at the end of each iteration must update them every step.
    T a = min(ax, cx); // sort so a ≤ b regardless of the input order
    T b = max(ax, cx);
    T x = bx, w = bx, v = bx;
    T fx = f(x), fw = fx, fv = fx;
    T d{}, e{};

    auto golden_section_step = [&CGOLD](auto &e, auto &d, const auto &a,
                                        const auto &b, const auto &x,
                                        const auto &xm) {
      e = (x >= xm ? a - x : b - x);
      d = CGOLD * e;
    };
    for (int i = 0; i < itmax; ++i) {
      const T xm = T{0.5} * (a + b);
      const T tol1 = tol * abs(x) + zeps;
      const T tol2 = T{2} * tol1;

      if (abs(x - xm) <= tol2 - T{0.5} * (b - a)) {
        return x;
      }

      T u;
      if (abs(e) > tol1) {
        // attempt parabolic interpolation through (x, w, v)
        T r = (x - w) * (fx - fv);
        T q = (x - v) * (fx - fw);
        T p = (x - v) * q - (x - w) * r;
        q = T{2} * (q - r);
        if (q > T{}) {
          p = -p;
        }
        q = abs(q);
        const T etemp = e;
        e = d;
        if (abs(p) >= abs(T{0.5} * q * etemp) || p <= q * (a - x) ||
            p >= q * (b - x)) {
          // parabolic step is too large or out of bracket; fall back to golden section
          golden_section_step(e, d, a, b, x, xm);
        } else {
          // parabolic step accepted: inside bracket and at most half of last step
          d = p / q;
          u = x + d;
          if (u - a < tol2 || b - u < tol2) {
            d = (xm >= x ? tol1 : -tol1);
          }
        }
      } else {
        golden_section_step(e, d, a, b, x, xm);
      }

      // clamp step to at least tol1 so u is never evaluated right on top of x
      u = (abs(d) >= tol1 ? x + d : x + (d >= T{} ? tol1 : -tol1));
      const T fu = f(u);

      // --- housekeeping: tighten the bracket and maintain the ranked trio ---
      // Part 1 — tighten [a, b]: u proves the minimum cannot be on one side of
      // x, so we move whichever bound is on that side inward.
      //   fu ≤ fx  →  x is now interior; the side opposite u can be discarded.
      //   fu > fx  →  u itself becomes the new bound on its side.
      //
      // Part 2 — slot u into the ranking so next iteration has fresh curvature:
      //   fu ≤ fx  →  u beats everything; demote the chain v ← w ← x ← u.
      //   fu ≤ fw  →  u is second-best; shift old w down to v, u takes w.
      //   fu ≤ fv  →  u is third-best; replace v.
      //   fu > fv  →  u is worse than all three; discard it (no update needed).
      if (fu <= fx) {
        if (u < x) {
          b = x;
        } else {
          a = x;
        }
        v = w;
        fv = fw;
        w = x;
        fw = fx;
        x = u;
        fx = fu;
      } else {
        if (u < x) {
          a = u;
        } else {
          b = u;
        }
        if (fu <= fw || w == x) {
          v = w;
          fv = fw;
          w = u;
          fw = fu;
        } else if (fu <= fv || v == x || v == w) {
          v = u;
          fv = fu;
        }
      }
    }
    return x;
  }
};

/// Callable object for the NR §10.3 Brent algorithm (see BrentFn).
inline constexpr BrentFn brent{};

/**
 * @brief NR §10.4 — Brent's method augmented with derivative information.
 *
 * Uses secant interpolation on f′ to propose steps; falls back to bisection
 * toward the sign-change side when the secant step leaves the bracket or
 * fails to shrink the interval by at least half.  Typically converges in
 * fewer evaluations than plain Brent on smooth functions.
 *
 * @pre  @p F must expose both @c operator()(T) (function value) and
 *       @c df(T) (first derivative).
 *
 * @tparam T      Numeric scalar type.
 * @tparam F      Functor exposing operator()(T) and df(T).
 * @param f       Objective functor.
 * @param ax      Left bracket endpoint.
 * @param bx      Interior point (initial best-guess abscissa).
 * @param cx      Right bracket endpoint.
 * @param tol     Fractional tolerance on the abscissa.
 * @param zeps    Small additive offset preventing tol from collapsing to zero.
 * @param itmax   Maximum number of iterations (default 100).
 * @return Abscissa of the minimum.
 */
struct DbrentFn {
  template <diff::Numeric T, std::invocable<T> F>
  constexpr T operator()(F &f, const T &ax, const T &bx, const T &cx,
                         const T &tol, const T &zeps, int itmax = 100) const {
    using std::abs;
    using std::max;
    using std::min;

    // Same three-point ranked trio as brent (x best, w 2nd, v 3rd, fx≤fw≤fv),
    // but each point also carries its derivative (dx, dw, dv) so the secant
    // step can be computed without extra f' calls during housekeeping.
    T a = min(ax, cx), b = max(ax, cx);
    T x = bx, w = bx, v = bx;
    T fx = f(bx), fw = fx, fv = fx;
    T dx = f.df(bx), dw = dx, dv = dx;
    T d{}, e{};

    for (int i = 0; i < itmax; ++i) {
      const T xm = T{0.5} * (a + b);
      const T tol1 = tol * abs(x) + zeps;
      const T tol2 = T{2} * tol1;
      if (abs(x - xm) <= tol2 - T{0.5} * (b - a)) {
        return x;
      }

      // Bisect toward the side where f' points downhill; used whenever the
      // secant step is unavailable or unreliable (same role as golden section
      // in plain Brent, but guided by the sign of the derivative).
      auto bisect = [](const T dx, const T a, const T b, const T x) {
        return dx >= T{} ? a - x : b - x;
      };

      if (abs(e) > tol1) {
        // Attempt secant steps using the (x,w) and (x,v) derivative pairs.
        // Initialise d1/d2 to a value outside [a,b] so ok1/ok2 stays false
        // for whichever pair has equal derivatives (no valid secant step).
        T d1 = T{2} * (b - a), d2 = d1;
        if (dw != dx) {
          d1 = (w - x) * dx / (dx - dw); // secant step through (x, w)
        }
        if (dv != dx) {
          d2 = (v - x) * dx / (dx - dv); // secant step through (x, v)
        }
        const T u1 = x + d1, u2 = x + d2;
        // a step is acceptable only if it lands inside [a,b] and moves
        // in the downhill direction (dx·d ≤ 0)
        const bool ok1 = (a - u1) * (u1 - b) > T{} && dx * d1 <= T{};
        const bool ok2 = (a - u2) * (u2 - b) > T{} && dx * d2 <= T{};
        const T olde = e;
        e = d;
        if (ok1 || ok2) {
          // prefer the shorter of the two valid steps
          d = (ok1 && ok2) ? (abs(d1) < abs(d2) ? d1 : d2) : (ok1 ? d1 : d2);
          if (abs(d) <= abs(T{0.5} * olde)) {
            const T u = x + d;
            if (u - a < tol2 || b - u < tol2)
              d = (xm >= x ? tol1 : -tol1);
          } else {
            e = bisect(dx, a, b, x);
            d = T{0.5} * e;
          }
        } else {
          e = bisect(dx, a, b, x);
          d = T{0.5} * e;
        }
      } else {
        e = bisect(dx, a, b, x);
        d = T{0.5} * e;
      }

      const T u = (abs(d) >= tol1 ? x + d : x + (d >= T{} ? tol1 : -tol1));
      const T fu = f(u);
      const T du = f.df(u);

      // --- housekeeping: identical bracket/ranking logic to brent, but each
      // slot in the trio also carries a derivative that must move with it.
      // The derivative at the demoted point is preserved because the secant
      // step next iteration needs dx at w and v, not just their positions.
      if (fu <= fx) {
        (u < x ? b : a) = x;
        v = w;
        fv = fw;
        dv = dw;
        w = x;
        fw = fx;
        dw = dx;
        x = u;
        fx = fu;
        dx = du;
      } else {
        (u < x ? a : b) = u;
        if (fu <= fw || w == x) {
          v = w;
          fv = fw;
          dv = dw;
          w = u;
          fw = fu;
          dw = du;
        } else if (fu <= fv || v == x || v == w) {
          v = u;
          fv = fu;
          dv = du;
        }
      }
    }
    return x;
  }
};

/// Callable object for the NR §10.4 Dbrent algorithm (see DbrentFn).
inline constexpr DbrentFn dbrent{};

/**
 * @brief NR §10.3 — Brent's method: parabolic interpolation with golden-section
 * fallback.
 */
template <diff::CExpression Expr> struct Brent : Bracketmethod<Expr> {
  using Base = Bracketmethod<Expr>;
  using value_type = typename Base::value_type;
  using Syms = typename Base::Syms;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;
  using Base::ax;
  using Base::bracket;
  using Base::bx;
  using Base::cx;
  using Base::eval_at;
  using Base::expr;
  static constexpr value_type ZEPS =
      std::numeric_limits<value_type>::epsilon() *
      static_cast<value_type>(1.0e-3);
  static constexpr int ITMAX = 100;

protected:
  value_type xmin{};
  value_type fmin{};
  const value_type tol;

public:
  constexpr value_type get_optimal_value() const { return fmin; }
  constexpr value_type const &get_optimal_x() const { return xmin; }
  constexpr value_type &get_optimal_x() { return xmin; }

  constexpr explicit Brent(Expr e,
                           value_type tol_ = static_cast<value_type>(3.0e-8))
      : Base(std::move(e)), tol(tol_) {}

  // N-D point evaluation — used when Brent<Expr> is owned by BFGS / LBFGS /
  // LinMin.
  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  // 1D-only overloads: only valid for single-variable expressions.
  constexpr value_type minimize()
    requires(N == 1)
  {
    auto f = [this](const value_type &x) { return eval_at(x); };
    xmin = brent_impl(f);
    fmin = eval_at(xmin);
    return xmin;
  }

  constexpr value_type minimize(const value_type &ax0, const value_type &bx0)
    requires(N == 1)
  {
    bracket(ax0, bx0);
    return minimize();
  }

  // Minimize any 1D callable — used by LinMin / BFGS / LBFGS.
  template <std::invocable<value_type> F>
  constexpr value_type minimize_fn(F f1d, value_type ax0, value_type bx0) {
    value_type a = ax0, b = bx0, c;
    value_type fa = f1d(a), fb = f1d(b), fc;
    exprmin::bracket(f1d, a, b, c, fa, fb, fc);
    xmin = brent(f1d, a, b, c, tol, ZEPS, ITMAX);
    fmin = f1d(xmin);
    return xmin;
  }

private:
  // Binds this instance's bracket [ax, bx, cx] and tolerances to the brent algorithm.
  template <std::invocable<value_type> F>
  constexpr value_type brent_impl(F &f) {
    return exprmin::brent(f, ax, bx, cx, tol, ZEPS, ITMAX);
  }
};

template <diff::CExpression Expr> Brent(Expr) -> Brent<Expr>;
template <diff::CExpression Expr, typename T> Brent(Expr, T) -> Brent<Expr>;

/**
 * @brief NR §10.4 — Brent + first-derivative (secant on f′ via reverse-mode AD).
 * Inherits all state from Brent; only minimize() differs.
 */
template <diff::CExpression Expr> struct Dbrent : Brent<Expr> {
  using Base = Brent<Expr>;
  using Base::Base; // inherit constructors
  using value_type = typename Base::value_type;
  using Syms = typename Base::Syms;
  using Base::ax;
  using Base::bracket;
  using Base::bx;
  using Base::cx;
  using Base::eval_at;
  using Base::fmin;
  using Base::tol;
  using Base::xmin;
  constexpr value_type minimize();

  constexpr value_type minimize(const value_type &ax0, const value_type &bx0);

  // Minimize any 1D callable with derivative.  FC must expose operator()(T)
  // and df(T).  Used by DLinMin / BFGS<Dbrent> / LBFGS<Dbrent>.
  template <typename FC>
  constexpr value_type minimize_fn(FC fc, value_type ax0, value_type bx0);

private:
  // Binds this instance's bracket [ax, bx, cx] and tolerances to the dbrent algorithm.
  template <typename FC>
  constexpr value_type dbrent_impl(FC &fc) {
    return exprmin::dbrent(fc, ax, bx, cx, tol, Base::ZEPS, Base::ITMAX);
  }
};

template <diff::CExpression Expr>
constexpr typename Dbrent<Expr>::value_type Dbrent<Expr>::minimize() {
  struct Funcd {
    Dbrent &self;
    value_type operator()(value_type t) { return self.eval_at(t); }
    value_type df(value_type t) {
      std::array<value_type, 1> v{t};
      self.expr.update(Syms{}, v);
      const auto g = diff::gradient<diff::DiffMode::Reverse>(self.expr);
      return g[0];
    }
  };
  Funcd fc{*this};
  xmin = dbrent_impl(fc);
  fmin = eval_at(xmin);
  return xmin;
}

template <diff::CExpression Expr>
constexpr typename Dbrent<Expr>::value_type
Dbrent<Expr>::minimize(const value_type &ax0, const value_type &bx0) {
  bracket(ax0, bx0);
  return minimize();
}

template <diff::CExpression Expr>
template <typename FC>
constexpr typename Dbrent<Expr>::value_type
Dbrent<Expr>::minimize_fn(FC fc, value_type ax0, value_type bx0) {
  value_type a = ax0, b = bx0, c;
  value_type fa = fc(a), fb = fc(b), fc_val;
  exprmin::bracket(fc, a, b, c, fa, fb, fc_val);
  xmin = dbrent(fc, a, b, c, tol, Base::ZEPS, Base::ITMAX);
  fmin = fc(xmin);
  return xmin;
}

template <diff::CExpression Expr> Dbrent(Expr) -> Dbrent<Expr>;
template <diff::CExpression Expr, typename T> Dbrent(Expr, T) -> Dbrent<Expr>;

} // namespace exprmin
