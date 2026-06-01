#pragma once

#include "bracketmethod.hpp"
#include "gradient.hpp"
#include <limits>

namespace exprmin {

/**
 * @brief The ranked trio shared by Brent (§10.3) and Dbrent (§10.4).
 *
 * Holds the three best abscissae seen so far — x (best), w (2nd), v (3rd),
 * ranked fx ≤ fw ≤ fv — that feed the parabolic / secant interpolation.  When
 * @p WithDeriv is true each slot also carries its first derivative (dx/dw/dv),
 * preserved through demotion so Dbrent's secant step needs no extra f' calls.
 *
 * A fixed-size, allocation-free, constexpr replacement for the hand-rolled
 * housekeeping: a sorted container cannot serve here because the bracket [a,b]
 * is ordered by position while the trio is ordered by value, the demotion is
 * bracket/recency-aware rather than a pure k-smallest queue, and node-based
 * containers are not usable in the library's constexpr paths.
 *
 * @tparam T          Numeric scalar type.
 * @tparam WithDeriv  Whether to track per-slot derivatives (Dbrent).
 */
template <diff::Numeric T, bool WithDeriv = false> struct BrentState {
  T x{}, fx{}; ///< best
  T w{}, fw{}; ///< 2nd best
  T v{}, fv{}; ///< 3rd best
  T dx{}, dw{}, dv{}; ///< per-slot derivatives — used only when WithDeriv

  /**
   * @brief Tighten the bracket [a,b] around new point u and re-rank the trio.
   *
   * Part 1 — tighten [a, b]: u proves the minimum cannot be on one side of x,
   * so move whichever bound is on that side inward.
   *   fu ≤ fx  →  x becomes interior; the side opposite u is discarded.
   *   fu > fx  →  u itself becomes the new bound on its side.
   * Part 2 — slot u into the value ranking so the next iteration has fresh
   * curvature: demote the chain v ← w ← x ← u, or insert at w / v, or discard.
   *
   * @param du  Derivative at u; ignored unless WithDeriv.
   */
  constexpr void update(T &a, T &b, T u, T fu, T du = T{}) {
    if (fu <= fx) {
      (u < x ? b : a) = x;
      v = w;
      fv = fw;
      if constexpr (WithDeriv) {
        dv = dw;
      }
      w = x;
      fw = fx;
      if constexpr (WithDeriv) {
        dw = dx;
      }
      x = u;
      fx = fu;
      if constexpr (WithDeriv) {
        dx = du;
      }
    } else {
      (u < x ? a : b) = u;
      if (fu <= fw || w == x) {
        v = w;
        fv = fw;
        if constexpr (WithDeriv) {
          dv = dw;
        }
        w = u;
        fw = fu;
        if constexpr (WithDeriv) {
          dw = du;
        }
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
        if constexpr (WithDeriv) {
          dv = du;
        }
      }
    }
  }
};

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
    using std::max;
    using std::min;
    constexpr T CGOLD = static_cast<T>(1.0 - 1.0 / std::numbers::phi_v<double>);

    // Three points x (best), w (2nd), v (3rd) are kept ranked as fx≤fw≤fv in a
    // BrentState. The parabolic interpolation below fits a curve through all
    // three; st.update() re-ranks them at the end of each iteration. References
    // alias the trio so the algorithm body reads unchanged.
    T a = min(ax, cx); // sort so a ≤ b regardless of the input order
    T b = max(ax, cx);
    const T f0 = f(bx);
    BrentState<T> st{.x = bx, .fx = f0, .w = bx, .fw = f0, .v = bx, .fv = f0};
    T &x = st.x, &w = st.w, &v = st.v;
    T &fx = st.fx, &fw = st.fw, &fv = st.fv;
    T d{}, e{};

    auto golden_section_step = [&CGOLD](auto &e, auto &d, const auto &a,
                                        const auto &b, const auto &x,
                                        const auto &xm) {
      e = (x >= xm ? a - x : b - x);
      d = CGOLD * e;
    };
    for (int i = 0; i < itmax; ++i) {
      const T xm = T{0.5} * (a + b);
      const T tol1 = tol * detail::abs_for_constexpr(x) + zeps;
      const T tol2 = T{2} * tol1;

      if (detail::abs_for_constexpr(x - xm) <= tol2 - T{0.5} * (b - a)) {
        return x;
      }

      T u;
      if (detail::abs_for_constexpr(e) > tol1) {
        // attempt parabolic interpolation through (x, w, v)
        T r = (x - w) * (fx - fv);
        T q = (x - v) * (fx - fw);
        T p = (x - v) * q - (x - w) * r;
        q = T{2} * (q - r);
        if (q > T{}) {
          p = -p;
        }
        q = detail::abs_for_constexpr(q);
        const T etemp = e;
        e = d;
        if (detail::abs_for_constexpr(p) >=
                detail::abs_for_constexpr(T{0.5} * q * etemp) ||
            p <= q * (a - x) || p >= q * (b - x)) {
          // parabolic step is too large or out of bracket; fall back to golden
          // section
          golden_section_step(e, d, a, b, x, xm);
        } else {
          // parabolic step accepted: inside bracket and at most half of last
          // step
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
      u = (detail::abs_for_constexpr(d) >= tol1
               ? x + d
               : x + (d >= T{} ? tol1 : -tol1));
      const T fu = f(u);

      // tighten the bracket and maintain the ranked trio
      st.update(a, b, u, fu);
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
    using std::max;
    using std::min;

    // Same three-point ranked trio as brent (x best, w 2nd, v 3rd, fx≤fw≤fv),
    // but each point also carries its derivative (dx, dw, dv) so the secant
    // step can be computed without extra f' calls during housekeeping.
    // References alias the trio so the algorithm body reads unchanged.
    T a = min(ax, cx), b = max(ax, cx);
    const T f0 = f(bx);
    const T d0 = f.df(bx);
    BrentState<T, true> st{.x = bx, .fx = f0, .w = bx, .fw = f0,
                           .v = bx, .fv = f0, .dx = d0, .dw = d0, .dv = d0};
    // Dbrent's body navigates by position and derivative; the trio's f-values
    // are consumed only inside st.update(), so no f-value aliases are needed.
    T &x = st.x, &w = st.w, &v = st.v;
    T &dx = st.dx, &dw = st.dw, &dv = st.dv;
    T d{}, e{};

    for (int i = 0; i < itmax; ++i) {
      const T xm = T{0.5} * (a + b);
      const T tol1 = tol * detail::abs_for_constexpr(x) + zeps;
      const T tol2 = T{2} * tol1;
      if (detail::abs_for_constexpr(x - xm) <= tol2 - T{0.5} * (b - a)) {
        return x;
      }

      // Bisect toward the side where f' points downhill; used whenever the
      // secant step is unavailable or unreliable (same role as golden section
      // in plain Brent, but guided by the sign of the derivative).
      auto bisect = [](const T dx, const T a, const T b, const T x) {
        return dx >= T{} ? a - x : b - x;
      };

      if (detail::abs_for_constexpr(e) > tol1) {
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
          d = (ok1 && ok2)
                  ? (detail::abs_for_constexpr(d1) < detail::abs_for_constexpr(d2)
                         ? d1
                         : d2)
                  : (ok1 ? d1 : d2);
          if (detail::abs_for_constexpr(d) <=
              detail::abs_for_constexpr(T{0.5} * olde)) {
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

      const T u = (detail::abs_for_constexpr(d) >= tol1
                       ? x + d
                       : x + (d >= T{} ? tol1 : -tol1));
      const T fu = f(u);
      const T du = f.df(u);

      // tighten the bracket and maintain the ranked trio (with derivatives)
      st.update(a, b, u, fu, du);
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
  /// @brief Returns f at the minimizing abscissa after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fmin; }
  /// @brief Returns the minimizing abscissa (read-only).
  constexpr value_type const &get_optimal_x() const { return xmin; }
  /// @brief Returns the minimizing abscissa (mutable).
  constexpr value_type &get_optimal_x() { return xmin; }

  /**
   * @brief Constructs a Brent minimizer wrapping the given expression.
   * @param e     Expression to minimize.
   * @param tol_  Fractional abscissa tolerance (default 3×10⁻⁸).
   */
  constexpr explicit Brent(Expr e,
                           value_type tol_ = static_cast<value_type>(3.0e-8))
      : Base(std::move(e)), tol(tol_) {}

  /// @brief N-D point evaluation — used when Brent is owned by BFGS/LBFGS/LinMin.
  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  /**
   * @brief 1-D minimization using the pre-computed bracket stored in @c ax, @c bx, @c cx.
   * @return Abscissa of the minimum.
   */
  constexpr value_type minimize()
    requires(N == 1)
  {
    auto f = [this](const value_type &x) { return eval_at(x); };
    xmin = brent_impl(f);
    fmin = eval_at(xmin);
    return xmin;
  }

  /**
   * @brief Brackets [ax0, bx0] then minimizes.
   * @return Abscissa of the minimum.
   */
  constexpr value_type minimize(const value_type &ax0, const value_type &bx0)
    requires(N == 1)
  {
    bracket(ax0, bx0);
    return minimize();
  }

  /**
   * @brief Minimizes any 1-D callable — used by LinMin, BFGS, LBFGS.
   *
   * Brackets @p f1d on [ax0, bx0], then refines with Brent.
   *
   * @param f1d  1-D callable T(T).
   * @param ax0  Left end of the initial interval.
   * @param bx0  Right end of the initial interval.
   * @return     Abscissa of the minimum.
   */
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
  // Binds this instance's bracket [ax, bx, cx] and tolerances to the brent
  // algorithm.
  template <std::invocable<value_type> F>
  constexpr value_type brent_impl(F &f) {
    return exprmin::brent(f, ax, bx, cx, tol, ZEPS, ITMAX);
  }
};

template <diff::CExpression Expr> Brent(Expr) -> Brent<Expr>;
template <diff::CExpression Expr, typename T> Brent(Expr, T) -> Brent<Expr>;

/**
 * @brief NR §10.4 — Brent + first-derivative (secant on f′ via reverse-mode
 * AD). Inherits all state from Brent; only minimize() differs.
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
  /**
   * @brief 1-D minimization using the pre-computed bracket and derivative
   *        information from the expression.
   * @return Abscissa of the minimum.
   */
  constexpr value_type minimize();

  /**
   * @brief Brackets [ax0, bx0] then minimizes using Dbrent.
   * @return Abscissa of the minimum.
   */
  constexpr value_type minimize(const value_type &ax0, const value_type &bx0);

  /**
   * @brief Minimizes any 1-D callable with derivative — used by DLinMin,
   *        BFGS<Dbrent>, LBFGS<Dbrent>.
   *
   * @p FC must expose @c operator()(T) (value) and @c df(T) (derivative).
   * Brackets on [ax0, bx0], then refines with Dbrent.
   *
   * @param fc   Functor with value and derivative.
   * @param ax0  Left end of initial interval.
   * @param bx0  Right end of initial interval.
   * @return     Abscissa of the minimum.
   */
  template <typename FC>
  constexpr value_type minimize_fn(FC fc, value_type ax0, value_type bx0);

private:
  // Binds this instance's bracket [ax, bx, cx] and tolerances to the dbrent
  // algorithm.
  template <typename FC> constexpr value_type dbrent_impl(FC &fc) {
    return exprmin::dbrent(fc, ax, bx, cx, tol, Base::ZEPS, Base::ITMAX);
  }
};

template <diff::CExpression Expr>
constexpr typename Dbrent<Expr>::value_type Dbrent<Expr>::minimize() {
  struct Funcd {
    Dbrent &self;
    constexpr value_type operator()(value_type t) { return self.eval_at(t); }
    constexpr value_type df(value_type t) {
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
