#pragma once

#include "bracketmethod.hpp"
#include "detail.hpp"
#include "gradient.hpp"
#include <limits>

namespace exprmin {

// NR §10.3 — Brent's method: parabolic interpolation with golden-section fallback.
template <diff::CExpression Expr>
struct Brent : Bracketmethod<Expr> {
  using Base = Bracketmethod<Expr>;
  using value_type = typename Base::value_type;
  using Syms = typename Base::Syms;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;
  using Base::ax;
  using Base::bx;
  using Base::cx;
  using Base::bracket;
  using Base::eval_at;

  static constexpr value_type ZEPS =
      std::numeric_limits<value_type>::epsilon() * static_cast<value_type>(1.0e-3);
  static constexpr int ITMAX = 100;

  value_type xmin{};
  value_type fmin{};
  const value_type tol;

  constexpr explicit Brent(Expr e,
                           value_type tol_ = static_cast<value_type>(3.0e-8))
      : Base(std::move(e)), tol(tol_) {}

  // N-D point evaluation — used when Brent<Expr> is owned by BFGS / LBFGS / LinMin.
  constexpr value_type eval_at(const Point &p) {
    this->expr.update(Syms{}, p);
    return this->expr.eval();
  }

  // 1D-only overloads: only valid for single-variable expressions.
  constexpr value_type minimize() requires (N == 1) {
    auto f = [this](const value_type &x) { return eval_at(x); };
    xmin = detail::brent(f, ax, bx, cx, tol, ZEPS, ITMAX);
    fmin = eval_at(xmin);
    return xmin;
  }

  constexpr value_type minimize(const value_type &ax0,
                                const value_type &bx0) requires (N == 1) {
    bracket(ax0, bx0);
    return minimize();
  }

  // Minimize any 1D callable — used by LinMin / BFGS / LBFGS.
  template <std::invocable<value_type> F>
  constexpr value_type minimize_fn(F f1d, value_type ax0, value_type bx0) {
    value_type a = ax0, b = bx0, c;
    value_type fa = f1d(a), fb = f1d(b), fc;
    detail::bracket(f1d, a, b, c, fa, fb, fc);
    xmin = detail::brent(f1d, a, b, c, tol, ZEPS, ITMAX);
    fmin = f1d(xmin);
    return xmin;
  }
};

template <diff::CExpression Expr> Brent(Expr) -> Brent<Expr>;
template <diff::CExpression Expr, typename T> Brent(Expr, T) -> Brent<Expr>;

// NR §10.4 — Brent + first-derivative (secant on f′ via reverse-mode AD).
// Inherits all state from Brent; only minimize() differs.
template <diff::CExpression Expr>
struct Dbrent : Brent<Expr> {
  using Base = Brent<Expr>;
  using Base::Base; // inherit constructors
  using value_type = typename Base::value_type;
  using Syms = typename Base::Syms;

  constexpr value_type minimize() {
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
    this->xmin = detail::dbrent(fc, this->ax, this->bx, this->cx,
                                this->tol, Base::ZEPS, Base::ITMAX);
    this->fmin = this->eval_at(this->xmin);
    return this->xmin;
  }

  constexpr value_type minimize(const value_type &ax0, const value_type &bx0) {
    this->bracket(ax0, bx0);
    return minimize();
  }

  // Minimize any 1D callable with derivative.  FC must expose operator()(T)
  // and df(T).  Used by DLinMin / BFGS<Dbrent> / LBFGS<Dbrent>.
  template <typename FC>
  constexpr value_type minimize_fn(FC fc, value_type ax0, value_type bx0) {
    value_type a = ax0, b = bx0, c;
    value_type fa = fc(a), fb = fc(b), fc_val;
    detail::bracket(fc, a, b, c, fa, fb, fc_val);
    this->xmin = detail::dbrent(fc, a, b, c, this->tol, Base::ZEPS, Base::ITMAX);
    this->fmin = fc(this->xmin);
    return this->xmin;
  }
};

template <diff::CExpression Expr> Dbrent(Expr) -> Dbrent<Expr>;
template <diff::CExpression Expr, typename T> Dbrent(Expr, T) -> Dbrent<Expr>;

} // namespace exprmin
