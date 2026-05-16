#pragma once

#include "detail.hpp"
#include "expressions.hpp"
#include "gradient.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <limits>

namespace exprmin {

namespace mp = boost::mp11;

// 1D line minimizer for N-dimensional expressions.
//
// Given a point p and direction dir in R^N, minimizes f(p + t·dir) over
// scalar t using bracket + Brent.  After minimize(p, dir):
//   - p   is updated to the minimum point
//   - dir is scaled by the optimal step t_min  (NR convention)
//   - fret holds f(p_min)
template <diff::CExpression Expr> struct LinMin {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

  static constexpr value_type ZEPS =
      std::numeric_limits<value_type>::epsilon() *
      static_cast<value_type>(1.0e-3);
  static constexpr int ITMAX = 100;

  Expr expr;
  value_type fret{};
  const value_type tol;

  constexpr explicit LinMin(Expr e,
                            value_type tol_ = static_cast<value_type>(3.0e-8))
      : expr(std::move(e)), tol(tol_) {}

  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  constexpr void minimize(Point &p, Eigen::Ref<Point> dir) {
    auto f1 = [&](const value_type &t) -> value_type {
      return eval_at(p + t * dir);
    };

    value_type ax{0}, bx{1}, cx;
    value_type fa = f1(ax), fb = f1(bx), fc;
    detail::bracket(f1, ax, bx, cx, fa, fb, fc);
    const value_type xmin = detail::brent(f1, ax, bx, cx, tol, ZEPS, ITMAX);

    dir *= xmin;
    p += dir;
    fret = eval_at(p);
  }
};

template <diff::CExpression Expr> LinMin(Expr) -> LinMin<Expr>;
template <diff::CExpression Expr, typename T> LinMin(Expr, T) -> LinMin<Expr>;

// Derivative-aware line minimizer — uses detail::dbrent (secant on f′).
// The directional derivative df/dt = ∇f(p + t·dir) · dir is computed via
// reverse-mode AD.  Drop-in replacement for LinMin.
template <diff::CExpression Expr> struct DLinMin : LinMin<Expr> {
  using Base = LinMin<Expr>;
  using Base::Base; // inherit constructors
  using value_type = typename Base::value_type;
  using Syms = typename Base::Syms;
  using Point = typename Base::Point;

  constexpr void minimize(Point &p, Eigen::Ref<Point> dir) {
    const Point p0 = p;
    const Point d0 = dir;

    struct Funcd {
      DLinMin &self;
      const Point &p0;
      const Point &d0;
      value_type operator()(value_type t) {
        self.expr.update(Syms{}, p0 + t * d0);
        return self.expr.eval();
      }
      value_type df(value_type t) {
        self.expr.update(Syms{}, p0 + t * d0);
        const auto g = diff::gradient<diff::DiffMode::Reverse>(self.expr);
        return Eigen::Map<const Point>(g.data()).dot(d0);
      }
    };
    Funcd fc{*this, p0, d0};

    value_type ax{0}, bx{1}, cx;
    value_type fa = fc(ax), fb = fc(bx), fc_val;
    detail::bracket(fc, ax, bx, cx, fa, fb, fc_val);
    const value_type xmin =
        detail::dbrent(fc, ax, bx, cx, this->tol, Base::ZEPS, Base::ITMAX);

    dir *= xmin;
    p += dir;
    this->fret = this->eval_at(p);
  }
};

template <diff::CExpression Expr> DLinMin(Expr) -> DLinMin<Expr>;
template <diff::CExpression Expr, typename T> DLinMin(Expr, T) -> DLinMin<Expr>;

} // namespace exprmin
