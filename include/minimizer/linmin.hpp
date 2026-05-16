#pragma once

#include "brent.hpp"
#include "expressions.hpp"
#include "gradient.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>

namespace exprmin {

namespace mp = boost::mp11;

// 1D line minimizer for N-dimensional expressions using Brent's method.
//
// Given a point p and direction dir in R^N, minimizes f(p + t·dir) over
// scalar t.  After minimize(p, dir):
//   - p   is updated to the minimum point
//   - dir is scaled by the optimal step t_min  (NR convention)
//   - fret holds f(p_min)
template <diff::CExpression Expr> struct LinMin {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

private:
  static constexpr int ITMAX = 100;

  Brent<Expr> ls;
  value_type fret{};
  const value_type tol;

public:
  constexpr explicit LinMin(Expr e,
                            value_type tol_ = static_cast<value_type>(3.0e-8))
      : ls(std::move(e), tol_), tol(tol_) {}

  constexpr value_type operator()(const Point &p) { return ls.eval_at(p); }
  constexpr value_type eval_at(const Point &p) { return ls.eval_at(p); }
  constexpr value_type get_optimal_value() const { return fret; }
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    ls.expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(ls.expr);
    return {ls.expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  constexpr void minimize(Point &p, Eigen::Ref<Point> dir) {
    auto f1d = [&](value_type t) -> value_type {
      return ls.eval_at(Point{p + t * dir});
    };
    const value_type t_min = ls.minimize_fn(f1d, value_type{0}, value_type{1});
    dir *= t_min;
    p += dir;
    fret = eval_at(p);
  }
};

template <diff::CExpression Expr> LinMin(Expr) -> LinMin<Expr>;
template <diff::CExpression Expr, typename T> LinMin(Expr, T) -> LinMin<Expr>;

// Derivative-aware line minimizer using Dbrent (secant on f′ via reverse AD).
// Drop-in replacement for LinMin; usable as template-template arg in Frprmn.
template <diff::CExpression Expr> struct DLinMin {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

  static constexpr int ITMAX = 100;

private:
  Dbrent<Expr> ls;
  value_type fret{};
  const value_type tol;

public:
  constexpr explicit DLinMin(Expr e,
                             value_type tol_ = static_cast<value_type>(3.0e-8))
      : ls(std::move(e), tol_), tol(tol_) {}

  constexpr value_type eval_at(const Point &p) { return ls.eval_at(p); }
  constexpr value_type get_optimal_value() const { return fret; }
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    ls.expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(ls.expr);
    return {ls.expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  constexpr void minimize(Point &p, Eigen::Ref<Point> dir) {
    const Point p0 = p;
    const Point d0 = dir;
    struct FC {
      DLinMin &self;
      const Point &p0_, &d0_;
      constexpr value_type operator()(value_type t) const {
        return self.ls.eval_at(Point{p0_ + t * d0_});
      }
      constexpr value_type df(value_type t) const {
        self.ls.expr.update(Syms{}, p0_ + t * d0_);
        const auto g = diff::gradient<diff::DiffMode::Reverse>(self.ls.expr);
        return Eigen::Map<const Point>(g.data()).dot(d0_);
      }
    } fc{*this, p0, d0};
    const value_type t_min = ls.minimize_fn(fc, value_type{0}, value_type{1});
    dir *= t_min;
    p += dir;
    fret = eval_at(p);
  }
};

template <diff::CExpression Expr> DLinMin(Expr) -> DLinMin<Expr>;
template <diff::CExpression Expr, typename T> DLinMin(Expr, T) -> DLinMin<Expr>;

} // namespace exprmin
