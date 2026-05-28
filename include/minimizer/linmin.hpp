#pragma once

#include "brent.hpp"
#include "expressions.hpp"
#include "gradient.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>

namespace exprmin {

namespace mp = boost::mp11;

/**
 * @brief 1-D line minimizer for N-dimensional expressions using Brent's method.
 *
 * Given a point @p p and direction @p dir in ℝᴺ, minimizes f(p + t·dir) over
 * the scalar step @c t.  After minimize(p, dir):
 *   - @p p is updated to the minimizing point p + t_min·dir.
 *   - @p dir is scaled in-place by t_min (NR convention, so the caller
 *     can accumulate net displacement).
 *   - get_optimal_value() returns f(p_min).
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 */
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
  /**
   * @brief Constructs a LinMin wrapping the given expression.
   * @param e     Expression to minimize.
   * @param tol_  Brent tolerance passed through to the underlying line search
   *              (default 3×10⁻⁸).
   */
  constexpr explicit LinMin(Expr e,
                            value_type tol_ = static_cast<value_type>(3.0e-8))
      : ls(std::move(e), tol_), tol(tol_) {}

  /// @brief Callable interface delegating to eval_at.
  constexpr value_type operator()(const Point &p) { return ls.eval_at(p); }

  /**
   * @brief Evaluates the expression at point @p p.
   * @param p  N-dimensional input point.
   * @return   f(p).
   */
  constexpr value_type eval_at(const Point &p) { return ls.eval_at(p); }

  /// @brief Returns f at the minimizing point after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fret; }

  /**
   * @brief Evaluates the expression and its reverse-mode gradient at @p p.
   * @param p  N-dimensional input point.
   * @return   {f(p), ∇f(p)} as an std::pair.
   */
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    ls.expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(ls.expr);
    return {ls.expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  /**
   * @brief Line-minimizes f(p + t·dir) over t using Brent's method.
   *
   * Brackets from t=0 to t=1, then refines with Brent.  On return:
   *   - @p p   is the new point p + t_min·dir.
   *   - @p dir is scaled by t_min (NR convention).
   *
   * @param p    Current point (in/out).
   * @param dir  Search direction (in/out, scaled by t_min on exit).
   */
  constexpr void minimize(Point &p, Eigen::Ref<Point> dir) {
    // Step 1: lift the N-D function to a 1-D scalar along the ray p + t·dir.
    auto f1d = [&](value_type t) -> value_type {
      return ls.eval_at(Point{p + t * dir});
    };
    // Step 2: bracket on [0, 1] and refine with Brent to find t_min.
    const value_type t_min = ls.minimize_fn(f1d, value_type{0}, value_type{1});
    // Step 3: scale dir by t_min (NR convention) and advance p.
    dir *= t_min;
    p += dir;
    // Step 4: record f at the new point.
    fret = eval_at(p);
  }
};

template <diff::CExpression Expr> LinMin(Expr) -> LinMin<Expr>;
template <diff::CExpression Expr, typename T> LinMin(Expr, T) -> LinMin<Expr>;

/**
 * @brief Derivative-aware line minimizer using Dbrent (secant on f′ via AD).
 *
 * Drop-in replacement for LinMin that uses Dbrent instead of Brent.
 * The directional derivative df/dt = ∇f(p + t·d)·d is computed via
 * reverse-mode automatic differentiation at each Dbrent step, so no
 * explicit gradient storage is needed by the caller.
 *
 * Usable as a template-template argument wherever LinMin is accepted
 * (e.g. Frprmn).
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 */
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
  /**
   * @brief Constructs a DLinMin wrapping the given expression.
   * @param e     Expression to minimize.
   * @param tol_  Dbrent tolerance (default 3×10⁻⁸).
   */
  constexpr explicit DLinMin(Expr e,
                             value_type tol_ = static_cast<value_type>(3.0e-8))
      : ls(std::move(e), tol_), tol(tol_) {}

  /**
   * @brief Evaluates the expression at point @p p.
   * @param p  N-dimensional input point.
   * @return   f(p).
   */
  constexpr value_type eval_at(const Point &p) { return ls.eval_at(p); }

  /// @brief Returns f at the minimizing point after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fret; }

  /**
   * @brief Evaluates the expression and its reverse-mode gradient at @p p.
   * @param p  N-dimensional input point.
   * @return   {f(p), ∇f(p)} as an std::pair.
   */
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    ls.expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(ls.expr);
    return {ls.expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  /**
   * @brief Line-minimizes f(p + t·dir) over t using Dbrent.
   *
   * Constructs a local functor exposing both operator()(t) = f(p₀ + t·d₀)
   * and df(t) = ∇f(p₀ + t·d₀)·d₀ via reverse-mode AD, then delegates to
   * Dbrent::minimize_fn.  On return @p p and @p dir are updated identically
   * to LinMin::minimize.
   *
   * @param p    Current point (in/out).
   * @param dir  Search direction (in/out, scaled by t_min on exit).
   */
  constexpr void minimize(Point &p, Eigen::Ref<Point> dir) {
    // Step 1: snapshot base point and direction so the 1-D functor captures
    //         fixed p0/d0 while p and dir are mutated below.
    const Point p0 = p;
    const Point d0 = dir;
    // Step 2: build a local functor exposing f(t) and df/dt = ∇f·d0 for Dbrent.
    struct FC {
      DLinMin &self;
      const Point &p0_, &d0_;
      constexpr value_type operator()(value_type t) const {
        return self.ls.eval_at(Point{p0_ + t * d0_});
      }
      // directional derivative df/dt = ∇f · d0 computed via reverse AD
      constexpr value_type df(value_type t) const {
        self.ls.expr.update(Syms{}, p0_ + t * d0_);
        const auto g = diff::gradient<diff::DiffMode::Reverse>(self.ls.expr);
        return Eigen::Map<const Point>(g.data()).dot(d0_);
      }
    } fc{*this, p0, d0};
    // Step 3: bracket on [0, 1] and refine with Dbrent (uses both f and df).
    const value_type t_min = ls.minimize_fn(fc, value_type{0}, value_type{1});
    // Step 4: scale dir by t_min (NR convention) and advance p.
    dir *= t_min;
    p += dir;
    // Step 5: record f at the new point.
    fret = eval_at(p);
  }
};

template <diff::CExpression Expr> DLinMin(Expr) -> DLinMin<Expr>;
template <diff::CExpression Expr, typename T> DLinMin(Expr, T) -> DLinMin<Expr>;

} // namespace exprmin
