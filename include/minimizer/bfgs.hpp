#pragma once

#include "brent.hpp"
#include "detail.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <type_traits>
#include <utility>

namespace exprmin {

namespace mp = boost::mp11;

// ── Armijo<Expr> ─────────────────────────────────────────────────────────────
// Backtracking sufficient-decrease line search.  Owns the expression for
// eval_at; minimize_fn takes any 1D callable + (fp, slope) and returns t_min.
template <diff::CExpression Expr> struct Armijo {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

  Expr expr;
  value_type fret{};
  const value_type tol;

  constexpr explicit Armijo(Expr e,
                            value_type tol_ = static_cast<value_type>(1e-8))
      : expr(std::move(e)), tol(tol_) {}

  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  template <std::invocable<value_type> F>
  constexpr value_type minimize_fn(F f1d, value_type fp,
                                   value_type slope) const {
    constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();
    constexpr value_type C1{1e-4};
    value_type alpha{1};
    for (int k = 0; k < 40 && alpha > EPS; ++k, alpha *= value_type{0.5})
      if (f1d(alpha) <= fp + C1 * alpha * slope)
        break;
    return alpha;
  }
};
template <diff::CExpression Expr> Armijo(Expr) -> Armijo<Expr>;
template <diff::CExpression Expr, typename T> Armijo(Expr, T) -> Armijo<Expr>;

// ── QuasiNewtonBase<Expr, LS1D>
// ─────────────────────────────────────────────── Shared base for BFGS and
// LBFGS: owns LS1D<Expr> ls, provides eval_at / eval_grad through ls.expr, and
// builds the N-D line search callable.
template <diff::CExpression Expr, template <diff::CExpression> class LS1D>
struct QuasiNewtonBase {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

protected:
  LS1D<Expr> ls;
  value_type fret{};
  int iter{};
  const value_type gtol;

public:
  constexpr explicit QuasiNewtonBase(Expr e, value_type gtol_)
      : ls(std::move(e), gtol_), gtol(gtol_) {}

  constexpr value_type eval_at(const Point &p) { return ls.eval_at(p); }
  constexpr value_type operator()(const Point &p) { return eval_at(p); }
  constexpr value_type get_optimal_value() const { return fret; }
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    ls.expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(ls.expr);
    return {ls.expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  // Returns a line search callable (xc, xi, fp, slope) → dx for
  // quasi_newton_impl. Dispatches to the owned ls.minimize_fn; Armijo needs
  // fp/slope, Brent/Dbrent do bracket + 1D minimization.
  constexpr auto make_line_search_fn() {
    return [this](const Point &xc, const Point &xi, value_type fp,
                  value_type slope) -> Point {
      auto f1d = [this, &xc, &xi](value_type t) {
        ls.expr.update(Syms{}, xc + t * xi);
        return ls.expr.eval();
      };
      value_type t_min;
      if constexpr (std::is_same_v<LS1D<Expr>, Armijo<Expr>>) {
        t_min = ls.minimize_fn(f1d, fp, slope);
      } else if constexpr (std::is_same_v<LS1D<Expr>, Dbrent<Expr>>) {
        struct FC {
          decltype(f1d) &f_;
          const Point &xi_;
          LS1D<Expr> &ls_;
          const Point &xc_;
          value_type operator()(value_type t) { return f_(t); }
          value_type df(value_type t) {
            ls_.expr.update(Syms{}, xc_ + t * xi_);
            const auto g = diff::gradient<diff::DiffMode::Reverse>(ls_.expr);
            return Eigen::Map<const Point>(g.data()).dot(xi_);
          }
        } fc{f1d, xi, ls, xc};
        t_min = ls.minimize_fn(fc, value_type{0}, value_type{1});
      } else {
        // Brent (default)
        t_min = ls.minimize_fn(f1d, value_type{0}, value_type{1});
      }
      return t_min * xi;
    };
  }
};

// ── BFGS<Expr, LS1D> ─────────────────────────────────────────────────────────
// NR §10.7 — quasi-Newton with full N×N inverse-Hessian approximation.
// LS1D selects the line search policy: Brent (default), Dbrent, or Armijo.
template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = Brent>
struct BFGS : QuasiNewtonBase<Expr, LS1D> {
  using Base = QuasiNewtonBase<Expr, LS1D>;
  using typename Base::Point;
  using typename Base::value_type;
  static constexpr int ITMAX = 200;

  constexpr explicit BFGS(Expr e,
                          value_type gtol_ = static_cast<value_type>(1e-8))
      : Base(std::move(e), gtol_) {}

  constexpr Point minimize(Point p) {
    const auto eg = [this](const Point &q) { return this->eval_grad(q); };
    auto ls_fn = this->make_line_search_fn();
    detail::BFGSDirState<value_type, static_cast<int>(Base::N)> ds;
    p = detail::quasi_newton_impl<value_type, static_cast<int>(Base::N)>(
        eg, std::move(p), this->gtol, ITMAX, ls_fn, ds, this->iter);
    this->fret = this->eval_at(p);
    return p;
  }
};

template <diff::CExpression Expr> BFGS(Expr) -> BFGS<Expr>;
template <diff::CExpression Expr, typename T> BFGS(Expr, T) -> BFGS<Expr>;

// Convenience aliases
template <diff::CExpression Expr> using DBFGS = BFGS<Expr, Dbrent>;
template <diff::CExpression Expr> using ABFGS = BFGS<Expr, Armijo>;

} // namespace exprmin
