#pragma once

#include "detail.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <utility>

namespace exprmin {

namespace mp = boost::mp11;

enum class LineSearch { Brent, Dbrent, Armijo };

// NR §10.7 — BFGS quasi-Newton minimization.
//
// Maintains an N×N approximation to the inverse Hessian, updated via the
// rank-2 BFGS formula after each step.  Line search is selected at compile
// time via the LS parameter:
//   LineSearch::Brent  — bracket + Brent 1-D exact minimization (default)
//   LineSearch::Dbrent — derivative-aware Dbrent (faster on smooth problems)
//   LineSearch::Armijo — backtracking Armijo (robust, no bracket needed)
// Convergence criterion: max |∂f/∂xᵢ| · max(|xᵢ|,1) / max(|f|,1) < gtol.
template <diff::CExpression Expr, LineSearch LS = LineSearch::Brent>
struct BFGS {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

  static constexpr int ITMAX = 200;

  Expr expr;
  value_type fret{};
  int iter{};
  const value_type gtol;

  constexpr explicit BFGS(Expr e,
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
    const auto eg = [this](const Point &q) -> std::pair<value_type, Point> {
      return eval_grad(q);
    };
    if constexpr (LS == LineSearch::Armijo) {
      p = detail::bfgs_armijo<value_type, static_cast<int>(N)>(eg, std::move(p),
                                                               gtol, ITMAX);
    } else if constexpr (LS == LineSearch::Dbrent) {
      p = detail::bfgs_dbrent<value_type, static_cast<int>(N)>(eg, std::move(p),
                                                               gtol, ITMAX);
    } else {
      p = detail::bfgs_brent<value_type, static_cast<int>(N)>(eg, std::move(p),
                                                              gtol, ITMAX);
    }
    fret = eval_at(p);
    return p;
  }
};

template <diff::CExpression Expr> BFGS(Expr) -> BFGS<Expr>;
template <diff::CExpression Expr, typename T> BFGS(Expr, T) -> BFGS<Expr>;

// Convenience aliases
template <diff::CExpression Expr> using DBFGS = BFGS<Expr, LineSearch::Dbrent>;
template <diff::CExpression Expr> using ABFGS = BFGS<Expr, LineSearch::Armijo>;

} // namespace exprmin
