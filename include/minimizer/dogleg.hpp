#pragma once

#include "detail.hpp"
#include "trustregionbase.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>

namespace exprmin {

namespace mp = boost::mp11;

enum class HessianMode { BFGS, ExactAD };

// Powell dogleg trust-region minimizer for scalar f(x).
//
// HM == BFGS (default): Hessian approximated via rank-2 BFGS updates.
// HM == ExactAD:        exact Hessian recomputed every iteration via
//                       diff::derivative_tensor<2>(expr).
//
// Inherits the minimize() loop from TrustRegionBase; provides the five
// CRTP hooks: eval_state, eval_trial, compute_step, adjust_tr, commit_state,
// and refresh_hessian (ExactAD only).
template <diff::CExpression Expr, HessianMode HM = HessianMode::BFGS>
struct Dogleg
    : TrustRegionBase<Dogleg<Expr, HM>,
                      typename Expr::value_type,
                      static_cast<int>(
                          mp::mp_size<diff::extract_symbols_from_expr_t<Expr>>::value)> {

  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr int N = static_cast<int>(mp::mp_size<Syms>::value);
  using Base   = TrustRegionBase<Dogleg<Expr, HM>, value_type, N>;
  using Point  = Eigen::Vector<value_type, N>;
  using Matrix = Eigen::Matrix<value_type, N, N>;

  // Expose ParamVec / NMat aliases consistent with the rest of the library.
  using ParamVec = Point;
  using NMat     = Matrix;

  // Bring iter and get_optimal_value into scope.
  using Base::iter;
  using Base::get_optimal_value;

  constexpr value_type operator()(const Point& p) { return eval_at(p); }
  constexpr value_type eval_at(const Point& p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  constexpr std::pair<value_type, Point> eval_grad(const Point& p) {
    expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(expr);
    return {expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  constexpr explicit Dogleg(Expr e,
                             value_type tol_           = value_type{1e-8},
                             int        itmax_         = 200,
                             value_type trustregion0_  = value_type{1e3},
                             value_type trustregion_min_ = value_type{1e-12})
      : Base{tol_, itmax_, trustregion0_, trustregion_min_},
        expr{std::move(e)} {}

  // ── CRTP hooks ────────────────────────────────────────────────────────────

  constexpr std::tuple<value_type, Point, Matrix> eval_state(const Point& p) {
    auto [f, g] = eval_grad(p);
    return {f, g, Matrix::Identity()};
  }

  constexpr value_type eval_trial(const Point& p_new) {
    auto [f, g] = eval_grad(p_new);
    g_new_ = g;
    return f;
  }

  // ExactAD: recompute B from current p at each iteration start.
  constexpr void refresh_hessian(const Point& p, Matrix& B) {
    if constexpr (HM == HessianMode::ExactAD) {
      expr.update(Syms{}, p);
      const auto H = diff::derivative_tensor<2>(expr);
      B = Eigen::Map<const Eigen::Matrix<value_type, N, N, Eigen::RowMajor>>(
          &H[0][0]);
    }
  }

  // Three-way dogleg: Cauchy / interpolated / Newton. Stores nn_ for adjust_tr.
  constexpr Point compute_step(const Point& g, const Matrix& B,
                                value_type delta) {
    using std::sqrt;

    const value_type gBg = g.dot(B * g);
    const value_type kc =
        (gBg > value_type{0}) ? (-g.squaredNorm() / gBg) : value_type{-1};
    const Point      p_c = kc * g;
    const value_type nc  = p_c.norm();

    const Point      p_n = -(B.ldlt().solve(g));
    nn_                  = p_n.norm();

    if (nn_ <= delta) {
      return p_n;
    } else if (nc >= delta) {
      return (delta / nc) * p_c;
    } else {
      const Point      d    = p_n - p_c;
      const value_type l2   = d.squaredNorm();
      const value_type c    = d.dot(p_c);
      value_type disc = c * c - l2 * (nc * nc - delta * delta);
      if (disc < value_type{0}) disc = value_type{0};
      return p_c + ((-c + sqrt(disc)) / l2) * d;
    }
  }

  // libdogleg ρ-policy: drop delta to Newton size before shrinking.
  constexpr void adjust_tr(value_type& delta, value_type rho,
                            bool at_boundary) {
    if (rho < Base::TR_DOWN_THRESHOLD) {
      if (!at_boundary) delta = nn_;
      delta *= Base::TR_DOWN_FACTOR;
    } else if (rho > Base::TR_UP_THRESHOLD && at_boundary) {
      delta *= Base::TR_UP_FACTOR;
    }
  }

  // BFGS rank-2 update: B ← B - (Bs)(Bs)ᵀ/(sᵀBs) + yyᵀ/(yᵀs).
  constexpr std::pair<Point, Matrix>
  commit_state(const Point& step, const Point& g_old, const Matrix& B_cur) {
    Matrix B_new = B_cur;
    if constexpr (HM == HessianMode::BFGS) {
      const Point      Bs  = B_new * step;
      const value_type sBs = step.dot(Bs);
      const Point      dg  = g_new_ - g_old;
      const value_type ys  = dg.dot(step);
      if (ys > value_type{0} && sBs > value_type{0})
        B_new += (dg * dg.transpose()) / ys - (Bs * Bs.transpose()) / sBs;
    }
    // ExactAD: B_new is returned unchanged; refresh_hessian rewrites it next iter.
    return {g_new_, B_new};
  }

private:
  Expr         expr;
  Point        g_new_{};
  value_type   nn_{};
};

template <diff::CExpression Expr> Dogleg(Expr) -> Dogleg<Expr>;
template <diff::CExpression Expr, typename T> Dogleg(Expr, T) -> Dogleg<Expr>;

} // namespace exprmin
