#pragma once

#include "bfgs.hpp"
#include "detail.hpp"
#include <utility>

namespace exprmin {

// Nocedal §7.2 — Limited-memory BFGS.
//
// Replaces the N×N inverse-Hessian with the M most-recent (sₖ,yₖ) curvature
// pairs and the two-loop recursion.  Storage O(M·N) vs O(N²).
// Line search selected by LS1D (same template-template as BFGS).
template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = Brent, int M = 10>
struct LBFGS : QuasiNewtonBase<Expr, LS1D> {
  using Base = QuasiNewtonBase<Expr, LS1D>;
  using typename Base::Point;
  using typename Base::value_type;
  static constexpr int ITMAX = 200;

  constexpr explicit LBFGS(Expr e,
                           value_type gtol_ = static_cast<value_type>(1e-8))
      : Base(std::move(e), gtol_) {}

  constexpr Point minimize(Point p) {
    const auto eg = [this](const Point &q) { return this->eval_grad(q); };
    auto ls_fn = this->make_ls_fn();
    detail::LBFGSDirState<value_type, static_cast<int>(Base::N), M> ds;
    p = detail::qn_impl<value_type, static_cast<int>(Base::N)>(
        eg, std::move(p), this->gtol, ITMAX, ls_fn, ds, this->iter);
    this->fret = this->eval_at(p);
    return p;
  }
};

template <diff::CExpression Expr> LBFGS(Expr) -> LBFGS<Expr>;
template <diff::CExpression Expr, typename T> LBFGS(Expr, T) -> LBFGS<Expr>;

} // namespace exprmin
