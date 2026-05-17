#pragma once

#include "lsq_base.hpp"

namespace exprmin {

// Gauss-Newton nonlinear least-squares fitting.
//
// Minimizes χ² = Σᵢ [wᵢ (yᵢ − f(xᵢ; a))]²  over parameters a ∈ ℝᴺ.
//
// Solves the normal equations without damping each iteration:
//   (JᵀJ) · step = Jᵀ r,   params += step
// Converges quadratically near the solution; may diverge far from it.
// Prefer LevenbergMarquardt when the initial guess is poor.
template <diff::CExpression Expr,
          typename ParamSyms = diff::extract_symbols_from_expr_t<Expr>,
          typename InputSyms = mp::mp_list<>>
struct GaussNewton : LeastSquaresBase<Expr, ParamSyms, InputSyms> {
  using Base = LeastSquaresBase<Expr, ParamSyms, InputSyms>;
  using Base::eval_rJ;
  using typename Base::DataPoint;
  using typename Base::ParamVec;
  using typename Base::value_type;
  constexpr explicit GaussNewton(Expr e, value_type ftol_ = value_type{1e-8},
                                 int itmax_ = 1000)
      : Base(std::move(e)), ftol(ftol_), itmax(itmax_) {}
  constexpr ParamVec fit(ParamVec params, const std::vector<DataPoint> &data);

private:
  value_type ftol;
  const int itmax;
};

template <diff::CExpression Expr, typename ParamSyms, typename InputSyms>
constexpr typename GaussNewton<Expr, ParamSyms, InputSyms>::ParamVec
GaussNewton<Expr, ParamSyms, InputSyms>::fit(
    ParamVec params, const std::vector<DataPoint> &data) {
  constexpr int Ni = static_cast<int>(Base::N);
  using NVec = Eigen::Vector<value_type, Ni>;

  for (int iter = 0; iter < itmax; ++iter) {
    auto [r, J] = eval_rJ(params, data);
    const NVec step =
        (J.transpose() * J)
            .ldlt()
            .solve(-(J.transpose() * r)); // J has negated sign convention
    params += step;
    if (step.norm() < ftol * (params.norm() + ftol)) {
      break;
    }
  }
  return params;
}

template <diff::CExpression Expr> GaussNewton(Expr) -> GaussNewton<Expr>;
template <diff::CExpression Expr, typename T>
GaussNewton(Expr, T) -> GaussNewton<Expr>;

} // namespace exprmin
