#pragma once

#include "lsq_base.hpp"
#include <cmath>

namespace exprmin {

// NR §15.5 — Levenberg-Marquardt nonlinear least-squares fitting.
//
// Minimizes χ² = Σᵢ [wᵢ (yᵢ − f(xᵢ; a))]²  over parameters a ∈ ℝᴺ.
//
// Adds Marquardt damping to the normal equations each iteration:
//   α = JᵀJ + λ·diag(JᵀJ)
// λ grows when a step is rejected and shrinks when accepted,
// blending steepest-descent (large λ) with Gauss-Newton (λ → 0).
template <diff::CExpression Expr,
          typename ParamSyms = diff::extract_symbols_from_expr_t<Expr>,
          typename InputSyms = mp::mp_list<>>
struct LevenbergMarquardt : LeastSquaresBase<Expr, ParamSyms, InputSyms> {
  using Base = LeastSquaresBase<Expr, ParamSyms, InputSyms>;
  using typename Base::DataPoint;
  using typename Base::ParamVec;
  using typename Base::value_type;

  static constexpr value_type LAMBDA_INIT{1e-3};
  static constexpr value_type LAMBDA_UP{10};
  static constexpr value_type LAMBDA_DOWN{0.1};

  constexpr explicit LevenbergMarquardt(Expr e,
                                        value_type ftol_ = value_type{1e-8},
                                        int itmax_ = 1000)
      : Base(std::move(e)), ftol(ftol_), itmax(itmax_) {}
  constexpr ParamVec fit(ParamVec params, const std::vector<DataPoint> &data);

private:
  value_type ftol;
  int itmax;
};

template <diff::CExpression Expr, typename ParamSyms, typename InputSyms>
constexpr typename LevenbergMarquardt<Expr, ParamSyms, InputSyms>::ParamVec
LevenbergMarquardt<Expr, ParamSyms, InputSyms>::fit(
    ParamVec params, const std::vector<DataPoint> &data) {
  using std::abs;
  constexpr int Ni = static_cast<int>(Base::N);
  using NMat = Eigen::Matrix<value_type, Ni, Ni>;
  using NVec = Eigen::Vector<value_type, Ni>;

  value_type lambda = LAMBDA_INIT;
  auto [r, J] = this->eval_rJ(params, data);
  value_type chi2 = r.squaredNorm();

  for (int iter = 0; iter < itmax; ++iter) {
    const NMat JtJ = J.transpose() * J;
    const NVec beta = -(J.transpose() * r); // descent direction: −Jᵀr

    // Marquardt damping: α = JᵀJ + λ·diag(JᵀJ)
    NMat alpha = JtJ;
    alpha.diagonal().array() *= (value_type{1} + lambda);

    const NVec da = alpha.ldlt().solve(beta);
    const ParamVec p_new = params + da;

    auto [r_new, J_new] = this->eval_rJ(p_new, data);
    const value_type chi2_new = r_new.squaredNorm();
    if (chi2_new < chi2) {
      lambda *= LAMBDA_DOWN;
      if ((da.norm() < ftol * (params.norm() + ftol)) ||
          (abs(chi2 - chi2_new) < ftol * (value_type{1} + chi2))) {
        return p_new;
      }
      params = p_new;
      chi2 = chi2_new;
      r = std::move(r_new);
      J = std::move(J_new);
    } else {
      lambda *= LAMBDA_UP;
    }
  }
  return params;
}

template <diff::CExpression Expr>
LevenbergMarquardt(Expr) -> LevenbergMarquardt<Expr>;
template <diff::CExpression Expr, diff::Numeric T>
LevenbergMarquardt(Expr, T) -> LevenbergMarquardt<Expr>;

template <diff::CExpression Expr> using LM = LevenbergMarquardt<Expr>;

} // namespace exprmin
