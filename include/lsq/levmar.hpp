#pragma once

#include "lsq_base.hpp"
#include "../callback/callback.hpp"
#include <cmath>

namespace exprmin {

/**
 * @brief Levenberg-Marquardt nonlinear least-squares solver
 *        (Nocedal & Wright §10.3; NR §15.5).
 *
 * Minimises χ² = ½ Σᵢ [wᵢ (yᵢ − f(xᵢ; a))]² over parameters a ∈ ℝᴺ by
 * iterating the *damped normal equations*:
 *
 *   (JᵀJ + λ·diag(JᵀJ)) δa = −Jᵀr                          (N&W eq. 10.24)
 *
 * The scalar λ interpolates between Gauss-Newton (λ → 0, fast quadratic
 * convergence near the solution) and scaled steepest descent (λ → ∞, robust
 * far from the solution).  λ is multiplied by LAMBDA_DOWN after an accepted
 * step and by LAMBDA_UP after a rejected one, following the heuristic of
 * Marquardt (1963).
 *
 * @tparam Expr      Expression satisfying diff::CExpression; defines the model
 * f.
 * @tparam ParamSyms Symbols treated as free parameters (default: all in Expr).
 * @tparam InputSyms Symbols treated as per-data-point inputs (default: none).
 */
template <diff::CExpression Expr,
          typename ParamSyms = diff::extract_symbols_from_expr_t<Expr>,
          typename InputSyms = mp::mp_list<>,
          typename Callbacks = callback::NoCallbacks>
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
                                        int itmax_ = 1000,
                                        Callbacks cbs = {})
      : Base(std::move(e)), ftol(ftol_), itmax(itmax_), cbs_(cbs) {}

  /**
   * @brief Run the LM iteration from an initial parameter vector.
   *
   * @param params  Initial guess a₀ ∈ ℝᴺ.
   * @param data    Observed (input, target, weight) triples.
   * @return        Best parameter vector found within @p itmax iterations.
   *
   * Convergence is declared after an accepted step when either
   *   ‖δa‖ < ftol (‖a‖ + ftol)       (step small relative to current point), or
   *   |Δχ²| < ftol (1 + χ²)           (objective change negligible).
   * These match the NR §15.5 stopping criteria.
   */
  constexpr ParamVec fit(ParamVec params, const std::vector<DataPoint> &data);

private:
  value_type ftol;
  int itmax;
  [[no_unique_address]] Callbacks cbs_{};
};

template <diff::CExpression Expr, typename ParamSyms, typename InputSyms, typename Callbacks>
constexpr LevenbergMarquardt<Expr, ParamSyms, InputSyms, Callbacks>::ParamVec
LevenbergMarquardt<Expr, ParamSyms, InputSyms, Callbacks>::fit(
    ParamVec params, const std::vector<DataPoint> &data) {
  using std::abs;
  constexpr int Ni = static_cast<int>(Base::N);
  using NMat = Eigen::Matrix<value_type, Ni, Ni>;
  using NVec = Eigen::Vector<value_type, Ni>;

  // Step 1: evaluate residuals r and Jacobian J at the initial parameter guess.
  value_type lambda = LAMBDA_INIT;
  auto [r, J] = this->eval_rJ(params, data);
  value_type chi2 = r.squaredNorm(); // χ² = ‖r‖²

  for (int iter = 0; iter < itmax; ++iter) {
    cbs_.on_lm_outer(iter, lambda, chi2);
    cbs_.on_iter_point(iter, std::span<const value_type>(params.data(), Base::N));
    // Step 2: build the Gauss-Newton normal matrix JᵀJ and RHS −Jᵀr.
    const NMat JtJ = J.transpose() * J;
    const NVec beta = -(J.transpose() * r); // −Jᵀr: gradient of ½χ²

    // Step 3: apply Marquardt damping — α = JᵀJ + λ·diag(JᵀJ)  (N&W eq. 10.24).
    // Scaling by diag(JᵀJ) makes λ dimensionless and parameter-scale-invariant.
    NMat alpha = JtJ;
    alpha.diagonal().array() *= (value_type{1} + lambda);

    // Step 4: solve the damped normal equations for the trial step δa.
    const NVec da = alpha.ldlt().solve(beta);
    const ParamVec p_new = params + da;

    // Step 5: evaluate χ²_new at the trial point.
    auto [r_new, J_new] = this->eval_rJ(p_new, data);
    const value_type chi2_new = r_new.squaredNorm();

    // Step 6: accept or reject the trial step.
    if (chi2_new < chi2) {
      // Step 6a: improvement — move more Newton-like (shrink λ).
      lambda *= LAMBDA_DOWN;
      cbs_.on_lm_inner(iter, lambda, chi2_new, true);
      // Step 6b: convergence test — step size and objective change both tiny.
      if ((da.norm() < ftol * (params.norm() + ftol)) ||
          (abs(chi2 - chi2_new) < ftol * (value_type{1} + chi2))) {
        return p_new;
      }
      // Step 6c: commit the step and refresh cached (r, J).
      params = p_new;
      chi2 = chi2_new;
      r = std::move(r_new);
      J = std::move(J_new);
    } else {
      // Step 6d: no improvement — move more gradient-descent-like (grow λ) and
      // retry.
      lambda *= LAMBDA_UP;
      cbs_.on_lm_inner(iter, lambda, chi2_new, false);
    }
  }
  return params;
}

template <diff::CExpression Expr>
LevenbergMarquardt(Expr) -> LevenbergMarquardt<Expr>;
template <diff::CExpression Expr, diff::Numeric T>
LevenbergMarquardt(Expr, T) -> LevenbergMarquardt<Expr>;
template <diff::CExpression Expr, diff::Numeric T>
LevenbergMarquardt(Expr, T, int) -> LevenbergMarquardt<Expr>;

template <diff::CExpression Expr> using LM = LevenbergMarquardt<Expr>;

} // namespace exprmin
