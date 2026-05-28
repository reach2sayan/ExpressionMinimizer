#pragma once

#include "lsq_base.hpp"

namespace exprmin {

/**
 * @brief Gauss-Newton nonlinear least-squares fitter (Nocedal & Wright §10.2;
 *        Dennis & Schnabel §10.1).
 *
 * Minimises @f$ \chi^2 = \tfrac{1}{2}\sum_i [w_i(y_i - f(x_i;\mathbf{a}))]^2
 * @f$ over the parameter vector @f$ \mathbf{a} \in \mathbb{R}^N @f$ by
 * iterating the *undamped* normal equations
 *
 * @f[
 *   J^\top J \,\delta\mathbf{a} = -J^\top \mathbf{r},
 *   \qquad \mathbf{a} \leftarrow \mathbf{a} + \delta\mathbf{a}
 * @f]
 *
 * where @f$ r_i = w_i(y_i - f(x_i;\mathbf{a})) @f$ and
 * @f$ J_{ij} = -w_i\,\partial f/\partial a_j @f$ (negated to absorb the minus
 * sign into the standard NLS form; the residual sign convention matches
 * LeastSquaresBase).  The @f$ J^\top J @f$ system is solved via LDLT
 * Cholesky factorisation.
 *
 * **Convergence**: quadratic near the solution when the residual is small
 * (zero-residual problem); may diverge when the initial guess is poor or the
 * problem is large-residual.  Use LevenbergMarquardt in those cases.
 *
 * **Stopping rule**: relative step-norm criterion
 * @f$ \|\delta a\| < \tau(\|\mathbf{a}\| + \tau) @f$
 * (Dennis & Schnabel eq. 8.2.8), where @f$ \tau = @f$ @p ftol.
 *
 * @tparam Expr      A type satisfying diff::CExpression.
 * @tparam ParamSyms Compile-time list of parameter symbol chars (default: all
 *                   symbols in Expr).
 * @tparam InputSyms Compile-time list of input (predictor) symbol chars
 *                   (default: empty — pure parameter fitting).
 */
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

  /**
   * @brief Fit parameters to @p data starting from @p params.
   *
   * @param params  Initial parameter vector @f$ \mathbf{a}_0 @f$.
   * @param data    Observed data points (input, target, weight).
   * @return        Parameter vector at convergence or after @c itmax steps.
   */
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
    // Step 1: evaluate residual vector r (M×1) and Jacobian J (M×N) at the
    // current parameters. Sign convention from LeastSquaresBase:
    //   r[i] = w_i (y_i − f(x_i; a)),   J[i,j] = −w_i ∂f/∂a_j
    // so J^T J = (J_f^T J_f) and −J^T r = J_f^T r_f, recovering the standard
    // normal equations (N&W eq. 10.8).
    auto [r, J] = eval_rJ(params, data);

    // Step 2: solve the normal equations (JᵀJ) δa = −Jᵀr via LDLT.
    // JᵀJ is symmetric positive semi-definite (exactly positive definite when
    // J has full column rank, which holds generically away from degeneracies).
    const NVec step =
        (J.transpose() * J)
            .ldlt()
            .solve(
                -(J.transpose() * r)); // minus: sign absorbed from J convention

    // Step 3: accept step unconditionally (no damping — pure Gauss-Newton).
    params += step;

    // Step 4: relative step-norm stopping rule (Dennis & Schnabel §8.2.8).
    // Stops when the step is negligible relative to the current parameter
    // scale.
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
