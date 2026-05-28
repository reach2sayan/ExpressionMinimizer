#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace exprmin {

/**
 * @brief Linear Programming via a primal-dual infeasible interior-point method.
 *
 * Solves the LP in standard (equality) form:
 * @f[
 *   \min_x \; \mathbf{c}^\top \mathbf{x}
 *   \quad \text{subject to} \quad
 *   A\mathbf{x} = \mathbf{b}, \quad \mathbf{x} \ge 0
 * @f]
 *
 * where @p A is \f$(m \times n)\f$, @p b is \f$(m)\f$, @p c is \f$(n)\f$.
 * The @p n columns must include any slack variables needed to convert
 * inequality constraints to equalities.
 *
 * **Typical setup for \f$A_\text{ub}\,x \le b_\text{ub},\ x \ge 0\f$**
 * @code
 *   A_std = [A_ub | I_m]     // m × (n_orig + m)
 *   b_std = b_ub
 *   c_std = [c; 0...0]       // n_orig + m
 * @endcode
 * The returned vector has @p n components; extract the first @p n_orig for @p
 * x.
 *
 * **Algorithm** (NR3 §10.11.6, Lustig–Marsten–Shanno 1994)\n
 * At each iteration the KKT conditions \f$(A x = b,\ A^\top y + z = c,\
 * X Z e = \tau e)\f$ are linearised to give the Newton system
 * \f$(A X Z^{-1} A^\top)\,\Delta y = \text{rhs}\f$ (the *normal equations*),
 * solved via Eigen LDLT.  The centering parameter
 * \f$\tau = \delta\,(x^\top z)/n\f$ keeps iterates away from the boundary.
 * Separate primal (\f$\alpha_p\f$) and dual (\f$\alpha_d\f$) step lengths
 * are found with @c std::transform_reduce and shrunk by a safety factor
 * \f$\sigma\f$ before updating \f$(x, y, z)\f$.
 *
 * After solve(), inspect #status for the outcome and #fret for the value.
 *
 * @tparam T  Scalar type (default: @c double).
 */
template <typename T = double> struct InteriorPointLP {

  /**
   * @brief Outcome of the last solve() call.
   *
   * - @c Optimal          — KKT conditions satisfied; #fret holds the value.
   * - @c PrimalInfeasible — primal residual norm growing unboundedly.
   * - @c DualInfeasible   — dual residual norm growing unboundedly
   *                         (typically indicates the primal is unbounded).
   * - @c MaxIter          — #ITMAX iterations reached without convergence.
   */
  enum class Status { Optimal, PrimalInfeasible, DualInfeasible, MaxIter };

  using VecX = Eigen::VectorX<T>; ///< Dynamic column vector.
  using MatX = Eigen::MatrixX<T>; ///< Dynamic matrix.

  static constexpr int ITMAX = 200; ///< Maximum Newton iterations.
  static constexpr T EPS{1e-6};     ///< Convergence tolerance.
  static constexpr T SIGMA{0.9};    ///< Step safety factor \f$\sigma\f$.
  static constexpr T DELTA{0.02};   ///< Centering parameter \f$\delta\f$.

  Status status{Status::MaxIter}; ///< Outcome set after each solve().
  T fret{};   ///< Optimal objective value (valid when Optimal).
  int iter{}; ///< Newton iteration count of the last solve().

  /**
   * @brief Solve the LP (standard form).
   *
   * @param A  Equality constraint matrix \f$(m \times n)\f$.
   * @param b  Right-hand side \f$(m)\f$.
   * @param c  Objective coefficients \f$(n)\f$; pad with zeros for slack
   * columns.
   * @return   Primal vector \f$\mathbf{x} \in \mathbb{R}^n\f$ at the optimum,
   *           or the last iterate when #status is not @c Optimal.
   * @post     #status and #fret are updated.
   */
  VecX solve(const MatX &A, const VecX &b, const VecX &c);
};

template <typename T>
InteriorPointLP<T>::VecX InteriorPointLP<T>::solve(const MatX &A, const VecX &b,
                                                   const VecX &c) {
  using std::sqrt, std::abs;
  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(A.cols());

  // Normalisation factors for residual convergence tests (NR p.545).
  const T rpfact = T{1} + sqrt(b.squaredNorm());
  const T rdfact = T{1} + sqrt(c.squaredNorm());
  const T BIG = std::numeric_limits<T>::max();

  // Initial strictly-interior point: x = z = y = 1000  (NR §10.11.6).
  VecX x = VecX::Constant(n, T{1000});
  VecX z = VecX::Constant(n, T{1000});
  VecX y = VecX::Constant(m, T{1000});

  T normrp_old = BIG;
  T normrd_old = BIG;

  const MatX At = A.transpose();

  for (iter = 0; iter < ITMAX; ++iter) {
    // Primal and dual residuals (eq. 10.11.21).
    const VecX rp = A * x - b;
    const VecX rd = At * y + z - c;

    const T normrp = sqrt(rp.squaredNorm()) / rpfact;
    const T normrd = sqrt(rd.squaredNorm()) / rdfact;
    const T gap = x.dot(z);
    const T primal_obj = c.dot(x);
    const T gamma_norm = gap / (T{1} + abs(primal_obj));

    // Optimal if all three KKT measures are below tolerance.
    if (normrp < EPS && normrd < EPS && gamma_norm < EPS) {
      status = Status::Optimal;
      fret = primal_obj;
      return x;
    }

    // Infeasibility detection: residual norms growing by factor > 1000.
    if (normrp > T{1000} * normrp_old && normrp > EPS) {
      status = Status::PrimalInfeasible;
      return x;
    }
    if (normrd > T{1000} * normrd_old && normrd > EPS) {
      status = Status::DualInfeasible;
      return x;
    }

    normrp_old = normrp;
    normrd_old = normrd;

    // Centering parameter μ = δ·(xᵀz)/n  (eq. 10.11.12).
    const T mu = DELTA * gap / static_cast<T>(n);
    // D = X·Z⁻¹  (dⱼ = xⱼ/zⱼ).
    const VecX d = x.cwiseQuotient(z);
    // Normal equations matrix M = A·D·Aᵀ  (eq. 10.11.23); LDLT factorisation.
    const MatX M = A * d.asDiagonal() * At;
    Eigen::LDLT<MatX> solver(M);

    // RHS of normal equations: tempnⱼ = xⱼ − μ/zⱼ − dⱼ·rdⱼ.
    const VecX tempn = x - mu * z.cwiseInverse() - d.cwiseProduct(rd);
    const VecX rhs = -rp + A * tempn;

    // Newton directions: Δy (normal eqs), Δz (eq. 10.11.24), Δx (eq. 10.11.25).
    const VecX dy = solver.solve(rhs);
    const VecX dz = -(At * dy) - rd;
    const VecX dx = -d.cwiseProduct(dz) + mu * z.cwiseInverse() - x;

    /**
     * Maximum feasible step-length along @p dv from @p v, keeping
     * \f$v + \alpha\,dv \ge 0\f$.
     *
     * @c std::transform_reduce finds \f$\min(-v_j/dv_j)\f$ over all @p j
     * where \f$dv_j < 0\f$, then @c std::clamp applies the safety factor σ.
     */
    const auto max_step = [&](const VecX &v, const VecX &dv) -> T {
      const T raw = std::transform_reduce(
          dv.data(), dv.data() + n, v.data(), T{1},
          [](T a, T b) { return std::min(a, b); },
          [](T dvj, T vj) -> T { return dvj < T{0} ? -vj / dvj : T{1}; });
      return std::clamp(raw * SIGMA, T{0}, T{1});
    };

    // Separate primal (αₚ) and dual (α_d) step lengths (eq. 10.11.26).
    const T alpha_p = max_step(x, dx);
    const T alpha_d = max_step(z, dz);

    x += alpha_p * dx;
    y += alpha_d * dy;
    z += alpha_d * dz;
  }

  status = Status::MaxIter;
  fret = c.dot(x);
  return x;
}

} // namespace exprmin
