#pragma once

#include "../minimizer/trustregionbase.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace exprmin {

/**
 * @brief 2D subspace trust-region NLS solver (Byrd, Schnabel & Schultz 1988).
 *
 * Minimises @f$ \varphi(\theta) = \tfrac{1}{2}\|R(\theta)\|^2 @f$ where
 * @f$ R : \mathbb{R}^N \to \mathbb{R}^M @f$ (@f$ M \ge N @f$) is supplied as
 * a diff::Equation<R₁,…,Rₘ>.  The outer trust-region loop and all NLS state
 * management (residuals, Jacobian, Gauss-Newton @f$ B = J^\top J @f$) are
 * inherited from NLSTrustRegionBase.  This class only implements
 * @c compute_step, which restricts the quadratic TR subproblem to a 2D
 * subspace and solves it exactly on the boundary via a secular equation.
 *
 * ### Algorithm (Byrd et al. §3)
 *
 * The quadratic model at each iterate is
 * @f[
 *   m(p) = \varphi + g^\top p + \tfrac{1}{2}p^\top B p,
 *   \quad B = J^\top J,\; g = J^\top r
 * @f]
 * The step @f$ p @f$ is found by minimising @f$ m @f$ over the trust region
 * @f$ \|p\| \le \Delta @f$ restricted to the 2D subspace
 * @f$ \mathcal{W} = \operatorname{span}\{q_0, q_1\} @f$:
 *
 *   - @f$ q_0 = p_\mathrm{sd}/\|p_\mathrm{sd}\| @f$ — normalised steepest-
 *     descent (Cauchy) direction
 *   - @f$ q_1 = \operatorname{orth}(p_\mathrm{gn},\, q_0) @f$ — component of
 *     the Gauss-Newton step orthogonal to @f$ q_0 @f$, normalised
 *
 * Writing @f$ p = W y @f$ with @f$ W = [q_0\; q_1] \in \mathbb{R}^{N\times2}
 * @f$ reduces the subproblem to a 2D constrained quadratic:
 * @f[
 *   \min_{y\in\mathbb{R}^2}\; g_s^\top y + \tfrac{1}{2}y^\top B_s y
 *   \quad\text{s.t.}\quad \|y\| \le \Delta
 * @f]
 * where @f$ g_s = W^\top g @f$ and @f$ B_s = W^\top B W @f$.
 *
 * If the Gauss-Newton step @f$ p_\mathrm{gn} @f$ already lies inside the trust
 * region, it is returned directly.  Otherwise the boundary optimum is found via
 * the secular equation @f$ \|(B_s + \lambda I)^{-1}g_s\| = \Delta @f$, which
 * for a @f$ 2\times2 @f$ system yields a degree-4 polynomial in @f$ \lambda
 * @f$. Its roots are the eigenvalues of the @f$ 4\times4 @f$ companion matrix;
 * all real roots are evaluated and the one that minimises @f$ m @f$ is
 * returned.
 *
 * @tparam System  Specialisation of diff::Equation<RExprs...>.
 */
template <typename System> struct Subspace2D;

template <diff::CExpression... RExprs>
struct Subspace2D<diff::Equation<RExprs...>>
    : NLSTrustRegionBase<Subspace2D<diff::Equation<RExprs...>>, RExprs...> {
  using Base =
      NLSTrustRegionBase<Subspace2D<diff::Equation<RExprs...>>, RExprs...>;
  using value_type = typename Base::value_type;
  static constexpr int N = Base::N;
  using ParamVec = typename Base::ParamVec;
  using NMat = typename Base::NMat;
  using RVec = typename Base::RVec;
  using JMat = typename Base::JMat;
  /// Column matrix W ∈ ℝᴺˣ² holding the two orthonormal subspace basis vectors.
  using WMat = Eigen::Matrix<value_type, N, 2>;

  using Base::Base;
  using Base::get_optimal_value;
  using Base::iter;

  /**
   * @brief Compute the 2D subspace trust-region step.
   *
   * Called by TrustRegionBase::minimize at each outer iteration.
   *
   * @param g      Current gradient @f$ g = J^\top r @f$ (N-vector).
   * @param B      Current Gauss-Newton Hessian @f$ B = J^\top J @f$ (N×N).
   * @param delta  Current trust-region radius @f$ \Delta @f$.
   * @return       Step @f$ p \in \mathbb{R}^N @f$ with @f$ \|p\| \le \Delta
   * @f$.
   */
  constexpr ParamVec compute_step(const ParamVec &g, const NMat &B,
                                  value_type delta) const;
};

template <diff::CExpression... RExprs>
constexpr typename Subspace2D<diff::Equation<RExprs...>>::ParamVec
Subspace2D<diff::Equation<RExprs...>>::compute_step(const ParamVec &g,
                                                    const NMat &B,
                                                    value_type delta) const {
  using std::abs, std::sqrt;
  constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();
  const JMat &J = this->current_J();

  // ── Step 1: Cauchy (steepest-descent) step ──────────────────────────────
  // The exact 1D minimiser of m along −g is α = ‖g‖²/‖Jg‖² (N&W §10.2.1).
  // Equivalently, this is the Cauchy point projected onto the SD ray.
  const RVec Jg = J * g;
  const auto Jg_sq = Jg.squaredNorm();
  const auto alpha =
      (Jg_sq > value_type{0}) ? g.squaredNorm() / Jg_sq : value_type{1};
  const ParamVec dx_sd = -alpha * g;

  // ── Step 2: Gauss-Newton step ────────────────────────────────────────────
  // Unconstrained minimiser of m: p_gn = −B⁻¹g = −(JᵀJ)⁻¹Jᵀr (N&W eq. 10.8).
  // Solved via LDLT; B = JᵀJ is SPSD (positive definite when J has full rank).
  const ParamVec dx_gn = -(B.ldlt().solve(g));
  const auto norm_gn = dx_gn.norm();

  // ── Step 3: return GN step if it fits inside the trust region ───────────
  if (norm_gn <= delta) {
    return dx_gn;
  }

  // ── Step 4: build orthonormal 2D basis {q₀, q₁} ────────────────────────
  // q₀: normalised steepest-descent direction.
  const auto norm_sd = dx_sd.norm();
  const ParamVec q0 = dx_sd / norm_sd;

  // q₁: component of p_gn orthogonal to q₀ (Gram-Schmidt), then normalised.
  // If this component is negligible, the GN direction lies in span{q₀} and
  // the subspace degenerates to 1D — just scale along the SD ray to the
  // boundary.
  ParamVec q1_raw = dx_gn - q0.dot(dx_gn) * q0;
  const auto q1_norm = q1_raw.norm();
  const int rank = (q1_norm < EPS * norm_gn) ? 1 : 2;

  if (rank == 1) {
    // 1D degenerate case: scale SD step to touch the trust-region boundary.
    return (delta / norm_sd) * dx_sd;
  }

  const ParamVec q1 = q1_raw / q1_norm;

  // ── Step 5: project model to 2D subspace W = [q₀ q₁] ───────────────────
  // gₛ = Wᵀg (2-vector), Bₛ = WᵀBW (2×2) — the restricted quadratic model.
  WMat W;
  W << q0, q1;
  const auto subg = W.transpose() * g;     // gₛ ∈ ℝ²
  const auto subB = W.transpose() * B * W; // Bₛ ∈ ℝ²ˣ²

  // ── Step 6: build the secular equation for the boundary solution ─────────
  // We want min ½yᵀBₛy + gₛᵀy  s.t. ‖y‖ = Δ (boundary, since GN overshot).
  // KKT: (Bₛ + λI)y = −gₛ,  ‖y‖ = Δ.
  // Eliminating y gives the secular equation h(λ) = ‖(Bₛ + λI)⁻¹gₛ‖² − Δ² = 0.
  //
  // For a 2×2 Bₛ, multiply through by det²(Bₛ + λI) to clear denominators.
  // Using adj(Bₛ + λI) = (trB·I − Bₛ) + λI (Cayley-Hamilton in 2D):
  //   ‖adj(Bₛ+λI)·gₛ‖² = Δ²·det²(Bₛ+λI)
  // Expanding with tr ≡ tr(Bₛ), det ≡ det(Bₛ), and the adjugate–adjugate
  // identity yields a degree-4 polynomial in λ (Byrd et al. §3, eq. 3.3):
  //   λ⁴·a₃ + λ³·a₂ + λ²·a₁ + λ·a₀ = 0  →  monic companion below.

  const auto trB = subB.trace();
  const auto detB = subB.determinant();
  const auto d2 = delta * delta;

  // Adjugate of Bₛ (exact 2×2 formula: swap diagonal, negate off-diagonal).
  Eigen::Matrix<value_type, 2, 2> adjB;
  adjB << subB(1, 1), -subB(0, 1), -subB(1, 0), subB(0, 0);

  const auto adjBg = adjB * subg;
  const auto term0 = adjBg.squaredNorm(); // ‖adj(Bₛ)gₛ‖²
  const auto term1 = subg.dot(adjBg);     // gₛᵀ adj(Bₛ) gₛ  (= gₛᵀ adj gₛ)
  const auto subg2 = subg.squaredNorm();  // ‖gₛ‖²

  // Coefficients of the monic degree-4 secular polynomial p(λ):
  //   p(λ) = λ⁴ + a₃λ³ + a₂λ² + a₁λ + a₀
  // Derived by expanding h(λ)·det²(Bₛ+λI) = 0 and dividing by the leading λ⁴.
  const auto a3 = value_type{2} * trB;
  const auto a2 = trB * trB + value_type{2} * detB - subg2 / d2;
  const auto a1 = value_type{2} * (detB * trB - term1 / d2);
  const auto a0 = detB * detB - term0 / d2;

  // ── Step 7: find roots via 4×4 companion matrix eigenvalues ─────────────
  // The companion matrix C of p(λ) has eigenvalues equal to the roots of p.
  // We use double precision throughout since Eigen's EigenSolver is real-only
  // for non-symmetric matrices, and the secular polynomial is degree 4.
  // clang-format off
  Eigen::Matrix4d C;
  C << 0.0, 0.0, 0.0, -static_cast<double>(a0),
       1.0, 0.0, 0.0, -static_cast<double>(a1),
       0.0, 1.0, 0.0, -static_cast<double>(a2),
       0.0, 0.0, 1.0, -static_cast<double>(a3);
  // clang-format on

  Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> es(
      C, /*computeEigenvectors=*/false);
  const auto eigs = es.eigenvalues();

  // ── Step 8: evaluate each real root; keep the one minimising m ──────────
  // A root is considered real when |Im(λ)| ≤ 1e-6·|Re(λ)|.
  // For each real λ, the boundary point is y = −Δ·(Bₛ+λI)⁻¹gₛ / ‖(Bₛ+λI)⁻¹gₛ‖
  // (project y_raw onto the sphere of radius Δ).
  value_type best_obj = std::numeric_limits<value_type>::max();
  Eigen::Vector<value_type, 2> y_best = Eigen::Vector<value_type, 2>::Zero();
  bool found = false;

  for (int k = 0; k < 4; ++k) {
    if (abs(eigs[k].imag()) > value_type{1e-6} * abs(eigs[k].real())) {
      continue; // discard complex root
    }
    const auto lam = static_cast<value_type>(eigs[k].real());

    // Compute (Bₛ + λI) and check it is non-singular before inverting.
    Eigen::Matrix<value_type, 2, 2> A2 = subB;
    A2.diagonal().array() += lam;
    const value_type det2 = A2.determinant();
    if (abs(det2) < EPS) {
      continue; // singular shift — skip
    }

    // y_raw = −(Bₛ+λI)⁻¹gₛ (unconstrained KKT solution for this λ).
    // Project onto the sphere: y = Δ · y_raw / ‖y_raw‖.
    const auto y_raw = -(A2.inverse() * subg);
    const value_type yn = y_raw.norm();
    if (yn < EPS) {
      continue;
    }
    const auto y = (delta / yn) * y_raw;

    // Evaluate model value m(Wy) = gₛᵀy + ½yᵀBₛy for this candidate.
    const value_type obj = subg.dot(y) + value_type{0.5} * y.dot(subB * y);
    if (obj < best_obj) {
      best_obj = obj;
      y_best = y;
      found = true;
    }
  }

  // ── Step 9: return best boundary point, fall back to scaled SD if no root ─
  if (found) {
    return W * y_best; // lift 2D solution back to N-dimensional space
  }
  return (delta / norm_sd) * dx_sd; // fallback: Cauchy point on boundary
}

template <diff::CExpression... RExprs>
Subspace2D(diff::Equation<RExprs...>) -> Subspace2D<diff::Equation<RExprs...>>;

template <diff::CExpression... RExprs, typename T>
Subspace2D(diff::Equation<RExprs...>, T)
    -> Subspace2D<diff::Equation<RExprs...>>;

} // namespace exprmin
