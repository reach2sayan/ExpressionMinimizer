#pragma once

#include "equation.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>

namespace exprmin {

namespace mp = boost::mp11;

// NLS 2D subspace trust-region method for minimizing ½‖R(θ)‖²,
// where R : ℝᴺ → ℝᴹ (M ≥ N) is given as diff::Equation<R1,...,RM>.
//
// Each iteration minimizes the Gauss-Newton quadratic model over the 2D
// subspace span{steepest-descent, Gauss-Newton direction} subject to the
// trust-region ball ‖dx‖ ≤ delta.  The constrained 2D minimizer is found by
// solving a degree-4 polynomial in the Lagrange multiplier λ via companion-
// matrix eigenvalues (Eigen::EigenSolver<Matrix4>).
//
// Reference: GSL multilarge_nlinear/subspace2D.c (Kaufman 1999)
template <typename System>
struct Subspace2D;

template <diff::CExpression... RExprs>
struct Subspace2D<diff::Equation<RExprs...>> {
  using Sys = diff::Equation<RExprs...>;
  static_assert(Sys::input_dim <= Sys::output_dim,
                "Subspace2D requires a system with M >= N residuals");

  using value_type = typename Sys::value_type;
  using Syms = typename Sys::symbols;
  static constexpr int M = static_cast<int>(Sys::output_dim);
  static constexpr int N = static_cast<int>(Sys::input_dim);
  using ParamVec = Eigen::Vector<value_type, N>;
  using RVec     = Eigen::Vector<value_type, M>;
  using JMat     = Eigen::Matrix<value_type, M, N>;
  using NMat     = Eigen::Matrix<value_type, N, N>;
  using WMat     = Eigen::Matrix<value_type, N, 2>;

  int iter{};

private:
  Sys system;
  value_type tol;
  int itmax;
  value_type trustregion0;
  value_type trustregion_min;
  value_type fret{};

  static constexpr value_type TR_DOWN_FACTOR    = value_type{0.1};
  static constexpr value_type TR_DOWN_THRESHOLD = value_type{0.25};
  static constexpr value_type TR_UP_FACTOR      = value_type{2.0};
  static constexpr value_type TR_UP_THRESHOLD   = value_type{0.75};

  constexpr auto to_arr(const ParamVec &p) const noexcept {
    std::array<value_type, static_cast<std::size_t>(N)> arr;
    for (int i = 0; i < N; ++i)
      arr[static_cast<std::size_t>(i)] = p[i];
    return arr;
  }

  constexpr std::pair<RVec, JMat> eval_rJ(const ParamVec &p) {
    system.update(Syms{}, to_arr(p));
    RVec r;
    if constexpr (M == 1) {
      r[0] = system.evaluate();
    } else {
      const auto r_arr = system.evaluate();
      for (int i = 0; i < M; ++i)
        r[i] = r_arr[static_cast<std::size_t>(i)];
    }
    const auto J_arr = system.template jacobian<diff::DiffMode::Reverse>();
    JMat J;
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j)
        J(i, j) = J_arr[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return {r, J};
  }

  // 2D subspace step: minimizes ½(g^T dx + dx^T B dx) over span{dx_sd, dx_gn} ∩ ‖dx‖ ≤ delta
  constexpr ParamVec compute_step(const ParamVec &g, const NMat &B,
                                   const JMat &J, value_type delta) const;

public:
  constexpr value_type get_optimal_value() const { return fret; }

  constexpr explicit Subspace2D(diff::Equation<RExprs...> sys,
                                 value_type tol_   = value_type{1e-8},
                                 int itmax_        = 200,
                                 value_type tr0_   = value_type{1e3},
                                 value_type trmin_ = value_type{1e-12})
      : system{std::move(sys)}, tol{tol_}, itmax{itmax_},
        trustregion0{tr0_}, trustregion_min{trmin_} {}

  constexpr ParamVec minimize(ParamVec p);
};

// ── compute_step ──────────────────────────────────────────────────────────────

template <diff::CExpression... RExprs>
constexpr typename Subspace2D<diff::Equation<RExprs...>>::ParamVec
Subspace2D<diff::Equation<RExprs...>>::compute_step(
    const ParamVec &g, const NMat &B, const JMat &J, value_type delta) const {
  using std::sqrt, std::abs;
  constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();

  // Cauchy step: dx_sd = -alpha * g
  const RVec Jg = J * g;
  const value_type Jg_sq = Jg.squaredNorm();
  const value_type alpha =
      (Jg_sq > value_type{0}) ? g.squaredNorm() / Jg_sq : value_type{1};
  const ParamVec dx_sd = -alpha * g;

  // Gauss-Newton step
  const ParamVec dx_gn = -(B.ldlt().solve(g));
  const value_type norm_gn = dx_gn.norm();

  // GN inside TR → exact solution of the quadratic model
  if (norm_gn <= delta)
    return dx_gn;

  // Build orthonormal basis W = [q0, q1] via Gram-Schmidt
  const value_type norm_sd = dx_sd.norm();
  const ParamVec q0 = dx_sd / norm_sd;

  ParamVec q1_raw = dx_gn - q0.dot(dx_gn) * q0;
  const value_type q1_norm = q1_raw.norm();
  const int rank = (q1_norm < EPS * norm_gn) ? 1 : 2;

  // Rank-1 degenerate: subspace collapses to steepest-descent direction
  if (rank == 1)
    return (delta / norm_sd) * dx_sd;

  const ParamVec q1 = q1_raw / q1_norm;

  // W is [q0 | q1] as N×2 matrix
  WMat W;
  W.col(0) = q0;
  W.col(1) = q1;

  // Reduced 2×2 system
  const Eigen::Vector<value_type, 2> subg = W.transpose() * g;
  const Eigen::Matrix<value_type, 2, 2> subB = W.transpose() * B * W;

  const value_type trB   = subB.trace();
  const value_type detB  = subB.determinant();
  const value_type d2    = delta * delta;

  // Precompute terms for the quartic in λ:
  //   ‖(B+λI)^{-1} g‖² = δ²  →  quartic a0 + a1λ + a2λ² + a3λ³ + λ⁴ = 0
  // adj(B)^T = adj(B) for symmetric 2×2
  //   term0 = g^T adj(B)^T adj(B) g  = ‖adj(B) g‖²
  //   term1 = g^T adj(B)^T g = g^T adj(B) g
  Eigen::Matrix<value_type, 2, 2> adjB;
  adjB(0,0) =  subB(1,1);  adjB(0,1) = -subB(0,1);
  adjB(1,0) = -subB(1,0);  adjB(1,1) =  subB(0,0);

  const Eigen::Vector<value_type, 2> adjBg = adjB * subg;
  const value_type term0 = adjBg.squaredNorm();
  const value_type term1 = subg.dot(adjBg);
  const value_type subg2 = subg.squaredNorm();

  // Quartic: λ⁴ + a3λ³ + a2λ² + a1λ + a0
  const value_type a3 = value_type{2} * trB;
  const value_type a2 = trB * trB + value_type{2} * detB - subg2 / d2;
  const value_type a1 = value_type{2} * (detB * trB - term1 / d2);
  const value_type a0 = detB * detB - term0 / d2;

  // Companion matrix (colleague / Frobenius form)
  Eigen::Matrix<double, 4, 4> C = Eigen::Matrix<double, 4, 4>::Zero();
  C(0, 3) = -static_cast<double>(a0);
  C(1, 3) = -static_cast<double>(a1);
  C(2, 3) = -static_cast<double>(a2);
  C(3, 3) = -static_cast<double>(a3);
  C(1, 0) = C(2, 1) = C(3, 2) = 1.0;

  Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> es(C, false);
  const auto eigs = es.eigenvalues();  // 4 complex values

  // Evaluate 2D objective for each real eigenvalue root
  value_type best_obj = std::numeric_limits<value_type>::max();
  Eigen::Vector<value_type, 2> y_best;
  y_best.setZero();
  bool found = false;

  for (int k = 0; k < 4; ++k) {
    if (abs(eigs[k].imag()) > value_type{1e-6} * abs(eigs[k].real()))
      continue;
    const value_type lam = static_cast<value_type>(eigs[k].real());

    // Solve (subB + lambda*I) y = -subg in 2D
    Eigen::Matrix<value_type, 2, 2> A2 = subB;
    A2(0, 0) += lam;
    A2(1, 1) += lam;

    const value_type det2 = A2.determinant();
    if (abs(det2) < EPS)
      continue;

    const Eigen::Vector<value_type, 2> y_raw =
        -(A2.inverse() * subg);

    // Scale to trust-region boundary
    const value_type yn = y_raw.norm();
    if (yn < EPS)
      continue;
    const Eigen::Vector<value_type, 2> y = (delta / yn) * y_raw;

    // Evaluate quadratic objective: subg^T y + ½ y^T subB y
    const value_type obj = subg.dot(y) + value_type{0.5} * y.dot(subB * y);
    if (obj < best_obj) {
      best_obj = obj;
      y_best = y;
      found = true;
    }
  }

  if (found)
    return W * y_best;

  // Fallback: Cauchy truncated to boundary
  return (delta / norm_sd) * dx_sd;
}

// ── minimize ──────────────────────────────────────────────────────────────────

template <diff::CExpression... RExprs>
constexpr typename Subspace2D<diff::Equation<RExprs...>>::ParamVec
Subspace2D<diff::Equation<RExprs...>>::minimize(ParamVec p) {
  using std::abs, std::max;

  auto [r, J] = eval_rJ(p);
  NMat B        = J.transpose() * J;
  ParamVec g    = J.transpose() * r;
  value_type phi = value_type{0.5} * r.squaredNorm();
  value_type delta = trustregion0;

  for (iter = 0; iter < itmax; ++iter) {
    const value_type den = max(abs(phi), value_type{1});
    const value_type gnorm =
        (g.cwiseAbs().array() * p.cwiseAbs().cwiseMax(value_type{1}).array())
            .maxCoeff();
    if (gnorm / den < tol) break;

    const ParamVec step = compute_step(g, B, J, delta);
    const bool at_boundary = step.norm() >= value_type{0.999} * delta;

    const value_type pred =
        -(g.dot(step) + value_type{0.5} * step.dot(B * step));

    const ParamVec p_new = p + step;
    auto [r_new, J_new] = eval_rJ(p_new);
    const value_type phi_new = value_type{0.5} * r_new.squaredNorm();

    const value_type rho =
        (pred > value_type{0}) ? (phi - phi_new) / pred : value_type{0};

    if (rho < TR_DOWN_THRESHOLD) {
      delta *= TR_DOWN_FACTOR;
    } else if (rho > TR_UP_THRESHOLD && at_boundary) {
      delta *= TR_UP_FACTOR;
    }
    if (delta < trustregion_min) break;

    if (rho > value_type{0}) {
      if (step.cwiseAbs().maxCoeff() < tol) break;
      p   = p_new;
      r   = std::move(r_new);
      J   = std::move(J_new);
      B   = J.transpose() * J;
      g   = J.transpose() * r;
      phi = phi_new;
    }
  }

  fret = phi;
  return p;
}

// ── deduction guides ──────────────────────────────────────────────────────────

template <diff::CExpression... RExprs>
Subspace2D(diff::Equation<RExprs...>)
    -> Subspace2D<diff::Equation<RExprs...>>;

template <diff::CExpression... RExprs, typename T>
Subspace2D(diff::Equation<RExprs...>, T)
    -> Subspace2D<diff::Equation<RExprs...>>;

} // namespace exprmin
