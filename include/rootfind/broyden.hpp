#pragma once

#include "equation.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>
#include <ranges>

namespace exprmin {

namespace mp = boost::mp11;

/**
 * @brief NR §9.7 — Broyden's rank-1 quasi-Newton root finder for F(x) = 0.
 *
 * Port of GSL @c gsl_multiroot_fsolver_broyden (Broyden 1965).  Uses the exact
 * reverse-mode AD Jacobian for the initial inverse-Jacobian H₀ = −J(x₀)⁻¹ and
 * after any line-search failure; all subsequent H updates are rank-1 and
 * Jacobian-free.
 *
 * **Usage:** @c Broyden<F1,F2,...,FN> where each @c Fi satisfies
 * @c diff::CExpression.  Pass the system as @c diff::Equation{f1,f2,...} to
 * the constructor.  Requires a square system: N equations in N unknowns.
 *
 * **Algorithm per iteration:**
 * @code
 *   p  = H · f               (Newton-like step; H ≈ −J⁻¹)
 *   t  = Hebden backtracking step-size (shrinks while ‖F(x+tp)‖ > ‖F(x)‖)
 *   y  = F(x+tp) − F(x)
 *   H ← H − (H·y + t·p)(p^T·H) / (p^T·H·y)   (rank-1 Broyden update)
 *   fallback: if line search fails, refresh H = −J(x)⁻¹ exactly via AD
 * @endcode
 *
 * @tparam FExprs  Pack of expression types, each satisfying diff::CExpression.
 */
template <diff::CExpression... FExprs> struct Broyden {
  using System = diff::Equation<FExprs...>;
  static_assert(System::input_dim == System::output_dim,
                "Broyden requires a square system: #equations == #variables");

  using value_type = typename System::value_type;
  using Syms = typename System::symbols;
  static constexpr int N = static_cast<int>(System::output_dim);
  using Point = Eigen::Vector<value_type, N>;
  using Matrix = Eigen::Matrix<value_type, N, N>;

  int iter{};

private:
  System system;
  value_type tol;
  int itmax;

  static constexpr int LS_ITMAX = 10;

  constexpr auto to_arr(const Point &p) const noexcept {
    std::array<value_type, static_cast<std::size_t>(N)> arr;
    std::copy(p.data(), p.data() + N, arr.begin());
    return arr;
  }

  constexpr Point eval_f(const Point &p) {
    system.update(Syms{}, to_arr(p));
    Point f;
    if constexpr (N == 1) {
      f[0] = system.evaluate();
    } else {
      const auto f_arr = system.evaluate();
      std::ranges::copy(f_arr | std::views::take(N), f.begin());
    }
    return f;
  }

  // Exact Jacobian via reverse-mode AD; returns J(p).
  constexpr Matrix eval_J(const Point &p) {
    system.update(Syms{}, to_arr(p));
    const auto J_arr = system.template jacobian<diff::DiffMode::Reverse>();
    Matrix J = Eigen::Map<const Eigen::MatrixXd>(&J_arr[0][0], N, N);
    return J;
  }

  // H = −J(p)⁻¹  (exact inverse via LU).
  constexpr Matrix make_H(const Point &p) {
    return -(eval_J(p).lu().solve(Matrix::Identity()));
  }

public:
  /// @brief Returns ‖F(x)‖ at the last accepted iterate.
  constexpr value_type residual_norm() const { return last_phi; }

  /**
   * @brief Constructs a Broyden solver wrapping the given equation system.
   * @param sys    Equation system (N expressions in N unknowns).
   * @param tol_   Convergence tolerance on ‖F(x)‖ (default 1×10⁻¹⁰).
   * @param itmax_ Maximum number of iterations (default 200).
   */
  constexpr explicit Broyden(diff::Equation<FExprs...> sys,
                             value_type tol_ = value_type{1e-10},
                             int itmax_ = 200)
      : system{std::move(sys)}, tol{tol_}, itmax{itmax_} {}

  /**
   * @brief Finds a root of F starting from initial guess @p p.
   * @param p  Initial guess (passed by value; used as the running iterate).
   * @return   Approximate root x* with ‖F(x*)‖ < tol, or the best point
   *           found after itmax iterations.
   */
  constexpr Point find_root(Point p);

private:
  value_type last_phi{};
};

template <diff::CExpression... FExprs>
constexpr typename Broyden<FExprs...>::Point
Broyden<FExprs...>::find_root(Point p) {
  using std::sqrt, std::abs;
  constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();

  // Step 1: evaluate F and build the initial exact inverse-Jacobian H = −J⁻¹.
  Point f = eval_f(p);
  Matrix H = make_H(p);
  value_type phi = f.norm();

  for (iter = 0; iter < itmax; ++iter) {
    last_phi = phi;
    // Step 2: converge when ‖F(x)‖ is within tolerance.
    if (phi < tol) {
      break;
    }

    // Step 3: compute the Newton-like step p_step = H · F(x).
    Point p_step = H * f;

    const value_type phi0 = phi;
    value_type t = value_type{1};
    Point f_new;
    value_type phi1{};

    // Step 4: Hebden backtracking — shrink t until ‖F(x+t·p)‖ ≤ ‖F(x)‖.
    for (int ls = 0; ls < LS_ITMAX; ++ls) {
      f_new = eval_f(p + t * p_step);
      phi1 = f_new.norm();
      if (phi1 <= phi0 || t <= value_type{0.1}) {
        break;
      }
      const value_type theta = phi1 / phi0;
      t *= (sqrt(value_type{1} + value_type{6} * theta) - value_type{1}) /
           (value_type{3} * theta);
    }

    // Step 5: if line search still fails, refresh H exactly from AD and retry.
    if (phi1 > phi0) {
      H = make_H(p);
      p_step = H * f;
      t = value_type{1};
      f_new = eval_f(p + t * p_step);
      phi1 = f_new.norm();
    }

    // Step 6: rank-1 Broyden update of H:
    //   y  = F(x+tp) − F(x),  v = H·y,  w = H^T·p,  λ = p·v
    //   H ← H − (v + t·p)·w^T / λ
    const Point y = f_new - f;
    Point v = H * y;
    const value_type lambda = p_step.dot(v);
    if (abs(lambda) > EPS) {
      v += t * p_step;
      const Point w = H.transpose() * p_step;
      H -= (v * w.transpose()) / lambda;
    }

    // Step 7: advance iterate.
    p += t * p_step;
    f = f_new;
    phi = phi1;
  }

  last_phi = phi;
  return p;
}

template <diff::CExpression... FExprs>
Broyden(diff::Equation<FExprs...>) -> Broyden<FExprs...>;

template <diff::CExpression... FExprs, typename T>
Broyden(diff::Equation<FExprs...>, T) -> Broyden<FExprs...>;

} // namespace exprmin
