#pragma once

#include "equation.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>

namespace exprmin {

namespace mp = boost::mp11;

// Broyden's rank-1 update method for finding roots of F(x) = 0.
//
// Port of GSL gsl_multiroot_fsolver_broyden (Broyden 1965), using exact
// reverse-mode AD Jacobian in place of finite differences.
//
// Usage: Broyden<F1, F2, ..., FN> where each Fi is a diff::CExpression.
// Pass the system as diff::Equation{f1, f2, ...} to the constructor.
// Requires a square system: N equations in N unknowns.
//
// Algorithm per iteration:
//   p    = H · f                   (Newton-like step; H ≈ −J⁻¹)
//   t    = line-search step size   (Hebden backtracking on ‖F‖)
//   y    = F(x+tp) − F(x)
//   H   ← H − (H·y + t·p)(p^T·H) / (p^T·H·y)   (rank-1 Broyden update)
//   fallback: if line search fails, refresh H = −J⁻¹ exactly (via AD)
template <diff::CExpression... FExprs>
struct Broyden {
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
    for (int i = 0; i < N; ++i)
      arr[static_cast<std::size_t>(i)] = p[i];
    return arr;
  }

  constexpr Point eval_f(const Point &p) {
    system.update(Syms{}, to_arr(p));
    Point f;
    if constexpr (N == 1) {
      f[0] = system.evaluate();
    } else {
      const auto f_arr = system.evaluate();
      for (int i = 0; i < N; ++i)
        f[i] = f_arr[static_cast<std::size_t>(i)];
    }
    return f;
  }

  // Exact Jacobian via reverse-mode AD; returns J(p).
  constexpr Matrix eval_J(const Point &p) {
    system.update(Syms{}, to_arr(p));
    const auto J_arr =
        system.template jacobian<diff::DiffMode::Reverse>();
    Matrix J;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        J(i, j) = J_arr[static_cast<std::size_t>(i)]
                        [static_cast<std::size_t>(j)];
    return J;
  }

  // H = −J(p)⁻¹  (exact inverse via LU).
  constexpr Matrix make_H(const Point &p) {
    return -(eval_J(p).lu().solve(Matrix::Identity()));
  }

public:
  constexpr value_type residual_norm() const { return last_phi; }

  constexpr explicit Broyden(diff::Equation<FExprs...> sys,
                              value_type tol_ = value_type{1e-10},
                              int itmax_ = 200)
      : system{std::move(sys)}, tol{tol_}, itmax{itmax_} {}

  constexpr Point find_root(Point p);

private:
  value_type last_phi{};
};

template <diff::CExpression... FExprs>
constexpr typename Broyden<FExprs...>::Point
Broyden<FExprs...>::find_root(Point p) {
  using std::sqrt, std::abs;
  constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();

  Point f = eval_f(p);
  Matrix H = make_H(p);
  value_type phi = f.norm();

  for (iter = 0; iter < itmax; ++iter) {
    last_phi = phi;
    if (phi < tol)
      break;

    // Newton-like step: p_step = H · f
    Point p_step = H * f;

    const value_type phi0 = phi;
    value_type t = value_type{1};
    Point f_new;
    value_type phi1{};

    // Hebden backtracking: shrink t while ‖F(x+t·p)‖ > ‖F(x)‖
    for (int ls = 0; ls < LS_ITMAX; ++ls) {
      f_new = eval_f(p + t * p_step);
      phi1 = f_new.norm();
      if (phi1 <= phi0 || t <= value_type{0.1})
        break;
      const value_type theta = phi1 / phi0;
      t *= (sqrt(value_type{1} + value_type{6} * theta) - value_type{1}) /
           (value_type{3} * theta);
    }

    // Fallback: refresh exact Jacobian and recompute step
    if (phi1 > phi0) {
      H = make_H(p);
      p_step = H * f;
      t = value_type{1};
      f_new = eval_f(p + t * p_step);
      phi1 = f_new.norm();
    }

    // Rank-1 Broyden update:
    //   v  = H·y  (then augmented to H·y + t·p)
    //   w  = H^T·p
    //   λ  = p · (H·y)
    //   H ← H − v·w^T / λ
    const Point y = f_new - f;
    Point v = H * y;
    const value_type lambda = p_step.dot(v);
    if (abs(lambda) > EPS) {
      v += t * p_step;
      const Point w = H.transpose() * p_step;
      H -= (v * w.transpose()) / lambda;
    }

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
