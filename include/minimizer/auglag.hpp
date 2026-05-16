#pragma once

#include "detail.hpp"
#include "equation.hpp"
#include "expressions.hpp"
#include "gradient.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

namespace exprmin {

namespace mp = boost::mp11;

// Trait: number of constraints for std::tuple<> (0) or diff::Equation<...>
// (output_dim).
namespace detail {
template <typename T>
struct constraint_count
    : std::integral_constant<std::size_t, std::tuple_size_v<T>> {};
template <diff::CExpression... Ts>
struct constraint_count<diff::Equation<Ts...>>
    : std::integral_constant<std::size_t, diff::Equation<Ts...>::output_dim> {};

template <typename T>
inline constexpr std::size_t constraint_count_v = constraint_count<T>::value;

} // namespace detail

// NLopt/Birgin-Martínez — Augmented Lagrangian constrained minimization.
//
// Minimizes f(x) subject to:
//   hᵢ(x) = 0  (equality,   EqConstraints   = diff::Equation<CExpression...>)
//   gⱼ(x) ≤ 0  (inequality, IneqConstraints = diff::Equation<CExpression...>)
//
// All constraint expressions must share the same symbol set as Obj.
//
// Outer loop: Birgin-Martínez multiplier/penalty updates.
// Inner loop: BFGS with backtracking Armijo on the augmented Lagrangian
//   L(x) = f(x) + Σ(ρ/2)(hᵢ + λᵢ/ρ)² + Σ(ρ/2) max(0, gⱼ + μⱼ/ρ)²
// EqConstraints / IneqConstraints: std::tuple<> (none) or diff::Equation<...>.
template <diff::CExpression Obj, typename EqConstraints = std::tuple<>,
          typename IneqConstraints = std::tuple<>>
struct AugLag {
  using value_type = typename Obj::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Obj>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  static constexpr std::size_t NEQ = detail::constraint_count_v<EqConstraints>;
  static constexpr std::size_t NINEQ =
      detail::constraint_count_v<IneqConstraints>;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

  // Birgin-Martínez magic constants
  static constexpr value_type TAU{0.5};
  static constexpr value_type GAM{10};
  static constexpr value_type LAM_MIN{-1e20};
  static constexpr value_type LAM_MAX{1e20};
  static constexpr value_type MU_MAX{1e20};
  static constexpr int INNER_ITMAX = 200;
  static constexpr int OUTER_ITMAX = 100;

  Obj obj;
  EqConstraints eq;
  IneqConstraints ineq;
  Eigen::Vector<value_type, static_cast<int>(NEQ)>
      lambda; // equality multipliers
  Eigen::Vector<value_type, static_cast<int>(NINEQ)>
      mu; // inequality multipliers
  value_type rho;
  value_type fret{};
  int iter{};
  const value_type ftol;

  constexpr explicit AugLag(Obj o, auto eq_ = {},
                            auto ineq_ = {},
                            value_type ftol_ = value_type{1e-8},
                            value_type rho0 = value_type{1}) :
  AugLag(std::move(o), EqConstraints{std::move(eq_)},
         IneqConstraints{std::move(ineq_)}, ftol_, rho0) {}

  constexpr explicit AugLag(Obj o, EqConstraints eq_ = {},
                            IneqConstraints ineq_ = {},
                            value_type ftol_ = value_type{1e-8},
                            value_type rho0 = value_type{1})
      : obj(std::move(o)), eq(std::move(eq_)), ineq(std::move(ineq_)),
        rho(rho0), ftol(ftol_) {
    if constexpr (NEQ > 0) {
      lambda.setZero();
    }
    if constexpr (NINEQ > 0) {
      mu.setZero();
    }
  }

  constexpr value_type eval_obj(const Point &x) {
    obj.update(Syms{}, x);
    return obj.eval();
  }

  // Assemble L(x) and ∇L(x) via reverse-mode AD on each expression.
  constexpr std::pair<value_type, Point> eval_aug(const Point &x) {
    obj.update(Syms{}, x);
    value_type L = obj.eval();
    const auto g0 = diff::gradient<diff::DiffMode::Reverse>(obj);
    Point g = Eigen::Map<const Point>(g0.data());

    if constexpr (NEQ > 0) {
      int ii = 0;
      eq.for_each_expr([&](auto &c) {
        c.update(Syms{}, x);
        const value_type h = c.eval() + lambda[ii] / rho;
        const auto hg0 = diff::gradient<diff::DiffMode::Reverse>(c);
        const Point hg = Eigen::Map<const Point>(hg0.data());
        L += value_type{0.5} * rho * h * h;
        g += rho * h * hg;
        ++ii;
      });
    }

    if constexpr (NINEQ > 0) {
      int ii = 0;
      ineq.for_each_expr([&](auto &c) {
        c.update(Syms{}, x);
        const value_type f = c.eval() + mu[ii] / rho;
        if (f > value_type{}) {
          const auto fg0 = diff::gradient<diff::DiffMode::Reverse>(c);
          const Point fg = Eigen::Map<const Point>(fg0.data());
          L += value_type{0.5} * rho * f * f;
          g += rho * f * fg;
        }
        ++ii;
      });
    }

    return {L, g};
  }

  // BFGS with backtracking Armijo on the augmented Lagrangian.
  constexpr Point inner_minimize(Point x) {
    return detail::bfgs_armijo<value_type, static_cast<int>(N)>(
        [this](const Point &p) { return eval_aug(p); }, std::move(x), ftol,
        INNER_ITMAX);
  }

  // Outer Birgin-Martínez loop.
  constexpr Point minimize(Point x) {
    using std::abs, std::max;

    // Estimate initial ρ from constraint violation at x₀ (B&M §3.2)
    if constexpr (NEQ > 0 || NINEQ > 0) {
      obj.update(Syms{}, x);
      const value_type fcur = obj.eval();
      value_type con2{};

      if constexpr (NEQ > 0) {
        eq.for_each_expr([&](auto &c) {
          c.update(Syms{}, x);
          const value_type h = c.eval();
          con2 += h * h;
        });
      }

      if constexpr (NINEQ > 0) {
        ineq.for_each_expr([&](auto &c) {
          c.update(Syms{}, x);
          const value_type f = c.eval();
          if (f > value_type{})
            con2 += f * f;
        });
      }

      if (con2 > value_type{}) {
        rho = std::clamp(value_type{2} * abs(fcur) / con2, value_type{1e-6},
                         value_type{10});
      }
    }

    value_type ICM = std::numeric_limits<value_type>::max();
    for (iter = 0; iter < OUTER_ITMAX; ++iter) {
      const value_type prev_ICM = ICM;

      x = inner_minimize(x);
      ICM = value_type{};

      // Update equality multipliers
      if constexpr (NEQ > 0) {
        int ii = 0;
        eq.for_each_expr([&](auto &c) {
          c.update(Syms{}, x);
          const value_type h = c.eval();
          ICM = max(ICM, abs(h));
          lambda[ii] = std::clamp(lambda[ii] + rho * h, LAM_MIN, LAM_MAX);
          ++ii;
        });
      }

      // Update inequality multipliers
      if constexpr (NINEQ > 0) {
        int ii = 0;
        ineq.for_each_expr([&](auto &c) {
          c.update(Syms{}, x);
          const value_type f = c.eval();
          ICM = max(ICM, abs(max(f, -mu[ii] / rho)));
          mu[ii] = std::clamp(mu[ii] + rho * f, value_type{}, MU_MAX);
          ++ii;
        });
      }

      if (ICM > TAU * prev_ICM) {
        rho *= GAM;
      }
      if (ICM <= ftol) {
        break;
      }
    }

    fret = eval_obj(x);
    return x;
  }
};

// Deduction guides
template <diff::CExpression O>
AugLag(O) -> AugLag<O, std::tuple<>, std::tuple<>>;

template <diff::CExpression O, typename E, typename I>
AugLag(O, E, I) -> AugLag<O, E, I>;

template <diff::CExpression O, typename E, typename I, typename T>
AugLag(O, E, I, T) -> AugLag<O, E, I>;

template <diff::CExpression O, typename E, typename I, typename T1, typename T2>
AugLag(O, E, I, T1, T2) -> AugLag<O, E, I>;

// Build constraint Equations.  make_eq(h1, h2) ≡ diff::Equation(h1, h2).
template <diff::CExpression... Cs> auto make_eq(Cs &&...cs) {
  return diff::Equation(std::forward<Cs>(cs)...);
}

template <diff::CExpression... Cs> auto make_ineq(Cs &&...cs) {
  return diff::Equation(std::forward<Cs>(cs)...);
}

} // namespace exprmin
