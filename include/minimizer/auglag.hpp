#pragma once

#include "bfgs.hpp"
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

namespace detail {

/**
 * @brief Compile-time constraint counter.
 *
 * Specialised for @c std::tuple<> (zero constraints) and for
 * @c diff::Equation<Ts...> (returns @c output_dim, i.e. the number of
 * scalar constraint expressions).  Used to size the multiplier vectors at
 * compile time.
 *
 * @tparam T  Either @c std::tuple<> or a @c diff::Equation specialisation.
 */
template <typename T>
struct constraint_count
    : std::integral_constant<std::size_t, std::tuple_size_v<T>> {};
template <diff::CExpression... Ts>
struct constraint_count<diff::Equation<Ts...>>
    : std::integral_constant<std::size_t, diff::Equation<Ts...>::output_dim> {};

/// @brief Convenience variable template for constraint_count<T>::value.
template <typename T>
inline constexpr std::size_t constraint_count_v = constraint_count<T>::value;

} // namespace detail

/**
 * @brief Augmented Lagrangian constrained minimizer (NLopt / Birgin–Martínez).
 *
 * Minimizes @f$f(\mathbf{x})@f$ subject to:
 * @f[
 *   h_i(\mathbf{x}) = 0 \quad (i = 1,\ldots,n_\text{eq}),
 *   \qquad
 *   g_j(\mathbf{x}) \le 0 \quad (j = 1,\ldots,n_\text{ineq})
 * @f]
 *
 * All constraint expressions must share the same symbol set as @p Obj.
 *
 * **Augmented Lagrangian** (the inner sub-problem objective):
 * @f[
 *   \mathcal{L}(\mathbf{x}) = f(\mathbf{x})
 *     + \sum_i \frac{\rho}{2}\!\left(h_i + \frac{\lambda_i}{\rho}\right)^{\!2}
 *     + \sum_j \frac{\rho}{2}\!\left[\max\!\left(0,\, g_j +
 *       \frac{\mu_j}{\rho}\right)\right]^{\!2}
 * @f]
 *
 * **Outer loop** (Birgin–Martínez §3.2, also mirrored in NLopt `auglag.c`):
 *
 * 1. **Initialise ρ** from constraint violation at @p x₀:
 *    @f$\rho_0 = \operatorname{clamp}(2|f_0|/\sum c_i^2,\;10^{-6},\;10)@f$.
 * 2. **Inner solve** — minimise @f$\mathcal{L}(\mathbf{x})@f$ w.r.t.
 *    @f$\mathbf{x}@f$ using BFGS with Armijo backtracking.
 * 3. **Multiplier updates**:
 *    - Equality:   @f$\lambda_i \leftarrow
 *      \operatorname{clamp}(\lambda_i + \rho h_i, \lambda_\min,
 *      \lambda_\max)@f$.
 *    - Inequality: @f$\mu_j \leftarrow
 *      \operatorname{clamp}(\mu_j + \rho g_j, 0, \mu_\max)@f$.
 * 4. **ICM** (infeasibility/constraint-violation measure):
 *    @f$\text{ICM} = \max_i|h_i|,\; \max_j|\max(g_j, -\mu_j/\rho)|@f$.
 * 5. **Penalty scaling**: if @f$\text{ICM} > \tau\cdot\text{ICM}_\text{prev}@f$
 *    then @f$\rho \leftarrow \gamma\rho@f$.
 * 6. **Convergence**: exit when @f$\text{ICM} \le \text{ftol}@f$.
 *
 * **Template parameters for constraints**
 * - Pass @c std::tuple<> for no constraints of a given type.
 * - Pass @c diff::Equation<CExpression...> (or use make_eq() / make_ineq())
 *   for one or more constraints.
 *
 * @tparam Obj             A type satisfying diff::CExpression (objective).
 * @tparam EqConstraints   @c std::tuple<> or @c diff::Equation<...> for
 * @f$h_i=0@f$.
 * @tparam IneqConstraints @c std::tuple<> or @c diff::Equation<...> for
 * @f$g_j\le0@f$.
 */
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

  // ── Birgin–Martínez algorithm constants ──────────────────────────────────

  /// @brief Penalty-scaling threshold τ: scale ρ if ICM > τ·ICM_prev (B&M
  /// §3.2).
  static constexpr value_type TAU{0.5};
  /// @brief Penalty growth factor γ: ρ ← γρ when constraint progress stalls.
  static constexpr value_type GAM{10};
  /// @brief Lower clamp bound for equality multipliers λᵢ.
  static constexpr value_type LAM_MIN{-1e20};
  /// @brief Upper clamp bound for equality multipliers λᵢ.
  static constexpr value_type LAM_MAX{1e20};
  /// @brief Upper clamp bound for inequality multipliers μⱼ (must stay ≥ 0).
  static constexpr value_type MU_MAX{1e20};
  /// @brief Maximum BFGS iterations per inner sub-problem solve.
  static constexpr int INNER_ITMAX = 200;
  /// @brief Maximum outer Birgin–Martínez iterations.
  static constexpr int OUTER_ITMAX = 100;

private:
  Obj obj;              ///< Objective expression.
  EqConstraints eq;     ///< Equality constraint expressions.
  IneqConstraints ineq; ///< Inequality constraint expressions.

  /// @brief Lagrange multiplier vector for equality constraints (size NEQ).
  Eigen::Vector<value_type, static_cast<int>(NEQ)> lambda;
  /// @brief Lagrange multiplier vector for inequality constraints (size NINEQ).
  Eigen::Vector<value_type, static_cast<int>(NINEQ)> mu;

  value_type rho;        ///< Current penalty parameter ρ.
  value_type fret{};     ///< Objective value at last minimizer found.
  int iter{};            ///< Outer-loop iteration count.
  const value_type ftol; ///< Convergence tolerance on ICM.

public:
  /**
   * @brief Forwarding constructor — accepts constraint arguments by universal
   * reference.
   *
   * Delegates to the concrete constructor after constructing @p EqConstraints
   * and @p IneqConstraints from the forwarded arguments.
   *
   * @param o      Objective expression.
   * @param eq_    Equality constraints (default: no constraints).
   * @param ineq_  Inequality constraints (default: no constraints).
   * @param ftol_  Convergence tolerance on ICM (default 10⁻⁸).
   * @param rho0   Initial penalty parameter ρ (default 1; auto-scaled in
   * minimize()).
   */
  constexpr explicit AugLag(Obj o, auto &&eq_ = {}, auto &&ineq_ = {},
                            value_type ftol_ = value_type{1e-8},
                            value_type rho0 = value_type{1})
      : AugLag(std::move(o), EqConstraints{std::forward<decltype(eq_)>(eq_)},
               IneqConstraints{std::forward<decltype(ineq_)>(ineq_)},
               std::move(ftol_), std::move(rho0)) {}

  /**
   * @brief Concrete constructor from fully-typed constraint objects.
   *
   * Initialises multiplier vectors to zero (B&M §3.1: cold start).
   *
   * @param o      Objective expression.
   * @param eq_    Equality constraint pack (type @p EqConstraints).
   * @param ineq_  Inequality constraint pack (type @p IneqConstraints).
   * @param ftol_  Convergence tolerance on ICM.
   * @param rho0   Initial penalty parameter ρ.
   */
  constexpr explicit AugLag(Obj o, EqConstraints eq_ = {},
                            IneqConstraints ineq_ = {},
                            value_type ftol_ = value_type{1e-8},
                            value_type rho0 = value_type{1})
      : obj(std::move(o)), eq(std::move(eq_)), ineq(std::move(ineq_)),
        rho(rho0), ftol(ftol_) {
    if constexpr (NEQ > 0)
      lambda.setZero();
    if constexpr (NINEQ > 0)
      mu.setZero();
  }

  /// @brief Returns @f$f(\mathbf{x}^*)@f$ at the last minimizer found.
  constexpr value_type get_optimal_value() const { return fret; }

  /**
   * @brief Run the Birgin–Martínez outer loop from starting point @p x.
   *
   * @param x  Initial primal point (modified in place during the loop).
   * @return   Approximate constrained minimizer.
   * @post     #fret holds @f$f(\mathbf{x}^*)@f$; #iter holds the outer-loop
   * count.
   */
  constexpr Point minimize(Point x);

private:
  /// @brief Evaluate @f$f(\mathbf{x})@f$ at @p x (no gradient).
  constexpr value_type eval_obj(const Point &x) {
    obj.update(Syms{}, x);
    return obj.eval();
  }

  /**
   * @brief Assemble the augmented Lagrangian value and gradient at @p x.
   *
   * Computes @f$\mathcal{L}(\mathbf{x})@f$ and @f$\nabla\mathcal{L}@f$ via
   * reverse-mode AD on each expression.  Both equality and inequality
   * penalty terms are accumulated; the inequality penalty is zero when
   * @f$g_j + \mu_j/\rho \le 0@f$ (constraint is satisfied with margin).
   *
   * @param x  Current primal point.
   * @return   @c {L, ∇L} pair consumed by the BFGS inner solver.
   */
  constexpr std::pair<value_type, Point> eval_aug(const Point &x);

  /**
   * @brief Solve the inner sub-problem: minimise @f$\mathcal{L}@f$ from @p x.
   *
   * Uses BFGS with Armijo backtracking (exprmin::bfgs_armijo) and the
   * current multipliers and penalty ρ frozen during the solve.
   *
   * @param x  Warm-start point.
   * @return   Approximate unconstrained minimizer of @f$\mathcal{L}@f$.
   */
  constexpr Point inner_minimize(Point x) {
    return exprmin::bfgs_armijo<value_type, static_cast<int>(N)>(
        [this](const Point &p) { return eval_aug(p); }, std::move(x), ftol,
        INNER_ITMAX);
  }
};

// ── eval_aug ─────────────────────────────────────────────────────────────────

template <diff::CExpression Obj, typename EqConstraints,
          typename IneqConstraints>
constexpr std::pair<
    typename AugLag<Obj, EqConstraints, IneqConstraints>::value_type,
    typename AugLag<Obj, EqConstraints, IneqConstraints>::Point>
AugLag<Obj, EqConstraints, IneqConstraints>::eval_aug(const Point &x) {
  // ── Step 1: objective contribution f(x) and ∇f(x). ──────────────────────
  obj.update(Syms{}, x);
  value_type L = obj.eval();
  const auto g0 = diff::gradient<diff::DiffMode::Reverse>(obj);
  Point g = Eigen::Map<const Point>(g0.data());

  // ── Step 2: equality penalty terms (ρ/2)(hᵢ + λᵢ/ρ)². ──────────────────
  // Shifted by λᵢ/ρ so the gradient at the Lagrangian saddle point vanishes.
  if constexpr (NEQ > 0) {
    int ii = 0;
    eq.for_each_expr([&](auto &c) {
      c.update(Syms{}, x);
      const value_type h = c.eval() + lambda[ii] / rho; // shifted violation
      const auto hg0 = diff::gradient<diff::DiffMode::Reverse>(c);
      const Point hg = Eigen::Map<const Point>(hg0.data());
      L += value_type{0.5} * rho * h * h; // penalty: (ρ/2)·h̃²
      g += rho * h * hg;                  // gradient: ρ·h̃·∇hᵢ
      ++ii;
    });
  }

  // ── Step 3: inequality penalty terms (ρ/2) max(0, gⱼ + μⱼ/ρ)². ─────────
  // The max(0,·) clamp activates the penalty only when the constraint is
  // violated after accounting for the current dual shift μⱼ/ρ.
  if constexpr (NINEQ > 0) {
    int ii = 0;
    ineq.for_each_expr([&](auto &c) {
      c.update(Syms{}, x);
      const value_type f = c.eval() + mu[ii] / rho; // shifted violation
      if (f > value_type{}) {                       // constraint active
        const auto fg0 = diff::gradient<diff::DiffMode::Reverse>(c);
        const Point fg = Eigen::Map<const Point>(fg0.data());
        L += value_type{0.5} * rho * f * f; // penalty: (ρ/2)·f̃²
        g += rho * f * fg;                  // gradient: ρ·f̃·∇gⱼ
      }
      ++ii;
    });
  }

  return {L, g};
}

// ── minimize (outer Birgin–Martínez loop) ────────────────────────────────────

template <diff::CExpression Obj, typename EqConstraints,
          typename IneqConstraints>
constexpr typename AugLag<Obj, EqConstraints, IneqConstraints>::Point
AugLag<Obj, EqConstraints, IneqConstraints>::minimize(Point x) {
  using std::abs, std::max;

  // ── Step 1: Auto-scale ρ₀ from constraint violation at x₀ (B&M §3.2). ──
  // Formula: ρ₀ = clamp(2|f₀| / Σcᵢ², 1e-6, 10).
  // Rationale: balance the objective scale against the initial violation so
  // neither the penalty nor the Lagrangian gradient is dominated by one term.
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
          con2 += f * f; // count only active violations
      });
    }

    if (con2 > value_type{}) {
      rho = std::clamp(value_type{2} * abs(fcur) / con2, value_type{1e-6},
                       value_type{10});
    }
  }

  // ── Outer loop ────────────────────────────────────────────────────────────
  value_type ICM = std::numeric_limits<value_type>::max();
  for (iter = 0; iter < OUTER_ITMAX; ++iter) {
    const value_type prev_ICM = ICM;

    // ── Step 2: Inner sub-problem — minimise L(x, λ, μ, ρ) w.r.t. x. ───
    // Multipliers and ρ are held fixed; BFGS-Armijo drives ∇ₓL → 0.
    x = inner_minimize(x);
    ICM = value_type{};

    // ── Step 3a: Update equality multipliers λᵢ ← λᵢ + ρ·hᵢ(x). ────────
    // Dual ascent step; clamped to [LAM_MIN, LAM_MAX] for safety.
    // ICM tracks max |hᵢ| as the feasibility metric for equality constraints.
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

    // ── Step 3b: Update inequality multipliers μⱼ ← max(0, μⱼ + ρ·gⱼ). ─
    // The max(0,·) enforces μⱼ ≥ 0 (dual feasibility for ≤ constraints).
    // ICM tracks the complementarity residual max(gⱼ, −μⱼ/ρ) per NLopt.
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

    // ── Step 4: Penalty scaling — grow ρ if feasibility progress is slow. ─
    // If ICM hasn't shrunk by at least a factor τ, multiply ρ by γ.
    // This forces the sub-problem to penalise constraint violations more
    // heavily in the next iteration, driving the iterates toward feasibility.
    if (ICM > TAU * prev_ICM) {
      rho *= GAM;
    }

    // ── Step 5: Convergence check — exit when ICM ≤ ftol. ────────────────
    if (ICM <= ftol) {
      break;
    }
  }

  fret = eval_obj(x);
  return x;
}

// ── Deduction guides ─────────────────────────────────────────────────────────

template <diff::CExpression O>
AugLag(O) -> AugLag<O, std::tuple<>, std::tuple<>>;

template <diff::CExpression O, typename E, typename I>
AugLag(O, E, I) -> AugLag<O, E, I>;

template <diff::CExpression O, typename E, typename I, typename T>
AugLag(O, E, I, T) -> AugLag<O, E, I>;

template <diff::CExpression O, typename E, typename I, typename T1, typename T2>
AugLag(O, E, I, T1, T2) -> AugLag<O, E, I>;

// ── Constraint factories ─────────────────────────────────────────────────────

/**
 * @brief Builds an equality-constraint @c diff::Equation from expressions.
 *
 * Each @p cs is a @c CExpression representing @f$h_k(\mathbf{x}) = 0@f$.
 * @code
 *   auto h = x + y - 1.0;
 *   AugLag al{f, make_eq(h), std::tuple{}};
 * @endcode
 */
template <diff::CExpression... Cs> auto make_eq(Cs &&...cs) {
  return diff::Equation(std::forward<Cs>(cs)...);
}

/**
 * @brief Builds an inequality-constraint @c diff::Equation from expressions.
 *
 * Each @p cs is a @c CExpression representing @f$g_k(\mathbf{x}) \le 0@f$.
 * @code
 *   auto g = x * x + y * y - 1.0;   // circle constraint g ≤ 0
 *   AugLag al{f, std::tuple{}, make_ineq(g)};
 * @endcode
 */
template <diff::CExpression... Cs> auto make_ineq(Cs &&...cs) {
  return diff::Equation(std::forward<Cs>(cs)...);
}

} // namespace exprmin
