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
 * @brief Augmented Lagrangian constrained minimizer (Birgin & Martínez 2014;
 *        NLopt `auglag.c`).
 *
 * Minimises @f$ f(\mathbf{x}) @f$ subject to
 * @f[
 *   h_i(\mathbf{x}) = 0 \quad (i=1,\ldots,n_{\rm eq}),
 *   \qquad
 *   g_j(\mathbf{x}) \le 0 \quad (j=1,\ldots,n_{\rm ineq}).
 * @f]
 * All constraint expressions must share the same symbol set as @p Obj.
 *
 * ### Augmented Lagrangian function
 *
 * The inner sub-problem objective is the shifted-penalty augmented Lagrangian
 * (Birgin & Martínez eq. 3.1 / NLopt convention):
 * @f[
 *   \mathcal{L}(\mathbf{x}) = f(\mathbf{x})
 *     + \frac{\rho}{2}\sum_i \!\Bigl(h_i + \tfrac{\lambda_i}{\rho}\Bigr)^2
 *     + \frac{\rho}{2}\sum_j \!\Bigl[\max\!\Bigl(0,\,
 *           g_j + \tfrac{\mu_j}{\rho}\Bigr)\Bigr]^2.
 * @f]
 * The shifted form absorbs the dual variables so the gradient
 * @f$ \nabla_{\!\mathbf{x}}\mathcal{L} = 0 @f$ at the KKT point coincides
 * with the original first-order conditions.
 *
 * ### Outer loop  (B&M Algorithm 3.1)
 *
 * 1. **Auto-scale** @f$ \rho_0 @f$ from the initial constraint violation.
 * 2. **Inner minimise** @f$ \mathcal{L} @f$ w.r.t. @f$ \mathbf{x} @f$
 *    (BFGS + Armijo, multipliers/ρ frozen).
 * 3. **Dual update**: @f$ \lambda_i \leftarrow \mathrm{clamp}(\lambda_i +
 *    \rho h_i) @f$; @f$ \mu_j \leftarrow \max(0,\mu_j+\rho g_j) @f$.
 * 4. **ICM** (infeasibility/complementarity measure): scalar summary of
 *    constraint violation used to drive convergence and penalty scaling.
 * 5. **Penalty growth**: if @f$ \mathrm{ICM} > \tau \cdot \mathrm{ICM}_{\rm
 *    prev} @f$ then @f$ \rho \leftarrow \gamma\rho @f$.
 * 6. **Convergence**: exit when @f$ \mathrm{ICM} \le \varepsilon @f$.
 *
 * ### Template parameters for constraints
 * - Pass @c std::tuple<> for no constraints of a given type.
 * - Pass @c diff::Equation<CExpression...> (or make_eq() / make_ineq())
 *   for one or more constraints.
 *
 * @tparam Obj             A type satisfying diff::CExpression (objective).
 * @tparam EqConstraints   @c std::tuple<> or @c diff::Equation<...> for
 *                         @f$ h_i = 0 @f$.
 * @tparam IneqConstraints @c std::tuple<> or @c diff::Equation<...> for
 *                         @f$ g_j \le 0 @f$.
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

  // ── Birgin–Martínez algorithm constants (B&M §3) ─────────────────────────

  /// @brief Penalty-scaling threshold τ: grow ρ when ICM > τ·ICM_prev (B&M
  /// §3.2).
  static constexpr value_type TAU{0.5};
  /// @brief Penalty growth factor γ: ρ ← γρ when constraint progress stalls.
  static constexpr value_type GAM{10};
  /// @brief Lower clamp bound for equality multipliers λᵢ.
  static constexpr value_type LAM_MIN{-1e20};
  /// @brief Upper clamp bound for equality multipliers λᵢ.
  static constexpr value_type LAM_MAX{1e20};
  /// @brief Upper clamp bound for inequality multipliers μⱼ (dual feasibility:
  /// μⱼ ≥ 0).
  static constexpr value_type MU_MAX{1e20};
  /// @brief Maximum BFGS iterations per inner sub-problem solve.
  static constexpr int INNER_ITMAX = 200;
  /// @brief Maximum outer Birgin–Martínez iterations.
  static constexpr int OUTER_ITMAX = 100;

private:
  Obj obj;              ///< Objective expression.
  EqConstraints eq;     ///< Equality constraint expressions.
  IneqConstraints ineq; ///< Inequality constraint expressions.

  /// @brief Lagrange multipliers for equality constraints λ ∈ ℝ^NEQ (B&M §3.1).
  Eigen::Vector<value_type, static_cast<int>(NEQ)> lambda;
  /// @brief Lagrange multipliers for inequality constraints μ ∈ ℝ^NINEQ, μ ≥ 0.
  Eigen::Vector<value_type, static_cast<int>(NINEQ)> mu;

  value_type rho;    ///< Current penalty parameter ρ > 0.
  value_type fret{}; ///< Objective value f(x*) at the last minimizer.
  int iter{};        ///< Outer-loop iteration count on last minimize() call.
  const value_type ftol; ///< Convergence tolerance on ICM.

public:
  /**
   * @brief Forwarding constructor — accepts constraint arguments by universal
   *        reference and delegates to the typed constructor.
   *
   * @param o      Objective expression.
   * @param eq_    Equality constraints (default: no constraints).
   * @param ineq_  Inequality constraints (default: no constraints).
   * @param ftol_  ICM convergence tolerance (default 10⁻⁸).
   * @param rho0   Initial penalty ρ (default 1; auto-scaled in minimize()).
   */
  constexpr explicit AugLag(Obj o, auto &&eq_ = {}, auto &&ineq_ = {},
                            value_type ftol_ = value_type{1e-8},
                            value_type rho0 = value_type{1})
      : AugLag(std::move(o), EqConstraints{std::forward<decltype(eq_)>(eq_)},
               IneqConstraints{std::forward<decltype(ineq_)>(ineq_)},
               std::move(ftol_), std::move(rho0)) {}

  /**
   * @brief Typed constructor.  Initialises multiplier vectors to zero
   *        (B&M §3.1 cold start: λ = 0, μ = 0).
   *
   * @param o      Objective expression.
   * @param eq_    Equality constraint pack (type @p EqConstraints).
   * @param ineq_  Inequality constraint pack (type @p IneqConstraints).
   * @param ftol_  ICM convergence tolerance.
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

  /// @brief Returns @f$ f(\mathbf{x}^*) @f$ at the last minimizer found.
  constexpr value_type get_optimal_value() const { return fret; }

  /**
   * @brief Run the Birgin–Martínez outer loop from starting point @p x.
   *
   * @param x  Initial primal point.
   * @return   Approximate constrained minimizer @f$ \mathbf{x}^* @f$.
   * @post     #fret holds @f$ f(\mathbf{x}^*) @f$; #iter holds the outer count.
   */
  constexpr Point minimize(Point x);

private:
  /// @brief Evaluate @f$ f(\mathbf{x}) @f$ at @p x (value only, no gradient).
  constexpr value_type eval_obj(const Point &x) {
    obj.update(Syms{}, x);
    return obj.eval();
  }

  /**
   * @brief Assemble the augmented Lagrangian @f$ \mathcal{L} @f$ and its
   *        gradient @f$ \nabla_{\!\mathbf{x}}\mathcal{L} @f$ at @p x.
   *
   * All reverse-mode AD sweeps are performed in a single forward pass over the
   * expression graph.  Equality and inequality penalty terms are accumulated
   * independently; the inequality penalty is zero when the shifted constraint
   * value @f$ g_j + \mu_j/\rho \le 0 @f$ (constraint satisfied with margin).
   *
   * @param x  Current primal point.
   * @return   @c {L, ∇L} pair consumed by bfgs_armijo.
   */
  constexpr std::pair<value_type, Point> eval_aug(const Point &x);

  /**
   * @brief Solve the inner sub-problem: minimise @f$ \mathcal{L}(\cdot) @f$
   *        from warm-start @p x with multipliers and ρ frozen.
   *
   * Uses BFGS with Armijo backtracking (bfgs_armijo), which requires only
   * value + gradient evaluations and imposes no curvature on the objective.
   *
   * @param x  Warm-start point (previous outer iterate).
   * @return   Approximate unconstrained minimizer of @f$ \mathcal{L} @f$.
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

  // ── Step 1: objective term — f(x) and ∇f(x) ──────────────────────────────
  // Seed the expression at x and run reverse-mode AD to get both value and
  // gradient in one pass over the expression graph.
  obj.update(Syms{}, x);
  value_type L = obj.eval();
  const auto g0 = diff::gradient<diff::DiffMode::Reverse>(obj);
  Point g = Eigen::Map<const Point>(g0.data());

  // ── Step 2: equality penalty — (ρ/2)(hᵢ + λᵢ/ρ)² per constraint ─────────
  // The shift λᵢ/ρ is the Rockafellar trick: at the exact Lagrangian saddle
  // point hᵢ = 0 ⟹ the shifted value h̃ = hᵢ + λᵢ/ρ = λᵢ/ρ ≠ 0 in general,
  // but ρ·h̃ = λᵢ + ρhᵢ matches the dual update, so ∇L = ∇f + Σλᵢ∇hᵢ + O(ρ)
  // recovers the KKT stationarity condition as ρ → ∞ (B&M §2.2).
  // Gradient contribution: ρ·h̃·∇hᵢ  (chain rule through the square).
  if constexpr (NEQ > 0) {
    int ii = 0;
    eq.for_each_expr([&](auto &c) {
      c.update(Syms{}, x);
      const value_type h = c.eval() + lambda[ii] / rho; // shifted violation h̃
      const auto hg0 = diff::gradient<diff::DiffMode::Reverse>(c);
      const Point hg = Eigen::Map<const Point>(hg0.data());
      L += value_type{0.5} * rho * h * h; // (ρ/2)·h̃²
      g += rho * h * hg;                  // ρ·h̃·∇hᵢ
      ++ii;
    });
  }

  // ── Step 3: inequality penalty — (ρ/2) max(0, gⱼ + μⱼ/ρ)² per constraint ─
  // The outer max(0,·) is the indicator for active constraints: the penalty is
  // zero when the shifted value f̃ = gⱼ + μⱼ/ρ ≤ 0, i.e. when gⱼ ≤ −μⱼ/ρ
  // (the constraint is satisfied with a slack at least μⱼ/ρ).  This matches
  // the complementary slackness condition μⱼ·gⱼ = 0 at the KKT point.
  // Gradient is zero in the inactive case (subdifferential of max(0,·) at 0).
  if constexpr (NINEQ > 0) {
    int ii = 0;
    ineq.for_each_expr([&](auto &c) {
      c.update(Syms{}, x);
      const value_type f = c.eval() + mu[ii] / rho; // shifted violation f̃
      if (f > value_type{}) {                       // constraint active
        const auto fg0 = diff::gradient<diff::DiffMode::Reverse>(c);
        const Point fg = Eigen::Map<const Point>(fg0.data());
        L += value_type{0.5} * rho * f * f; // (ρ/2)·f̃²
        g += rho * f * fg;                  // ρ·f̃·∇gⱼ
      }
      ++ii;
    });
  }

  return {L, g};
}

// ── minimize (outer Birgin–Martínez loop, B&M Algorithm 3.1) ─────────────────

template <diff::CExpression Obj, typename EqConstraints,
          typename IneqConstraints>
constexpr typename AugLag<Obj, EqConstraints, IneqConstraints>::Point
AugLag<Obj, EqConstraints, IneqConstraints>::minimize(Point x) {
  using std::abs, std::max;

  // ── Step 1: auto-scale ρ₀ from the constraint violation at x₀ ─────────────
  // Formula (NLopt auglag.c §init_penalty):
  //   ρ₀ = clamp(2|f(x₀)| / Σcᵢ(x₀)², 10⁻⁶, 10)
  // Rationale: the penalty term ρΣcᵢ² should be of the same order as |f| so
  // neither the objective nor the constraint violation dominates ∇L.  Without
  // scaling, a large |f| relative to ‖c‖ leads to poorly penalised constraint
  // violations, while a tiny |f| leads to an overly stiff sub-problem.
  // Only active inequality violations are included in Σcᵢ² (same convention as
  // the penalty in eval_aug step 3).
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
          con2 += f * f; // only active violations
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

    // ── Step 2: inner sub-problem — minimise L(x; λ, μ, ρ) w.r.t. x ─────────
    // Multipliers λ, μ and penalty ρ are treated as constants during this
    // solve. BFGS-Armijo drives ∇ₓL → 0; warm-started from the previous iterate
    // so each inner solve needs far fewer steps than a cold start (B&M
    // Remark 3.1).
    x = inner_minimize(x);
    ICM = value_type{};

    // ── Step 3a: equality multiplier update — λᵢ ← clamp(λᵢ + ρhᵢ(x)) ───────
    // This is the standard dual ascent step for equality constraints
    // (B&M eq. 3.3a / Rockafellar 1974).  Adding ρhᵢ pushes λ toward the KKT
    // multiplier; clamping to [LAM_MIN, LAM_MAX] prevents overflow.
    // ICM accumulates max|hᵢ| as the equality feasibility metric.
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

    // ── Step 3b: inequality multiplier update — μⱼ ← max(0, μⱼ + ρgⱼ(x)) ───
    // The max(0,·) enforces dual feasibility μⱼ ≥ 0 required for ≤ constraints.
    // ICM uses the NLopt complementarity metric: max(gⱼ, −μⱼ/ρ).
    //   • gⱼ > 0: constraint violated — direct infeasibility measure.
    //   • gⱼ ≤ 0 but −μⱼ/ρ > 0: complementary slackness violated (μⱼ < 0
    //     would be dual infeasible, but we clamp so this branch means μⱼ = 0
    //     after the previous update and gⱼ < 0 — complementarity holds, metric
    //     = 0).
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

    // ── Step 4: penalty scaling — grow ρ if feasibility progress is slow
    // ────── If the ICM reduction relative to the previous iteration is less
    // than τ, the current ρ is not large enough to force the iterates toward
    // feasibility. Multiplying by γ stiffens the penalty, trading a harder
    // inner sub-problem for faster constraint satisfaction (B&M §3.2,
    // Proposition 3.2).
    if (ICM > TAU * prev_ICM) {
      rho *= GAM;
    }

    // ── Step 5: convergence — exit when ICM ≤ ftol
    // ──────────────────────────── ICM ≤ ftol means all equality residuals |hᵢ|
    // and all complementarity residuals max(gⱼ, −μⱼ/ρ) are below the requested
    // tolerance; the primal iterate is approximately feasible (B&M §3.2,
    // stopping criterion S1).
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
 * Each @p cs is a @c CExpression representing @f$ h_k(\mathbf{x}) = 0 @f$.
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
 * Each @p cs is a @c CExpression representing @f$ g_k(\mathbf{x}) \le 0 @f$.
 * @code
 *   auto g = x * x + y * y - 1.0;   // unit-disk constraint
 *   AugLag al{f, std::tuple{}, make_ineq(g)};
 * @endcode
 */
template <diff::CExpression... Cs> auto make_ineq(Cs &&...cs) {
  return diff::Equation(std::forward<Cs>(cs)...);
}

} // namespace exprmin
