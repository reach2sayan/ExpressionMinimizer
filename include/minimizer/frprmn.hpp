#pragma once

#include "gradient.hpp"
#include "linmin.hpp"
#include <Eigen/Dense>
#include <limits>
#include <utility>

namespace exprmin {

/**
 * @brief Selects the beta formula used to update the conjugate direction.
 *
 * Both variants share the same iteration structure; only the numerator of the
 * scaling factor γ = dgg / gg differs:
 *   - **FletcherReeves**: dgg = ‖g_new‖²  — simple, guaranteed descent with
 *     exact line search, but can restart poorly after near-zero gradients.
 *   - **PolakRibiere**: dgg = (g_new − g)·g_new — incorporates the change in
 *     gradient, effectively self-restarting when the new and old gradients are
 *     nearly parallel (γ ≈ 0), which improves practical convergence.
 */
enum class CGMethod { FletcherReeves, PolakRibiere };

/**
 * @brief NR §10.7 — Conjugate gradient minimizer (Fletcher–Reeves /
 * Polak–Ribière).
 *
 * Gradient ∇f is obtained for free via reverse-mode AD on the expression tree.
 * Line minimization is delegated to the @p LM template parameter:
 *   - LinMin  (default) — Brent derivative-free line search.
 *   - DLinMin — Dbrent line search exploiting the directional derivative
 *               (use via the @ref DFrprmn alias).
 *
 * @tparam Expr    A type satisfying the diff::CExpression concept.
 * @tparam Method  CGMethod::PolakRibiere (default) or CGMethod::FletcherReeves.
 * @tparam LM      Line-minimizer template; must provide eval_at, eval_grad,
 *                 minimize(Point&, dir), and get_optimal_value().
 */
template <diff::CExpression Expr, CGMethod Method = CGMethod::PolakRibiere,
          template <diff::CExpression> class LM = LinMin>
struct Frprmn {
  using LMType = LM<Expr>;
  using value_type = typename LMType::value_type;
  using Point = typename LMType::Point;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = LMType::N;

  static constexpr diff::Constant<value_type> ZEPS{
      std::numeric_limits<value_type>::epsilon() * 1.0e-3};
  static constexpr int ITMAX = 200;

private:
  LMType lm;
  value_type fret{};
  int iter{};
  const value_type ftol;

public:
  /**
   * @brief Constructs a Frprmn minimizer wrapping the given expression.
   * @param e      Expression to minimize.
   * @param ftol_  Convergence tolerance on the relative function-value change
   *               per iteration (default 3×10⁻⁸).
   */
  constexpr explicit Frprmn(Expr e,
                            value_type ftol_ = static_cast<value_type>(3.0e-8))
      : lm(std::move(e), ftol_), ftol(ftol_) {}

  /// @brief Callable interface delegating to eval_at.
  constexpr value_type operator()(const Point &p) { return lm.eval_at(p); }

  /**
   * @brief Evaluates the expression at point @p p.
   * @param p  N-dimensional input point.
   * @return   f(p).
   */
  constexpr value_type eval_at(const Point &p) { return lm.eval_at(p); }

  /// @brief Returns f at the best point after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fret; }

  /**
   * @brief Evaluates the expression and its reverse-mode gradient at @p p.
   * @param p  N-dimensional input point.
   * @return   {f(p), ∇f(p)} as an std::pair.
   */
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    return lm.eval_grad(p);
  }

  /**
   * @brief Runs conjugate gradient minimization from initial point @p p.
   * @param p  Initial point.
   * @return Approximate minimizer.
   */
  constexpr Point minimize(Point p);
};

/**
 * @brief Conjugate gradient main loop.
 *
 * **Initialization**: evaluates f and g = ∇f at p, then sets the first search
 * direction to steepest descent: xi = h = −g.
 *
 * **Each iteration**:
 *  1. Line-minimizes f along xi, advancing p to the new best point.
 *  2. Checks convergence: returns if the relative change in f < ftol.
 *  3. Recomputes g_new = ∇f(p_new).
 *  4. Computes the CG scaling factor γ = dgg / gg where:
 *       - gg  = ‖g‖²  (squared norm of the old gradient)
 *       - dgg = (g_new − g)·g_new  [Polak–Ribière]  or  ‖g_new‖²
 * [Fletcher–Reeves] If gg = 0 the gradient has vanished and we return
 * immediately.
 *  5. Updates the conjugate direction: h = −g_new + γ·h,  xi = h.
 *     γ = 0 restarts to steepest descent; γ > 0 carries conjugacy forward.
 */
template <diff::CExpression Expr, CGMethod Method,
          template <diff::CExpression> class LM>
constexpr typename Frprmn<Expr, Method, LM>::Point
Frprmn<Expr, Method, LM>::minimize(Point p) {
  using std::abs;

  auto fg0 = eval_grad(p);
  value_type fp = fg0.first;
  Point g = std::move(fg0.second);
  fret = fp;

  Point xi = -g; // first direction: pure steepest descent
  Point h = xi;  // h tracks the conjugate direction history

  for (iter = 0; iter < ITMAX; ++iter) {
    lm.minimize(p, xi); // line-minimize f along xi; p and xi updated in place
    fret = lm.get_optimal_value();

    // termination condition: relative change in f below tolerance
    if (value_type{2} * abs(fret - fp) <=
        ftol * (abs(fret) + abs(fp)) + ZEPS.get()) {
      return p;
    }

    fp = fret;
    Point g_new = eval_grad(p).second;

    const value_type gg = g.squaredNorm(); // ‖g_old‖²: denominator of γ

    // Numerator of γ: PR uses only the component of g_new orthogonal to g,
    // which gives γ ≈ 0 (steepest-descent restart) when the gradient barely
    // changed direction.  FR uses ‖g_new‖² regardless.
    const value_type dgg = [&] {
      if constexpr (Method == CGMethod::PolakRibiere) {
        return (g_new - g).dot(g_new);
      } else {
        return g_new.squaredNorm();
      }
    }();

    if (gg == value_type{}) {
      return p; // gradient vanished exactly — already at the minimum
    }

    const value_type gam = dgg / gg; // CG scaling factor γ
    h = -g_new + gam * h;            // update conjugate direction
    xi = h;                          // next line-search direction
    g = std::move(g_new);
  }
  return p;
}

template <diff::CExpression Expr> Frprmn(Expr) -> Frprmn<Expr>;
template <diff::CExpression Expr, diff::Numeric T>
Frprmn(Expr, T) -> Frprmn<Expr>;

/**
 * @brief Derivative-aware CG: uses Dbrent line search via DLinMin.
 *
 * Drop-in alias for Frprmn with the LM parameter fixed to DLinMin.  Each line
 * search exploits the directional derivative ∇f·d via reverse-mode AD, which
 * reduces the number of function evaluations per line search relative to
 * the plain Brent version.
 *
 * @tparam Expr    A type satisfying the diff::CExpression concept.
 * @tparam Method  CGMethod::PolakRibiere (default) or CGMethod::FletcherReeves.
 */
template <diff::CExpression Expr, CGMethod Method = CGMethod::PolakRibiere>
using DFrprmn = Frprmn<Expr, Method, DLinMin>;

} // namespace exprmin
