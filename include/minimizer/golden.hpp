#pragma once

#include "bracketmethod.hpp"
#include <cmath>

namespace exprmin {

/**
 * @brief NR §10.2 — Golden-section 1-D minimizer.
 *
 * Inherits from Bracketmethod to obtain a valid bracket [ax, bx, cx], then
 * narrows it by the golden ratio (≈ 0.61803) each iteration until the
 * interval width falls below @c tol × (|x1| + |x2|).  Convergence is linear;
 * @c tol should not be set smaller than √ε ≈ 3 × 10⁻⁸ for @c double.
 *
 * Results are accessible via get_optimal_x() and get_optimal_value() after
 * minimize() returns.
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 */
template <diff::CExpression Expr> struct Golden : Bracketmethod<Expr> {
  using Base = Bracketmethod<Expr>;
  using value_type = typename Base::value_type;
  using Base::ax;
  using Base::bracket;
  using Base::bx;
  using Base::cx;
  using Base::eval_at;
  using Base::fa;
  using Base::fb;
  using Base::fc;

  // R = 1/φ ≈ 0.61803 — contraction ratio (NOT φ itself)
  static constexpr diff::Constant<value_type> R{1.0 /
                                                std::numbers::phi_v<double>};
  static constexpr diff::Constant<value_type> C{
      1.0 - 1.0 / std::numbers::phi_v<double>};

private:
  value_type xmin{};
  value_type fmin{};
  const value_type tol;

public:
  /// @brief Returns f(xmin) after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fmin; }

  /// @brief Returns the minimizer abscissa after the last minimize() call.
  constexpr value_type get_optimal_x() const { return xmin; }

  /**
   * @brief Constructs a Golden searcher wrapping the given expression.
   * @param e     Expression to minimize.
   * @param tol_  Convergence tolerance (default √ε ≈ 3 × 10⁻⁸).
   */
  constexpr explicit Golden(Expr e,
                            value_type tol_ = static_cast<value_type>(3.0e-8))
      : Base(std::move(e)), tol(tol_) {}

  /**
   * @brief Runs golden-section search on the current bracket [ax, cx].
   * @pre   bracket() must have been called (or use the two-argument overload).
   * @return Minimizer abscissa xmin.
   */
  constexpr value_type minimize();

  /**
   * @brief Brackets [ax0, bx0] first, then runs golden-section search.
   * @param ax0  Left initial guess.
   * @param bx0  Right initial guess.
   * @return Minimizer abscissa xmin.
   */
  constexpr value_type minimize(const value_type &ax0, const value_type &bx0) {
    bracket(ax0, bx0);
    return minimize();
  }
};

template <diff::CExpression Expr>
constexpr typename Golden<Expr>::value_type Golden<Expr>::minimize() {
  value_type x0 = ax, x3 = cx, x1{}, x2{};
  if (detail::abs_for_constexpr(cx - bx) > detail::abs_for_constexpr(bx - ax)) {
    x1 = bx;
    x2 = bx + C * (cx - bx);
  } else {
    x2 = bx;
    x1 = bx - C * (bx - ax);
  } // x0–x1 is now the smaller sub-interval

  auto f1 = eval_at(x1); // bracket endpoints ax, cx already have known f-values
  auto f2 = eval_at(
      x2); // from bracket(); only interior trial points are evaluated here

  while (detail::abs_for_constexpr(x3 - x0) >
         tol * (detail::abs_for_constexpr(x1) + detail::abs_for_constexpr(x2))) {
    if (f2 < f1) { // minimum in [x1, x3]: discard x0, shift left boundary right
      x0 = x1;
      x1 = x2;
      x2 = R * x2 + C * x3;
      f1 = f2;
      f2 = eval_at(x2);
    } else { // minimum in [x0, x2]: discard x3, shift right boundary left
      x3 = x2;
      x2 = x1;
      x1 = R * x1 + C * x0;
      f2 = f1;
      f1 = eval_at(x1);
    }
  }
  if (f1 < f2) {
    xmin = x1;
    fmin = f1;
  } else {
    xmin = x2;
    fmin = f2;
  }
  return xmin;
}

template <diff::CExpression Expr> Golden(Expr) -> Golden<Expr>;
template <diff::CExpression Expr, typename T> Golden(Expr, T) -> Golden<Expr>;

} // namespace exprmin
