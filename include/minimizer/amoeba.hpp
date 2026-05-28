#pragma once

#include "detail.hpp"
#include "expressions.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <ranges>

namespace exprmin {

namespace mp = boost::mp11;

/**
 * @brief NR §10.5 — Nelder–Mead downhill simplex minimizer.
 *
 * Derivative-free N-dimensional minimizer.  Maintains a simplex of N+1
 * vertices, progressively shrinking it toward the minimum via reflection,
 * expansion, and contraction moves.  No line search or gradient required.
 *
 * The simplex is stored as an N×(N+1) Eigen matrix where each column is a
 * vertex; @c FVals is an (N+1)-vector of the corresponding function values.
 *
 * **amotry step convention** (NR §10.5): @c psum is the column-sum of all
 * N+1 vertices.  For reflection factor @c fac:
 * @code
 *   fac1 = (1 − fac) / N,  fac2 = fac1 − fac
 *   ptry = psum·fac1 − s.col(ihi)·fac2
 *        = c + fac·(c − s.col(ihi)),   c = centroid of the remaining N vertices
 * @endcode
 * Canonical values: fac = −1 (reflect), 2 (expand), 0.5 (contract).
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 */
template <diff::CExpression Expr> struct Amoeba {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;
  using Simplex =
      Eigen::Matrix<value_type, static_cast<int>(N), static_cast<int>(N + 1)>;
  using FVals = Eigen::Vector<value_type, static_cast<int>(N + 1)>;

  static constexpr value_type TINY{1.0e-10};
  static constexpr int ITMAX = 5000;

private:
  Expr expr;
  value_type fret{};
  int iter{};
  const value_type ftol;

public:
  /**
   * @brief Constructs an Amoeba minimizer wrapping the given expression.
   * @param e      Expression to minimize.
   * @param ftol_  Convergence tolerance on the relative function-value spread
   *               between the best and worst simplex vertices (default 3×10⁻⁸).
   */
  constexpr explicit Amoeba(Expr e,
                            value_type ftol_ = static_cast<value_type>(3.0e-8))
      : expr{std::move(e)}, ftol{ftol_} {}

  /**
   * @brief Evaluates the expression at point @p p.
   * @param p  N-dimensional input point.
   * @return   f(p).
   */
  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  /// @brief Callable interface for eval_at — lets Amoeba act as a functor for amotry_impl.
  constexpr value_type operator()(const Point &p) { return eval_at(p); }

  /// @brief Returns f at the best vertex after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fret; }

  /**
   * @brief Builds an initial simplex around @p p and minimizes.
   *
   * Constructs the simplex via detail::make_simplex (vertex 0 = p; vertex i+1
   * = p with component i perturbed by @p delta), then delegates to the
   * Simplex overload.
   *
   * @param p      Initial centre point.
   * @param delta  Side-length of the initial simplex.
   * @return Approximate minimizer (best vertex on convergence or ITMAX).
   */
  constexpr Point minimize(const Point &p, const value_type &delta) {
    return minimize(detail::make_simplex(p, delta));
  }

  /**
   * @brief Runs the Nelder–Mead loop on a pre-built simplex.
   * @param s  N×(N+1) matrix whose columns are the initial vertices.
   * @return Approximate minimizer (best vertex on convergence or ITMAX).
   */
  constexpr Point minimize(Simplex s);
};

/**
 * @brief Nelder–Mead main loop.
 *
 * Each iteration:
 *  1. Identifies the best (ilo), worst (ihi), and second-worst (inhi) vertices.
 *  2. Checks convergence: exits when the relative spread |y[ihi]−y[ilo]| /
 *     (|y[ihi]|+|y[ilo]|) drops below ftol.
 *  3. **Reflect** ihi through the centroid (fac = −1).
 *     - If the reflection beats the current best → **expand** (fac = 2).
 *     - Else if reflection is still worse than second-worst → **contract**
 *       (fac = 0.5).  If contraction also fails → **shrink** every vertex
 *       halfway toward the best.
 *
 * Returns the best vertex found, with the function value stored in fret.
 */
template <diff::CExpression Expr>
constexpr typename Amoeba<Expr>::Point Amoeba<Expr>::minimize(Simplex s) {
  FVals y;
  for (auto i : std::views::iota(0uz, N + 1)) {
    y[i] = eval_at(s.col(i));
  }

  Point psum = s.rowwise().sum();
  for (iter = 0; iter < ITMAX; ++iter) {
    // Indices of best (ilo), worst (ihi), second-worst (inhi)
    Eigen::Index ilo_idx;
    y.minCoeff(&ilo_idx);
    const auto ilo = static_cast<std::size_t>(ilo_idx);

    std::size_t ihi = (y[0] > y[1]) ? 0uz : 1uz;
    std::size_t inhi = 1uz - ihi;
    for (auto i : std::views::iota(2uz, N + 1)) {
      if (y[i] > y[ihi]) {
        inhi = ihi;
        ihi = i;
      } else if (y[i] > y[inhi] && i != ihi) {
        inhi = i;
      }
    }

    // Convergence: relative spread between best and worst
    if (value_type{2} * std::abs(y[ihi] - y[ilo]) /
            (std::abs(y[ihi]) + std::abs(y[ilo]) + TINY) <
        ftol) {
      fret = y[ilo];
      return s.col(ilo);
    }
    value_type ytry = detail::amotry_impl<value_type, N>(*this, s, y, psum, ihi,
                                                         value_type{-1});
    if (ytry <= y[ilo]) {
      detail::amotry_impl<value_type, N>(*this, s, y, psum, ihi, value_type{2});
    } else if (ytry >= y[inhi]) {
      const value_type ysave = y[ihi];
      ytry = detail::amotry_impl<value_type, N>(*this, s, y, psum, ihi,
                                                value_type{0.5});
      if (ytry >= ysave) {
        // Contraction failed — shrink whole simplex toward best
        for (auto i : std::views::iota(0uz, N + 1)) {
          if (i != ilo) {
            s.col(i) = value_type{0.5} * (s.col(i) + s.col(ilo));
            y[i] = eval_at(s.col(i));
          }
        }
        psum = s.rowwise().sum();
      }
    }
  }
  Eigen::Index ilo_idx;
  y.minCoeff(&ilo_idx);
  fret = y[ilo_idx];
  return s.col(ilo_idx);
}

template <diff::CExpression Expr> Amoeba(Expr) -> Amoeba<Expr>;
template <diff::CExpression Expr, typename T> Amoeba(Expr, T) -> Amoeba<Expr>;

} // namespace exprmin
