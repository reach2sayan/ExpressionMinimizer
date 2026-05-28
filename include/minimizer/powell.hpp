#pragma once

#include "linmin.hpp"

namespace exprmin {

/**
 * @brief NR §10.6 — Powell's method for N-dimensional minimization.
 *
 * Successively line-minimizes along N conjugate directions, starting from
 * the coordinate axes.  After each outer iteration the direction that produced
 * the largest single-step decrease is replaced by the net step vector taken
 * during that iteration.  This progressively builds a set of mutually
 * conjugate directions without requiring any derivatives.
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 */
template <diff::CExpression Expr> struct Powell {
  using value_type = typename LinMin<Expr>::value_type;
  using Point = typename LinMin<Expr>::Point;
  static constexpr std::size_t N = LinMin<Expr>::N;
  /// N×N matrix whose columns are the current search directions.
  using Dirs =
      Eigen::Matrix<value_type, static_cast<int>(N), static_cast<int>(N)>;

  static constexpr diff::Constant<value_type> FTINY{1.0e-25};
  static constexpr int ITMAX = 200;

private:
  LinMin<Expr> lm;
  value_type fret{};
  int iter{};
  const value_type ftol;

public:
  /**
   * @brief Constructs a Powell minimizer wrapping the given expression.
   * @param e      Expression to minimize.
   * @param ftol_  Convergence tolerance on the relative function-value change
   *               per outer iteration (default 3×10⁻⁸).
   */
  constexpr explicit Powell(Expr e,
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

  /**
   * @brief Minimizes starting from @p p along the N coordinate axes.
   *
   * Initialises the direction matrix to the identity and delegates to the
   * two-argument overload.
   *
   * @param p  Initial point.
   * @return Approximate minimizer.
   */
  constexpr Point minimize(Point p) {
    return minimize(std::move(p), Dirs::Identity());
  }

  /**
   * @brief Minimizes starting from @p p using the provided direction set @p xi.
   * @param p   Initial point.
   * @param xi  N×N matrix whose columns are the initial search directions.
   * @return Approximate minimizer.
   */
  constexpr Point minimize(Point p, Dirs xi);

  /// @brief Returns f at the best point after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fret; }
};

/**
 * @brief Powell's method main loop.
 *
 * Each outer iteration:
 *  1. **Inner sweep**: line-minimizes along every direction in @p xi, tracking
 *     which direction @c ibig produced the largest decrease @c del.
 *  2. **Convergence check**: returns if the relative change in @c fret is
 *     below @c ftol.
 *  3. **Direction replacement** (NR heuristic): computes the extrapolated
 *     point @c ptt = 2p − pt and net step @c xit = p − pt.  Replaces @c xi[ibig]
 *     with @c xi[N−1] and appends @c xit only when both conditions hold:
 *       - f(ptt) < f(pt)  — the extrapolated direction still descends, and
 *       - 2(fp − 2·fret + f(ptt))·(fp − fret − del)² < (fp − f(ptt))²·del
 *         — the direction of maximum decrease is not already "used up" by
 *           the new direction, so replacing it preserves conjugacy.
 */
template <diff::CExpression Expr>
constexpr typename Powell<Expr>::Point Powell<Expr>::minimize(Point p,
                                                              Dirs xi) {
  using std::abs;
  fret = eval_at(p);
  Point pt = p; // position at the start of the current outer iteration

  for (iter = 0; iter < ITMAX; ++iter) {
    const value_type fp = fret; // function value at start of this iteration
    int ibig = 0;
    value_type del{}; // largest decrease seen across all directions this sweep

    // Inner sweep: line-minimize along each direction in turn
    for (int i = 0; i < static_cast<int>(N); ++i) {
      const value_type fptt = fret;
      lm.minimize(p, xi.col(i));
      fret = lm.get_optimal_value();
      if (abs(fptt - fret) > del) {
        del = abs(fptt - fret);
        ibig = i;
      }
    }

    // check for termination
    if (value_type{2} * abs(fp - fret) <=
        ftol * (abs(fp) + abs(fret)) + FTINY.get()) {
      return p;
    }

    // ptt: extrapolated point one net-step beyond p
    // xit: net displacement taken this iteration (new candidate direction)
    const Point ptt = value_type{2} * p - pt;
    Point xit = p - pt;
    pt = p;

    const value_type fptt = eval_at(ptt);
    if (fptt < fp) {
      const value_type a = fp - fret - del;
      const value_type b = fp - fptt;
      // NR conjugacy guard: only replace if the new direction adds value
      // without degrading the direction that was already working best
      if (value_type{2} * (fp - value_type{2} * fret + fptt) * a * a <
          b * b * del) {
        lm.minimize(p, xit);
        fret = lm.get_optimal_value();
        // drop ibig (biggest-decrease direction) to make room for xit
        xi.col(ibig) = xi.col(static_cast<int>(N) - 1);
        xi.col(static_cast<int>(N) - 1) = xit;
      }
    }
  }
  return p;
}

template <diff::CExpression Expr> Powell(Expr) -> Powell<Expr>;
template <diff::CExpression Expr, typename T> Powell(Expr, T) -> Powell<Expr>;

} // namespace exprmin
