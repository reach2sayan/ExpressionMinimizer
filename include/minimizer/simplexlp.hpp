#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace exprmin {

/**
 * @brief Linear Programming via the two-phase full-tableau Simplex method.
 *
 * Solves the LP in natural (inequality) form:
 * @f[
 *   \min_x \; \mathbf{c}^\top \mathbf{x}
 *   \quad \text{subject to} \quad
 *   A\mathbf{x} \le \mathbf{b}, \quad \mathbf{x} \ge 0
 * @f]
 *
 * where @p A is \f$(m \times n)\f$, @p b is \f$(m)\f$, @p c is \f$(n)\f$.
 * The right-hand side @p b may have any sign.
 *
 * **Constraint encoding**
 * - @f$\ge@f$ constraint: negate the row \f$(-a_i^\top x \le -b_i)\f$.
 * - @f$=@f$ constraint: supply two rows \f$(+a_i^\top x \le b_i\f$ and
 *   \f$-a_i^\top x \le -b_i)\f$.
 *
 * **Algorithm** (NR3 §10.10, Dantzig 1948)\n
 * Slack variables are appended to reach standard form \f$Ax = b, x \ge 0\f$.
 * When any \f$b_i < 0\f$ the corresponding row is negated and an artificial
 * variable is added.  **Phase 1** minimises the sum of artificials; if the
 * minimum is non-zero the problem is infeasible.  **Phase 2** minimises the
 * original objective from the feasible basis found in Phase 1.
 *
 * After solve(), inspect #status for the outcome and #fret for the value.
 *
 * @tparam T  Scalar type (default: @c double).
 */
template <typename T = double> struct SimplexLP {

  /**
   * @brief Outcome of the last solve() call.
   *
   * - @c Optimal    — a minimiser was found; #fret holds the value.
   * - @c Infeasible — no feasible point exists (Phase 1 value > 0).
   * - @c Unbounded  — the objective is unbounded below.
   * - @c MaxIter    — iteration limit reached without convergence.
   */
  enum class Status { Optimal, Infeasible, Unbounded, MaxIter };

  using VecX = Eigen::VectorX<T>; ///< Dynamic column vector.
  using MatX = Eigen::MatrixX<T>; ///< Dynamic matrix.

  static constexpr int ITMAX = 10'000; ///< Maximum pivot iterations.
  static constexpr T EPS{1e-9};        ///< Pivot and feasibility tolerance.

  Status status{Status::MaxIter}; ///< Outcome set after each solve().
  T fret{};   ///< Optimal objective value (valid when Optimal).
  int iter{}; ///< Total pivot count of the last solve().

  /**
   * @brief Solve the LP.
   *
   * @param A  Constraint matrix \f$(m \times n)\f$.
   * @param b  Right-hand side \f$(m)\f$; may have any sign.
   * @param c  Objective coefficients \f$(n)\f$.
   * @return   Optimal primal vector \f$\mathbf{x} \in \mathbb{R}^n\f$,
   *           or the zero vector when #status is not @c Optimal.
   * @post     #status and #fret are updated.
   */
  VecX solve(const MatX &A, const VecX &b, const VecX &c);

private:
  using VecI = std::vector<int>;

  /**
   * @brief Gauss-Jordan pivot on tableau entry (q, p).
   *
   * Normalises row @p q so that @c tab(q,p)==1, then eliminates column @p p
   * from every other row.  Records the new basic variable in @p basis[q].
   *
   * @param tab    Full simplex tableau (in-place).
   * @param q      Pivot row (leaving variable row).
   * @param p      Pivot column (entering variable column).
   * @param basis  Basis column-index vector; updated to @c basis[q]=p.
   */
  static void do_pivot(MatX &tab, int q, int p, VecI &basis) {
    tab.row(q) /= tab(q, p);
    const int rows = static_cast<int>(tab.rows());
    for (int i = 0; i < rows; ++i)
      if (i != q && tab(i, p) != T{0})
        tab.row(i) -= tab(i, p) * tab.row(q);
    basis[q] = p;
  }

  /**
   * @brief Run the simplex pivot loop on @p tab.
   *
   * The cost row is @c tab.row(m) where @c m = tab.rows()-1.
   * The last column of @p tab is the RHS.  Only columns @c [0, ncols) are
   * eligible to enter the basis (used to exclude artificial columns in
   * Phase 2).
   *
   * @param tab    Full simplex tableau (modified in place).
   * @param basis  Current basis column indices (updated on each pivot).
   * @param ncols  Number of variable columns eligible to enter.
   * @return  0 = optimal (all reduced costs ≥ 0),
   *          1 = unbounded (no positive pivot element found),
   *          2 = #ITMAX reached.
   */
  int run(MatX &tab, VecI &basis, int ncols) {
    const int m = static_cast<int>(tab.rows()) - 1;
    const int rhs = static_cast<int>(tab.cols()) - 1;
    for (; iter < ITMAX; ++iter) {
      // Entering column: most-negative reduced cost.
      int p = -1;
      T rc = -EPS;
      for (int j = 0; j < ncols; ++j)
        if (tab(m, j) < rc) {
          rc = tab(m, j);
          p = j;
        }
      if (p < 0)
        return 0; // optimal

      // Leaving row: minimum ratio test.
      int q = -1;
      T best = std::numeric_limits<T>::max();
      for (int i = 0; i < m; ++i)
        if (tab(i, p) > EPS) {
          T ratio = tab(i, rhs) / tab(i, p);
          if (ratio < best - EPS) {
            best = ratio;
            q = i;
          }
        }
      if (q < 0)
        return 1; // unbounded

      do_pivot(tab, q, p, basis);
    }
    return 2; // max iterations exceeded
  }
};

template <typename T>
typename SimplexLP<T>::VecX SimplexLP<T>::solve(const MatX &A, const VecX &b,
                                                const VecX &c) {
  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(A.cols());

  // Find rows requiring Phase-1 artificials (initial slack would be negative).
  VecI art_rows;
  for (int i = 0; i < m; ++i)
    if (b[i] < -EPS)
      art_rows.push_back(i);
  const int na = static_cast<int>(art_rows.size());
  const int ntot = n + m + na; // original + slack + artificial columns

  // Tableau layout: [x_orig(n) | s_slack(m) | a_art(na) | RHS]
  // Rows 0..m-1: constraints.  Row m: cost row.
  MatX tab = MatX::Zero(m + 1, ntot + 1);
  tab.topLeftCorner(m, n) = A;
  tab.block(0, n, m, m).setIdentity(); // slack columns
  tab.col(ntot).head(m) = b;

  // Initial basis: slack s_i for b[i]≥0, artificial a_k for b[i]<0.
  VecI basis(m);
  std::iota(basis.begin(), basis.end(), n);

  // For negative-b rows: negate so RHS > 0, then add artificial column.
  for (int k = 0; k < na; ++k) {
    const int i = art_rows[k];
    tab.row(i).head(ntot + 1) *= T(-1); // negate entire row
    tab(i, n + m + k) = T(1);           // artificial column entry
    basis[i] = n + m + k;
  }

  iter = 0;

  // ── Phase 1: minimise Σ artificials ──────────────────────────────────────
  if (na > 0) {
    tab.row(m).setZero();
    for (int k = 0; k < na; ++k)
      tab(m, n + m + k) = T(1);
    // Express cost row in reduced form: eliminate basic (artificial) columns.
    for (int k = 0; k < na; ++k)
      tab.row(m) -= tab.row(art_rows[k]);

    if (run(tab, basis, ntot) == 2) {
      status = Status::MaxIter;
      return VecX::Zero(n);
    }
    // Phase-1 optimal value = –tab(m, ntot).  Must be ≈ 0 for feasibility.
    if (-tab(m, ntot) > EPS) {
      status = Status::Infeasible;
      return VecX::Zero(n);
    }
  }

  // ── Phase 2: minimise c^T x ───────────────────────────────────────────────
  // Rebuild cost row for the original objective, expressed in the current
  // basis.
  tab.row(m).setZero();
  tab.row(m).head(n) = c.transpose();
  for (int i = 0; i < m; ++i)
    if (basis[i] < n)
      tab.row(m) -= c[basis[i]] * tab.row(i);
  // Slack and artificial basis variables have cost 0; no elimination needed.

  const int ret = run(tab, basis, n + m); // artificials excluded from search
  if (ret == 1) {
    status = Status::Unbounded;
    return VecX::Zero(n);
  }
  if (ret == 2) {
    status = Status::MaxIter;
    return VecX::Zero(n);
  }

  // Degenerate artificials still in basis with positive value → infeasible.
  for (int i = 0; i < m; ++i)
    if (basis[i] >= n + m && tab(i, ntot) > EPS) {
      status = Status::Infeasible;
      return VecX::Zero(n);
    }

  status = Status::Optimal;
  fret = -tab(m, ntot); // tab(m, ntot) accumulates –(objective)

  VecX x = VecX::Zero(n);
  for (int i = 0; i < m; ++i)
    if (basis[i] < n)
      x[basis[i]] = tab(i, ntot);
  return x;
}

} // namespace exprmin
