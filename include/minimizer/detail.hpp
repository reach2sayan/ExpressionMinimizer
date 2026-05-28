#pragma once

#include "expressions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace exprmin::detail {

/**
 * @brief Constructs an (N+1)-vertex simplex around a centre point.
 *
 * Column 0 is @p p; column i+1 is @p p with component i displaced by
 * @p delta.  The result is an N×(N+1) matrix where each column is a vertex.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension of the space.
 * @param p      Centre vertex.
 * @param delta  Side-length of the initial simplex.
 * @return N×(N+1) matrix whose columns are the simplex vertices.
 */
template <typename T, int N>
Eigen::Matrix<T, N, N + 1> constexpr make_simplex(const Eigen::Vector<T, N> &p,
                                                  const T &delta) noexcept {
  Eigen::Matrix<T, N, N + 1> s = p.replicate(1, N + 1);
  s.diagonal(1).array() += delta;
  return s;
}



/**
 * @brief Direction state for full BFGS quasi-Newton updates.
 *
 * Maintains a dense N×N inverse-Hessian approximation @c H, updated at each
 * step via the rank-2 Broyden–Fletcher–Goldfarb–Shanno formula.  Suitable
 * for small-to-medium N where storing the full matrix is affordable.
 *
 * @tparam T  Numeric scalar type.
 * @tparam N  Dimension of the search space.
 */
template <diff::Numeric T, int N> struct BFGSDirState {
  using Point = Eigen::Vector<T, N>;
  Eigen::Matrix<T, N, N> H = Eigen::Matrix<T, N, N>::Identity();

  /// @brief Returns the BFGS search direction −H·g.
  /// @param g Current gradient.
  constexpr Point compute(const Point &g) const { return -(H * g); }

  /// @brief Resets H to the identity (used after a direction failure).
  constexpr void reset() { H.setIdentity(); }

  /**
   * @brief Applies the rank-2 BFGS update to the inverse-Hessian H.
   *
   * Skips the update when the curvature condition dg·dx ≤ 0 or when the
   * step is below machine-epsilon scale, to prevent numerical blow-up.
   *
   * @param dx  Step vector x_new − x_old.
   * @param dg  Gradient difference g_new − g_old.
   */
  constexpr void update(const Point &dx, const Point &dg) {
    const Point Hdg = H * dg;
    T fac = dg.dot(dx);
    const T fae = dg.dot(Hdg);
    constexpr T EPS = std::numeric_limits<T>::epsilon();
    if (fac > T{} && fac * fac > EPS * dg.squaredNorm() * dx.squaredNorm()) {
      fac = T{1} / fac;
      const T fad = T{1} / fae;
      const Point u = fac * dx - fad * Hdg;
      H += fac * dx * dx.transpose();
      H -= fad * Hdg * Hdg.transpose();
      H += fae * u * u.transpose();
    }
  }
};

/**
 * @brief Direction state for limited-memory BFGS (L-BFGS).
 *
 * Stores the M most-recent (s, y) curvature pairs in a circular buffer and
 * applies the Nocedal two-loop recursion to compute the search direction in
 * O(M·N) time and O(M·N) space instead of the O(N²) required by full BFGS.
 *
 * @tparam T  Numeric scalar type.
 * @tparam N  Dimension of the search space.
 * @tparam M  History size (number of stored curvature pairs).
 */
template <diff::Numeric T, int N, int M> struct LBFGSDirState {
  using Point = Eigen::Vector<T, N>;
  std::array<Point, M> s_buf, y_buf;
  std::array<T, M> rho_buf{};
  int buf_size = 0, buf_head = 0;

  /**
   * @brief Computes the L-BFGS search direction via the two-loop recursion.
   *
   * Returns −H_k·g where H_k is the implicit inverse-Hessian built from the
   * stored (s, y) pairs.  Falls back to −g on the first iteration when the
   * buffer is empty.
   *
   * @param g  Current gradient.
   * @return   Search direction vector.
   */
  constexpr Point compute(const Point &g) const {
    Point q = g;
    std::array<T, M> al{};
    for (int j = 0; j < buf_size; ++j) {
      const int idx = (buf_head - 1 - j + M) % M;
      al[j] = rho_buf[idx] * s_buf[idx].dot(q);
      q -= al[j] * y_buf[idx];
    }
    Point r;
    if (buf_size > 0) {
      const int last = (buf_head - 1 + M) % M;
      r = (s_buf[last].dot(y_buf[last]) / y_buf[last].squaredNorm()) * q;
    } else {
      r = q;
    }
    for (int j = buf_size - 1; j >= 0; --j) {
      const int idx = (buf_head - 1 - j + M) % M;
      r += s_buf[idx] * (al[j] - rho_buf[idx] * y_buf[idx].dot(r));
    }
    return -r;
  }

  /**
   * @brief Stores a new (s, y) pair, evicting the oldest when the buffer is full.
   *
   * Skips pairs that violate the curvature condition y·s > ε·‖y‖² to keep
   * the implicit inverse-Hessian positive-definite.
   *
   * @param dx  Step vector s = x_new − x_old.
   * @param dg  Gradient difference y = g_new − g_old.
   */
  constexpr void update(const Point &dx, const Point &dg) {
    constexpr T EPS = std::numeric_limits<T>::epsilon();
    const T ys = dg.dot(dx);
    if (ys > EPS * dg.squaredNorm()) {
      s_buf[buf_head] = dx;
      y_buf[buf_head] = dg;
      rho_buf[buf_head] = T{1} / ys;
      buf_head = (buf_head + 1) % M;
      if (buf_size < M) {
        ++buf_size;
      }
    }
  }

  /// @brief Clears the curvature-pair buffer (called after a direction failure).
  constexpr void reset() {
    buf_size = 0;
    buf_head = 0;
  }
};

/**
 * @brief Unified quasi-Newton minimization loop.
 *
 * Iterates until the scaled gradient infinity-norm drops below @p gtol or
 * @p itmax steps are exhausted.  If the direction supplied by @p ds is not a
 * descent direction (slope ≥ 0), the state is reset and a steepest-descent
 * step is used instead.
 *
 * @tparam T            Numeric scalar type.
 * @tparam N            Dimension of the search space.
 * @tparam EvalGrad     Callable returning std::pair<T, Point> (value, gradient).
 * @tparam LineSearchFn Callable with signature Point(xc, xi, fp, slope),
 *                      returning the step vector dx = α·xi.
 * @tparam DirState     Type satisfying the DirState concept:
 *                      compute(g)→xi, update(dx, dg), reset().
 * @param eg        Evaluation functor (value + gradient).
 * @param x         Initial point.
 * @param gtol      Convergence tolerance on the scaled gradient.
 * @param itmax     Maximum number of iterations.
 * @param ls_fn     Line-search functor.
 * @param ds        Direction state (BFGSDirState or LBFGSDirState).
 * @param iter_out  Set to the number of iterations performed on return.
 * @return Approximate minimizer.
 */
template <diff::Numeric T, int N, typename EvalGrad, typename LineSearchFn,
          typename DirState>
constexpr Eigen::Vector<T, N>
quasi_newton_impl(EvalGrad &eg, Eigen::Vector<T, N> x, T gtol, int itmax,
                  LineSearchFn ls_fn, DirState &ds, int &iter_out) {
  using std::abs, std::max;
  using Point = Eigen::Vector<T, N>;

  auto [fp, g] = eg(x);

  for (iter_out = 0; iter_out < itmax; ++iter_out) {
    const T den = max(abs(fp), T{1});
    const T scaled_grad_inf_norm =
        (g.cwiseAbs().array() * x.cwiseAbs().cwiseMax(T{1}).array()).maxCoeff();
    if (scaled_grad_inf_norm / den < gtol) {
      break;
    }

    Point xi = ds.compute(g);
    const T slope = g.dot(xi);
    if (slope >= T{}) {
      ds.reset();
      xi = -g;
    }

    const Point dx = ls_fn(x, xi, fp, g.dot(xi));
    x += dx;

    auto [fn, g_new] = eg(x);
    fp = fn;
    ds.update(dx, g_new - g);
    g = std::move(g_new);
  }
  return x;
}

/**
 * @brief BFGS minimization with a backtracking Armijo line search.
 *
 * Used for objectives that are not @c CExpression instances (e.g. the
 * augmented-Lagrangian merit function in AugLag), which cannot own a typed
 * @c Brent<Expr>.  Halves α up to 40 times until the Armijo sufficient-
 * decrease condition f(x + α·xi) ≤ f(x) + c₁·α·∇f·xi is met.
 *
 * @tparam T          Numeric scalar type.
 * @tparam N          Dimension of the search space.
 * @tparam EvalGrad   Callable returning std::pair<T, Point> (value, gradient).
 * @param eval_grad   Evaluation functor.
 * @param x           Initial point.
 * @param ftol        Gradient convergence tolerance.
 * @param itmax       Maximum number of BFGS iterations.
 * @return Approximate minimizer.
 */
template <diff::Numeric T, int N, typename EvalGrad>
constexpr Eigen::Vector<T, N>
bfgs_armijo(EvalGrad eval_grad, Eigen::Vector<T, N> x, T ftol, int itmax) {
  using Point = Eigen::Vector<T, N>;
  constexpr T EPS = std::numeric_limits<T>::epsilon();
  constexpr T C1{1e-4};
  auto line_search_fn = [&](const Point &xc, const Point &xi, T fp, T slope) {
    T alpha{1};
    for (int k = 0; k < 40 && alpha > EPS; ++k, alpha *= T{0.5}) {
      if (eval_grad(xc + alpha * xi).first <= fp + C1 * alpha * slope) {
        break;
      }
    }
    return alpha * xi;
  };
  BFGSDirState<T, N> ds;
  int dummy = 0;
  return quasi_newton_impl(eval_grad, std::move(x), ftol, itmax,
                           std::move(line_search_fn), ds, dummy);
}

/**
 * @brief Nelder–Mead trial reflection/expansion/contraction step.
 *
 * Reflects vertex @p ihi through the centroid of the remaining vertices,
 * scaled by @p fac.  Updates the simplex @p s, function-value vector @p y,
 * and the column-sum cache @p psum in place when the trial point improves.
 *
 * @tparam T    Scalar type.
 * @tparam N    Simplex dimension.
 * @param ptr   Object exposing eval_at(Point) → T.
 * @param s     N×(N+1) simplex matrix (columns are vertices).
 * @param y     Function values at each vertex.
 * @param psum  Sum of all vertex columns (cached centroid numerator).
 * @param ihi   Index of the highest (worst) vertex to reflect.
 * @param fac   Reflection factor: −1 = reflect, >1 = expand, 0<fac<1 = contract.
 * @return Function value at the trial point.
 */
template <typename T, std::size_t N>
constexpr T
amotry_impl(auto &&ptr,
            Eigen::Matrix<T, static_cast<int>(N), static_cast<int>(N + 1)> &s,
            Eigen::Vector<T, static_cast<int>(N + 1)> &y,
            Eigen::Vector<T, static_cast<int>(N)> &psum, const std::size_t ihi,
            const T &fac) {
  const T fac1 = (T{1} - fac) / static_cast<T>(N);
  const T fac2 = fac1 - fac;
  const Eigen::Vector<T, static_cast<int>(N)> ptry =
      fac1 * psum - fac2 * s.col(ihi);
  const T ytry = ptr.eval_at(ptry);
  if (ytry < y[ihi]) {
    psum += ptry - s.col(ihi);
    s.col(ihi) = ptry;
    y[ihi] = ytry;
  }
  return ytry;
};

} // namespace exprmin::detail
