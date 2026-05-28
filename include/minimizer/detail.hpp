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
 * @brief NR §10.1 — Downhill bracket search for a 1-D minimum.
 *
 * Expands [ax, bx] until a bracket (ax, bx, cx) satisfying f(bx) < f(ax)
 * and f(bx) < f(cx) is found.  Uses golden-ratio and parabolic extrapolation
 * steps, capped at GLIMIT×(cx−bx) to prevent runaway.
 *
 * @pre  @p fa = f(ax) and @p fb = f(bx) must be initialised by the caller.
 * @post ax < bx < cx and f(bx) ≤ min(f(ax), f(cx)).
 *
 * @tparam T  Numeric scalar type.
 * @tparam F  Callable with signature T(T).
 * @param f   Objective function.
 * @param ax  Left endpoint (in/out).
 * @param bx  Middle point (in/out).
 * @param cx  Right endpoint (out).
 * @param fa  f(ax) (in/out).
 * @param fb  f(bx) (in/out).
 * @param fc  f(cx) (out).
 */
template <diff::Numeric T, std::invocable<T> F>
constexpr void bracket(F &f, T &ax, T &bx, T &cx, T &fa, T &fb, T &fc) {
  using std::abs;
  using std::max;
  using std::swap;
  constexpr T GOLD = static_cast<T>(std::numbers::phi_v<double>);
  constexpr T GLIMIT{100};
  constexpr T TINY{1.0e-20};

  auto reset = [&f, &GOLD](double &u, double &fu, const double cx,
                           const double bx) {
    u = cx + GOLD * (cx - bx);
    fu = f(u);
  };

  if (fb > fa) {
    swap(ax, bx);
    swap(fa, fb);
  } // swap so A -> B is downhill
  cx = bx + GOLD * (bx - ax); // first guess
  fc = f(cx);

  while (fb > fc) {
    const T r = (bx - ax) * (fb - fc);
    const T q = (bx - cx) * (fb - fa);
    const T qdiff = q - r;
    const T denom =
        T{2} * (qdiff >= T{} ? T{1} : T{-1}) * max(abs(qdiff), TINY);
    T u = bx - ((bx - cx) * q - (bx - ax) * r) / denom;
    T ulim = bx + GLIMIT * (cx - bx);
    T fu = f(u);

    if ((bx - u) * (u - cx) > T{}) { // parabolic u in [b,c]
      fu = f(u);
      if (fu < fc) { // minimum between [b,c]
        ax = bx;
        fa = fb;
        bx = u;
        fb = fu;
        return;
      }
      if (fu > fb) { // minimum between [a,u]
        cx = u;
        fc = fu;
        return;
      }
      reset(u, fu, cx, bx);
    } else if ((cx - u) * (u - ulim) > T{}) { // parabolic between c and limit
      fu = f(u);
      if (fu < fc) {
        bx = cx;
        fb = fc;
        cx = u;
        fc = fu;
        reset(u, fu, cx, bx);
      }
    } else if ((u - ulim) * (ulim - cx) >= T{}) { // parabolic between u and lim
      u = ulim;
      fu = f(u);
    } else { // parabolic step overshot ax; take default GOLD step past cx
      reset(u, fu, cx, bx);
    }
    // eliminate oldest point and continue
    ax = bx;
    bx = cx;
    cx = u;
    fa = fb;
    fb = fc;
    fc = fu;
  }
}

/**
 * @brief NR §10.3 — Brent's method for derivative-free 1-D minimization.
 *
 * Combines golden-section search with parabolic interpolation inside a
 * pre-computed bracket [ax, cx] with interior point bx.  Converges
 * super-linearly on smooth unimodal functions.
 *
 * @tparam T      Numeric scalar type.
 * @tparam F      Callable with signature T(T).
 * @param f       Objective function.
 * @param ax      Left bracket endpoint.
 * @param bx      Interior point (initial best-guess abscissa).
 * @param cx      Right bracket endpoint.
 * @param tol     Fractional tolerance on the abscissa.
 * @param zeps    Small additive offset preventing tol from collapsing to zero.
 * @param itmax   Maximum number of iterations (default 100).
 * @return Abscissa of the minimum.
 */
template <diff::Numeric T, std::invocable<T> F>
constexpr T brent(F &f, const T &ax, const T &bx, const T &cx, const T &tol,
                  const T &zeps, const int itmax = 100) {
  using std::abs;
  using std::max;
  using std::min;
  constexpr T CGOLD = static_cast<T>(1.0 - 1.0 / std::numbers::phi_v<double>);

  // Three points x (best), w (2nd), v (3rd) are kept ranked as fx≤fw≤fv.
  // The parabolic interpolation below fits a curve through all three, so the
  // housekeeping at the end of each iteration must update them every step.
  T a = min(ax, cx); // sort so a ≤ b regardless of the input order
  T b = max(ax, cx);
  T x = bx, w = bx, v = bx;
  T fx = f(x), fw = fx, fv = fx;
  T d{}, e{};

  auto golden_section_step = [&CGOLD](auto &e, auto &d, const auto &a,
                                      const auto &b, const auto &x,
                                      const auto &xm) {
    e = (x >= xm ? a - x : b - x);
    d = CGOLD * e;
  };
  for (int i = 0; i < itmax; ++i) {
    const T xm = T{0.5} * (a + b);
    const T tol1 = tol * abs(x) + zeps;
    const T tol2 = T{2} * tol1;

    if (abs(x - xm) <= tol2 - T{0.5} * (b - a)) {
      return x;
    }

    T u;
    if (abs(e) > tol1) {
      // attempt parabolic interpolation through (x, w, v)
      T r = (x - w) * (fx - fv);
      T q = (x - v) * (fx - fw);
      T p = (x - v) * q - (x - w) * r;
      q = T{2} * (q - r);
      if (q > T{}) {
        p = -p;
      }
      q = abs(q);
      const T etemp = e;
      e = d;
      if (abs(p) >= abs(T{0.5} * q * etemp) || p <= q * (a - x) ||
          p >= q * (b - x)) {
        // parabolic step is too large or out of bracket; fall back to golden section
        golden_section_step(e,d,a,b,x,xm);
      } else {
        // parabolic step accepted: inside bracket and at most half of last step
        d = p / q;
        u = x + d;
        if (u - a < tol2 || b - u < tol2) {
          d = (xm >= x ? tol1 : -tol1);
        }
      }
    } else {
      golden_section_step(e,d,a,b,x,xm);
    }

    // clamp step to at least tol1 so u is never evaluated right on top of x
    u = (abs(d) >= tol1 ? x + d : x + (d >= T{} ? tol1 : -tol1));
    const T fu = f(u);

    // --- housekeeping: tighten the bracket and maintain the ranked trio ---
    // Part 1 — tighten [a, b]: u proves the minimum cannot be on one side of
    // x, so we move whichever bound is on that side inward.
    //   fu ≤ fx  →  x is now interior; the side opposite u can be discarded.
    //   fu > fx  →  u itself becomes the new bound on its side.
    //
    // Part 2 — slot u into the ranking so next iteration has fresh curvature:
    //   fu ≤ fx  →  u beats everything; demote the chain v ← w ← x ← u.
    //   fu ≤ fw  →  u is second-best; shift old w down to v, u takes w.
    //   fu ≤ fv  →  u is third-best; replace v.
    //   fu > fv  →  u is worse than all three; discard it (no update needed).
    if (fu <= fx) {
      if (u < x) {
        b = x;
      } else {
        a = x;
      }
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    } else {
      if (u < x) {
        a = u;
      } else {
        b = u;
      }
      if (fu <= fw || w == x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }
  return x;
}

/**
 * @brief NR §10.4 — Brent's method augmented with derivative information.
 *
 * Uses secant interpolation on f′ to propose steps; falls back to bisection
 * toward the sign-change side when the secant step leaves the bracket or
 * fails to shrink the interval by at least half.  Typically converges in
 * fewer evaluations than plain Brent on smooth functions.
 *
 * @pre  @p F must expose both @c operator()(T) (function value) and
 *       @c df(T) (first derivative).
 *
 * @tparam T      Numeric scalar type.
 * @tparam F      Functor exposing operator()(T) and df(T).
 * @param f       Objective functor.
 * @param ax      Left bracket endpoint.
 * @param bx      Interior point (initial best-guess abscissa).
 * @param cx      Right bracket endpoint.
 * @param tol     Fractional tolerance on the abscissa.
 * @param zeps    Small additive offset preventing tol from collapsing to zero.
 * @param itmax   Maximum number of iterations (default 100).
 * @return Abscissa of the minimum.
 */
template <diff::Numeric T, std::invocable<T> F>
constexpr T dbrent(F &f, const T &ax, const T &bx, const T &cx, const T &tol,
                   const T &zeps, int itmax = 100) {
  using std::abs;
  using std::max;
  using std::min;

  // Same three-point ranked trio as brent (x best, w 2nd, v 3rd, fx≤fw≤fv),
  // but each point also carries its derivative (dx, dw, dv) so the secant
  // step can be computed without extra f' calls during housekeeping.
  T a = min(ax, cx), b = max(ax, cx);
  T x = bx, w = bx, v = bx;
  T fx = f(bx), fw = fx, fv = fx;
  T dx = f.df(bx), dw = dx, dv = dx;
  T d{}, e{};

  for (int i = 0; i < itmax; ++i) {
    const T xm = T{0.5} * (a + b);
    const T tol1 = tol * abs(x) + zeps;
    const T tol2 = T{2} * tol1;
    if (abs(x - xm) <= tol2 - T{0.5} * (b - a)) {
      return x;
    }

    // Bisect toward the side where f' points downhill; used whenever the
    // secant step is unavailable or unreliable (same role as golden section
    // in plain Brent, but guided by the sign of the derivative).
    auto bisect = [](const T dx, const T a, const T b, const T x) {
      return dx >= T{} ? a - x : b - x;
    };

    if (abs(e) > tol1) {
      // Attempt secant steps using the (x,w) and (x,v) derivative pairs.
      // Initialise d1/d2 to a value outside [a,b] so ok1/ok2 stays false
      // for whichever pair has equal derivatives (no valid secant step).
      T d1 = T{2} * (b - a), d2 = d1;
      if (dw != dx) {
        d1 = (w - x) * dx / (dx - dw); // secant step through (x, w)
      }
      if (dv != dx) {
        d2 = (v - x) * dx / (dx - dv); // secant step through (x, v)
      }
      const T u1 = x + d1, u2 = x + d2;
      // a step is acceptable only if it lands inside [a,b] and moves
      // in the downhill direction (dx·d ≤ 0)
      const bool ok1 = (a - u1) * (u1 - b) > T{} && dx * d1 <= T{};
      const bool ok2 = (a - u2) * (u2 - b) > T{} && dx * d2 <= T{};
      const T olde = e;
      e = d;
      if (ok1 || ok2) {
        // prefer the shorter of the two valid steps
        d = (ok1 && ok2) ? (abs(d1) < abs(d2) ? d1 : d2) : (ok1 ? d1 : d2);
        if (abs(d) <= abs(T{0.5} * olde)) {
          const T u = x + d;
          if (u - a < tol2 || b - u < tol2)
            d = (xm >= x ? tol1 : -tol1);
        } else {
          e = bisect(dx, a, b, x);
          d = T{0.5} * e;
        }
      } else {
        e = bisect(dx, a, b, x);
        d = T{0.5} * e;
      }
    } else {
      e = bisect(dx, a, b, x);
      d = T{0.5} * e;
    }

    const T u = (abs(d) >= tol1 ? x + d : x + (d >= T{} ? tol1 : -tol1));
    const T fu = f(u);
    const T du = f.df(u);

    // --- housekeeping: identical bracket/ranking logic to brent, but each
    // slot in the trio also carries a derivative that must move with it.
    // The derivative at the demoted point is preserved because the secant
    // step next iteration needs dx at w and v, not just their positions.
    if (fu <= fx) {
      (u < x ? b : a) = x;
      v = w;
      fv = fw;
      dv = dw;
      w = x;
      fw = fx;
      dw = dx;
      x = u;
      fx = fu;
      dx = du;
    } else {
      (u < x ? a : b) = u;
      if (fu <= fw || w == x) {
        v = w;
        fv = fw;
        dv = dw;
        w = u;
        fw = fu;
        dw = du;
      } else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
        dv = du;
      }
    }
  }
  return x;
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
