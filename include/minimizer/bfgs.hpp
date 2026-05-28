#pragma once

#include "brent.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <type_traits>
#include <utility>

namespace exprmin {

namespace mp = boost::mp11;

/// Selects the inverse-Hessian update formula used by QuasiNewton /
/// bfgs_armijo.
enum class QNUpdate {
  BFGS, ///< Broyden-Fletcher-Goldfarb-Shanno: rank-2, SPD when sTy > 0
  DFP,  ///< Davidon-Fletcher-Powell: rank-2, dual of BFGS
  SR1,  ///< Symmetric Rank-1: rank-1, no positive-definiteness guarantee
};

/**
 * @brief Armijo backtracking sufficient-decrease line search.
 *
 * Owns the expression for eval_at.  minimize_fn accepts any 1-D callable
 * together with the current function value @p fp and the directional slope,
 * and returns the accepted step-length t satisfying the Armijo condition
 * @f$ f(t) \le f(0) + C_1 \alpha \nabla f \cdot d @f$ (c₁ = 10⁻⁴).
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 */
template <diff::CExpression Expr> struct Armijo {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

  Expr expr;
  value_type fret{};
  const value_type tol;

  /**
   * @brief Constructs an Armijo line search wrapping the given expression.
   * @param e     Expression to evaluate.
   * @param tol_  Tolerance passed through (unused internally; kept for API
   *              symmetry with Brent/Dbrent).
   */
  constexpr explicit Armijo(Expr e,
                            value_type tol_ = static_cast<value_type>(1e-8))
      : expr{std::move(e)}, tol{tol_} {}

  /**
   * @brief Evaluates the expression at point @p p.
   * @param p  N-dimensional input point.
   * @return   f(p).
   */
  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  /**
   * @brief Backtracking Armijo line search along a 1-D slice.
   *
   * Halves @c alpha up to 40 times until @f$ f(\alpha) \le f_p + C_1 \alpha
   * \cdot \text{slope} @f$ (Armijo / sufficient-decrease condition).
   *
   * @param f1d    1-D callable @c T(T) evaluating @c f(t).
   * @param fp     @c f(0) — function value at the start of the step.
   * @param slope  Directional derivative @c ∇f·d at @c t=0.
   * @return       Accepted step-length α.
   */
  template <std::invocable<value_type> F>
  constexpr value_type minimize_fn(F f1d, value_type fp,
                                   value_type slope) const {
    constexpr value_type EPS = std::numeric_limits<value_type>::epsilon();
    constexpr value_type C1{1e-4};
    value_type alpha{1};
    for (int k = 0; k < 40 && alpha > EPS; ++k, alpha *= value_type{0.5})
      if (f1d(alpha) <= fp + C1 * alpha * slope) {
        break;
      }
    return alpha;
  }
};
template <diff::CExpression Expr> Armijo(Expr) -> Armijo<Expr>;
template <diff::CExpression Expr, typename T> Armijo(Expr, T) -> Armijo<Expr>;

/**
 * @brief Reusable direction state for quasi-Newton update formulas.
 *
 * Maintains a dense N×N inverse-Hessian approximation H (initialised to I).
 * All three formulas share compute() and reset(); only update() differs,
 * dispatched at compile time via @c if constexpr with zero runtime overhead.
 *
 * Used as the @c DirState inside QuasiNewton<…> and directly in bfgs_armijo.
 *
 * @tparam T      Numeric scalar type.
 * @tparam N      Dimension of the search space.
 * @tparam Update Inverse-Hessian update formula (BFGS, DFP, or SR1).
 */
template <diff::Numeric T, int N, QNUpdate Update> struct QNDirState {
  using Point = Eigen::Vector<T, N>;
  Eigen::Matrix<T, N, N> H = Eigen::Matrix<T, N, N>::Identity();

  /// @brief Returns the quasi-Newton search direction −H·g.
  constexpr Point compute(const Point &g) const { return -(H * g); }

  /// @brief Resets H to the identity (used after a non-descent-direction failure).
  constexpr void reset() { H.setIdentity(); }

  /**
   * @brief Updates the inverse-Hessian approximation given a step (dx, dg).
   *
   * BFGS — rank-2, preserves SPD when sTy > 0.
   *         Skips when sTy ≤ 0 or the step is below machine-epsilon scale.
   * DFP  — rank-2 dual of BFGS: H ← H + ss^T/(s·y) − Hyy^TH/(y^THy).
   *         Skips when sTy ≤ 0 or yTHy ≤ EPS·‖y‖².
   * SR1  — rank-1 symmetric: H ← H + (s−Hy)(s−Hy)^T / (s−Hy)·y.
   *         Nocedal §6.2 safeguard: skips when |(s−Hy)·y| ≤ EPS·‖s−Hy‖·‖y‖.
   *         Does NOT guarantee positive definiteness.
   *
   * @param dx  Step vector s = x_new − x_old.
   * @param dg  Gradient difference y = g_new − g_old.
   */
  constexpr void update(const Point &dx, const Point &dg) {
    using std::abs;
    constexpr T EPS = std::numeric_limits<T>::epsilon();

    if constexpr (Update == QNUpdate::BFGS) {
      const Point Hdg = H * dg;
      T fac = dg.dot(dx);        // sTy
      const T fae = dg.dot(Hdg); // yTHy
      if (fac > T{} && fac * fac > EPS * dg.squaredNorm() * dx.squaredNorm()) {
        fac = T{1} / fac;
        const T fad = T{1} / fae;
        const Point u = fac * dx - fad * Hdg;
        H += fac * dx * dx.transpose();
        H -= fad * Hdg * Hdg.transpose();
        H += fae * u * u.transpose();
      }
    } else if constexpr (Update == QNUpdate::DFP) {
      const Point Hy = H * dg;
      const T sTy = dx.dot(dg);
      const T yTHy = dg.dot(Hy);
      if (sTy > T{} && yTHy > EPS * dg.squaredNorm()) {
        H += dx * dx.transpose() / sTy;
        H -= Hy * Hy.transpose() / yTHy;
      }
    } else {                       // SR1
      const Point v = dx - H * dg; // s − Hy
      const T vTy = v.dot(dg);
      if (abs(vTy) > EPS * v.norm() * dg.norm()) {
        H += v * v.transpose() / vTy;
      }
    }
  }
};

/**
 * @brief CRTP base shared by QuasiNewton and LBFGS.
 *
 * Owns the 1-D line-search object @c ls, gradient evaluation (via reverse-mode
 * AD), and the unified quasi_newton_impl loop.  Derived classes supply their
 * own @c DirState and call quasi_newton_impl with it.
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 * @tparam LS1D  Line-search policy (Brent, Dbrent, or Armijo).
 */
template <diff::CExpression Expr, template <diff::CExpression> class LS1D>
struct QuasiNewtonBase {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;

protected:
  LS1D<Expr> ls;
  value_type fret{};
  int iter{};
  const value_type gtol;

public:
  /**
   * @brief Constructs the base wrapping the given expression.
   * @param e      Expression to minimize.
   * @param gtol_  Scaled-gradient convergence tolerance; passed to ls.
   */
  constexpr explicit QuasiNewtonBase(Expr e, value_type gtol_)
      : ls(std::move(e), gtol_), gtol(gtol_) {}

  /// @brief Evaluates the expression at point @p p.
  constexpr value_type eval_at(const Point &p) { return ls.eval_at(p); }
  /// @brief Callable interface delegating to eval_at.
  constexpr value_type operator()(const Point &p) { return eval_at(p); }
  /// @brief Returns f at the best point after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fret; }
  /**
   * @brief Evaluates the expression and its reverse-mode gradient at @p p.
   * @param p  N-dimensional input point.
   * @return   {f(p), ∇f(p)} as an std::pair.
   */
  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    ls.expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(ls.expr);
    return {ls.expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  /**
   * @brief Returns a line-search callable @c (xc, xi, fp, slope) → dx.
   *
   * Dispatches to the owned @c ls.minimize_fn.  Armijo uses @c fp and
   * @c slope for the sufficient-decrease check; Brent/Dbrent bracket on
   * [0, 1] and ignore those arguments.
   */
  constexpr auto make_line_search_fn();

protected:
  /**
   * @brief Unified quasi-Newton minimization loop.
   *
   * Iterates until the scaled gradient infinity-norm drops below @p conv_tol
   * or @p itmax steps are exhausted.  If the direction from @p ds is not a
   * descent direction (slope ≥ 0 — can happen when the Hessian approximation
   * has become ill-conditioned), the state is reset and steepest descent is
   * used for that step.
   *
   * @tparam EvalGrad     Callable returning std::pair<value_type, Point>.
   * @tparam LineSearchFn Callable (xc, xi, fp, slope) → Point (step dx = α·xi).
   * @tparam DirState     Satisfies: compute(g)→xi, update(dx,dg), reset().
   * @param eg        Evaluation functor (value + gradient).
   * @param x         Initial point.
   * @param conv_tol  Convergence tolerance on the scaled gradient.
   * @param itmax     Maximum iterations.
   * @param ls_fn     Line-search functor.
   * @param ds        Direction state (QNDirState).
   * @param iter_out  Set to the iteration count on return.
   * @return Approximate minimizer.
   */
  template <typename EvalGrad, typename LineSearchFn, typename DirState>
  constexpr Point quasi_newton_impl(EvalGrad &eg, Point x, value_type conv_tol,
                                    int itmax, LineSearchFn ls_fn, DirState &ds,
                                    int &iter_out) {
    using std::abs, std::max;

    auto [fp, g] = eg(x); // evaluate f and ∇f at the starting point

    for (iter_out = 0; iter_out < itmax; ++iter_out) {
      // Scaled gradient convergence check.  Each component of ∇f is weighted
      // by max(|xᵢ|, 1) so the tolerance is relative to the current scale of
      // x rather than absolute — avoids false convergence near the origin.
      const value_type den = max(abs(fp), value_type{1});
      const value_type scaled_grad_inf_norm =
          (g.cwiseAbs().array() * x.cwiseAbs().cwiseMax(value_type{1}).array())
              .maxCoeff();
      if (scaled_grad_inf_norm / den < conv_tol) {
        break;
      }

      // Ask the direction state for a quasi-Newton step.  If it is not a
      // descent direction (g·xi ≥ 0), reset the Hessian approximation and
      // fall back to steepest descent for this step only.
      Point xi = ds.compute(g);
      const value_type slope = g.dot(xi);
      if (slope >= value_type{}) {
        ds.reset();
        xi = -g;
      }

      // Line search along xi; returns the accepted step vector dx = α·xi
      const Point dx = ls_fn(x, xi, fp, g.dot(xi));
      x += dx;

      // Re-evaluate at the new point, feed (s, y) = (dx, Δg) to the
      // direction state so it can update its Hessian approximation.
      auto [fn, g_new] = eg(x);
      fp = fn;
      ds.update(dx, g_new - g);
      g = std::move(g_new);
    }
    return x;
  }
};

/**
 * @brief NR §10.7 — quasi-Newton minimizer with a full N×N inverse-Hessian.
 *
 * The update formula is selected at compile time via @p Update; all three
 * variants share the same minimization loop and line-search infrastructure
 * (QuasiNewtonBase), and differ only in QNDirState::update.
 *
 * @tparam Expr   A type satisfying the diff::CExpression concept.
 * @tparam LS1D   Line-search policy: Brent (default), Dbrent, or Armijo.
 * @tparam Update Inverse-Hessian update: QNUpdate::BFGS (default), DFP, SR1.
 */
template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = Brent,
          QNUpdate Update = QNUpdate::BFGS>
struct QuasiNewton : QuasiNewtonBase<Expr, LS1D> {
  using Base = QuasiNewtonBase<Expr, LS1D>;
  using Base::eval_at;
  using Base::fret;
  using Base::gtol;
  using typename Base::Point;
  using typename Base::value_type;
  using Base::make_line_search_fn;
  using Base::quasi_newton_impl;

  static constexpr int ITMAX = 200;

  /**
   * @brief Constructs a QuasiNewton minimizer wrapping the given expression.
   * @param e      Expression to minimize.
   * @param gtol_  Scaled-gradient convergence tolerance (default 10⁻⁸).
   */
  constexpr explicit QuasiNewton(
      Expr e, value_type gtol_ = static_cast<value_type>(1e-8))
      : Base(std::move(e), gtol_) {}

  /**
   * @brief Minimizes from initial point @p p.
   * @param p  Initial point.
   * @return Approximate minimizer.
   */
  constexpr Point minimize(Point p) {
    const auto eg = [this](const Point &q) { return this->eval_grad(q); };
    auto ls_fn = make_line_search_fn();
    DirState ds;
    p = quasi_newton_impl(eg, std::move(p), gtol, ITMAX, ls_fn, ds,
                                this->iter);
    fret = eval_at(p);
    return p;
  }

private:
  using DirState = QNDirState<value_type, static_cast<int>(Base::N), Update>;
};

template <diff::CExpression Expr, template <diff::CExpression> class LS1D>
constexpr auto QuasiNewtonBase<Expr, LS1D>::make_line_search_fn() {
  return [this](const Point &xc, const Point &xi, value_type fp,
                value_type slope) -> Point {
    auto f1d = [this, &xc, &xi](value_type t) {
      ls.expr.update(Syms{}, xc + t * xi);
      return ls.expr.eval();
    };
    value_type t_min;
    if constexpr (std::is_same_v<LS1D<Expr>, Armijo<Expr>>) {
      t_min = ls.minimize_fn(f1d, fp, slope);
    } else if constexpr (std::is_same_v<LS1D<Expr>, Dbrent<Expr>>) {
      struct FC {
        decltype(f1d) &f_;
        const Point &xi_;
        LS1D<Expr> &ls_;
        const Point &xc_;
        value_type operator()(value_type t) { return f_(t); }
        value_type df(value_type t) {
          ls_.expr.update(Syms{}, xc_ + t * xi_);
          const auto g = diff::gradient<diff::DiffMode::Reverse>(ls_.expr);
          return Eigen::Map<const Point>(g.data()).dot(xi_);
        }
      } fc{f1d, xi, ls, xc};
      t_min = ls.minimize_fn(fc, value_type{0}, value_type{1});
    } else {
      // Brent (default)
      t_min = ls.minimize_fn(f1d, value_type{0}, value_type{1});
    }
    return t_min * xi;
  };
}

// Deduction guides — C++20 CTAD propagates these to the aliases below.
template <diff::CExpression Expr> QuasiNewton(Expr) -> QuasiNewton<Expr>;
template <diff::CExpression Expr, typename T>
QuasiNewton(Expr, T) -> QuasiNewton<Expr>;

// ── Named update aliases ──────────────────────────────────────────────────────

/// @brief Quasi-Newton minimizer with the BFGS inverse-Hessian update (rank-2).
/// This is the default and is identical in behaviour to the former @c BFGS
/// class. Positive-definiteness of H is maintained whenever s·y > 0, so it is
/// the most robust choice for general smooth objectives.
template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = Brent>
using BFGS = QuasiNewton<Expr, LS1D, QNUpdate::BFGS>;

/// @brief Quasi-Newton minimizer with the DFP inverse-Hessian update (rank-2).
/// The Davidon-Fletcher-Powell formula is the algebraic dual of BFGS — it
/// applies the same two rank-1 corrections but in transposed order. DFP also
/// preserves positive-definiteness when s·y > 0, but is empirically less
/// robust than BFGS on ill-conditioned problems (Nocedal §6.1).
template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = Brent>
using DFP = QuasiNewton<Expr, LS1D, QNUpdate::DFP>;

/// @brief Quasi-Newton minimizer with the SR1 inverse-Hessian update (rank-1).
/// The Symmetric Rank-1 formula can build a more accurate Hessian approximation
/// than BFGS in the spectral sense, but does @b not guarantee positive-
/// definiteness (Nocedal §6.2). The Nocedal safeguard skips the update when
/// |(s−Hy)·y| ≤ ε·‖s−Hy‖·‖y‖, and the loop falls back to steepest descent
/// whenever the resulting search direction is not a descent direction.
template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = Brent>
using SR1 = QuasiNewton<Expr, LS1D, QNUpdate::SR1>;

// ── Convenience LS × update aliases ──────────────────────────────────────────
template <diff::CExpression Expr> using DBFGS = BFGS<Expr, Dbrent>; ///< BFGS + derivative-based Brent line search
template <diff::CExpression Expr> using ABFGS = BFGS<Expr, Armijo>; ///< BFGS + Armijo backtracking
template <diff::CExpression Expr> using DSR1  = SR1<Expr, Dbrent>;  ///< SR1  + derivative-based Brent line search
template <diff::CExpression Expr> using DDFP  = DFP<Expr, Dbrent>;  ///< DFP  + derivative-based Brent line search

/**
 * @brief Quasi-Newton minimization with Armijo backtracking for non-expression
 *        objectives (e.g. the augmented-Lagrangian merit function in AugLag).
 *
 * The update formula is selected via @p Update (default: QNUpdate::BFGS).
 * Existing call sites that specify only @c T and @c N are unaffected.
 *
 * @tparam T      Numeric scalar type.
 * @tparam N      Dimension of the search space.
 * @tparam Update Inverse-Hessian update formula (default: QNUpdate::BFGS).
 * @param eval_grad   Callable returning std::pair<T, Point> (value, gradient).
 * @param x           Initial point.
 * @param ftol        Scaled-gradient convergence tolerance.
 * @param itmax       Maximum number of iterations.
 * @return Approximate minimizer.
 */
template <diff::Numeric T, int N, typename EvalGrad,
          QNUpdate Update = QNUpdate::BFGS>
constexpr Eigen::Vector<T, N>
bfgs_armijo(EvalGrad eval_grad, Eigen::Vector<T, N> x, T ftol, int itmax) {
  using std::abs, std::max;
  using Point = Eigen::Vector<T, N>;
  constexpr T EPS = std::numeric_limits<T>::epsilon();
  constexpr T C1{1e-4};

  auto line_search_fn = [&](const Point &xc, const Point &xi, T fp,
                            T /*slope*/) {
    T alpha{1};
    for (int k = 0; k < 40 && alpha > EPS; ++k, alpha *= T{0.5}) {
      if (eval_grad(xc + alpha * xi).first <= fp + C1 * alpha * xi.dot(xi)) {
        break;
      }
    }
    return alpha * xi;
  };

  QNDirState<T, N, Update> ds;
  auto [fp, g] = eval_grad(x);
  for (int i = 0; i < itmax; ++i) {
    const T den = max(abs(fp), T{1});
    const T scaled_inf =
        (g.cwiseAbs().array() * x.cwiseAbs().cwiseMax(T{1}).array()).maxCoeff();
    if (scaled_inf / den < ftol) {
      break;
    }
    Point xi = ds.compute(g);
    if (g.dot(xi) >= T{}) {
      ds.reset();
      xi = -g;
    }
    const Point dx = line_search_fn(x, xi, fp, g.dot(xi));
    x += dx;
    auto [fn, g_new] = eval_grad(x);
    fp = fn;
    ds.update(dx, g_new - g);
    g = std::move(g_new);
  }
  return x;
}

} // namespace exprmin
