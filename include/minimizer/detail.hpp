#pragma once

#include "expressions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace exprmin::detail {

// Build an (N+1)-vertex simplex: col 0 = p, col i+1 = p with col[i+1][i] +=
// delta. Stored as an N×(N+1) Eigen matrix so each column is a Point.
template <typename T, int N>
Eigen::Matrix<T, N, N + 1> constexpr make_simplex(const Eigen::Vector<T, N> &p,
                                                  const T &delta) noexcept {
  Eigen::Matrix<T, N, N + 1> s = p.replicate(1, N + 1);
  s.diagonal(1).array() += delta;
  return s;
}

// NR §10.1 bracket algorithm for any callable T(T).
// fa = f(ax) and fb = f(bx) must be set by the caller before entry.
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

// NR §10.3 Brent's method for any callable T(T).
template <diff::Numeric T, std::invocable<T> F>
constexpr T brent(F &f, const T &ax, const T &bx, const T &cx, const T &tol,
                  const T &zeps, const int itmax = 100) {
  using std::abs;
  using std::max;
  using std::min;
  constexpr T CGOLD = static_cast<T>(1.0 - 1.0 / std::numbers::phi_v<double>);

  T a = min(ax, cx), b = max(ax, cx);
  T x = bx, w = bx, v = bx;
  T fx = f(x), fw = fx, fv = fx;
  T d{}, e{};

  for (int i = 0; i < itmax; ++i) {
    const T xm = T{0.5} * (a + b);
    const T tol1 = tol * abs(x) + zeps;
    const T tol2 = T{2} * tol1;

    if (abs(x - xm) <= tol2 - T{0.5} * (b - a)) {
      return x;
    }

    T u;
    if (abs(e) > tol1) {
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
        e = (x >= xm ? a - x : b - x);
        d = CGOLD * e;
      } else {
        d = p / q;
        u = x + d;
        if (u - a < tol2 || b - u < tol2) {
          d = (xm >= x ? tol1 : -tol1);
        }
      }
    } else {
      e = (x >= xm ? a - x : b - x);
      d = CGOLD * e;
    }

    u = (abs(d) >= tol1 ? x + d : x + (d >= T{} ? tol1 : -tol1));
    const T fu = f(u);

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

// NR §10.4 Dbrent — Brent's method with derivative information.
// Functor F must expose operator()(T) (value) and df(T) (derivative).
// Uses secant interpolation on f′; bisects toward the zero-crossing side
// when the secant step is outside the bracket or not improving.
template <diff::Numeric T, std::invocable<T> F>
constexpr T dbrent(F &f, const T &ax, const T &bx, const T &cx, const T &tol,
                   const T &zeps, int itmax = 100) {
  using std::abs;
  using std::max;
  using std::min;

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

    if (abs(e) > tol1) {
      // Attempt secant steps using (x,w) and (x,v) pairs
      T d1 = T{2} * (b - a), d2 = d1;
      if (dw != dx) {
        d1 = (w - x) * dx / (dx - dw);
      }
      if (dv != dx) {
        d2 = (v - x) * dx / (dx - dv);
      }
      const T u1 = x + d1, u2 = x + d2;
      const bool ok1 = (a - u1) * (u1 - b) > T{} && dx * d1 <= T{};
      const bool ok2 = (a - u2) * (u2 - b) > T{} && dx * d2 <= T{};
      const T olde = e;
      e = d;
      if (ok1 || ok2) {
        d = (ok1 && ok2) ? (abs(d1) < abs(d2) ? d1 : d2) : (ok1 ? d1 : d2);
        if (abs(d) <= abs(T{0.5} * olde)) {
          const T u = x + d;
          if (u - a < tol2 || b - u < tol2)
            d = (xm >= x ? tol1 : -tol1);
        } else {
          e = (dx >= T{} ? a - x : b - x);
          d = T{0.5} * e;
        }
      } else {
        e = (dx >= T{} ? a - x : b - x);
        d = T{0.5} * e;
      }
    } else {
      e = (dx >= T{} ? a - x : b - x);
      d = T{0.5} * e;
    }

    const T u = (abs(d) >= tol1 ? x + d : x + (d >= T{} ? tol1 : -tol1));
    const T fu = f(u);
    const T du = f.df(u);

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

// BFGS: maintains an N×N inverse-Hessian approximation H via the rank-2 update.
template <diff::Numeric T, int N> struct BFGSDirState {
  using Point = Eigen::Vector<T, N>;
  Eigen::Matrix<T, N, N> H = Eigen::Matrix<T, N, N>::Identity();

  constexpr Point compute(const Point &g) const { return -(H * g); }
  constexpr void reset() { H.setIdentity(); }
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

// L-BFGS: maintains a circular buffer of M most-recent (s,y) curvature pairs.
template <diff::Numeric T, int N, int M> struct LBFGSDirState {
  using Point = Eigen::Vector<T, N>;
  std::array<Point, M> s_buf, y_buf;
  std::array<T, M> rho_buf{};
  int buf_size = 0, buf_head = 0;

  // Nocedal two-loop recursion.
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

  constexpr void reset() {
    buf_size = 0;
    buf_head = 0;
  }
};

// Unified quasi-Newton loop.
// LineSearchFn: (xc, xi, fp, slope) → Point (step dx = α·xi).
// DirState: .compute(g)→xi, .update(dx,dg), .reset().
// iter_out is set to the iteration count on return.
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

// Backtracking Armijo on any eval_grad callable — used by AugLag (augmented
// Lagrangian objective is not a CExpression so it cannot own a Brent<Expr>).
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
