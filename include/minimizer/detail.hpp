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

  if (fb > fa) {
    swap(ax, bx);
    swap(fa, fb);
  }
  cx = bx + GOLD * (bx - ax);
  fc = f(cx);

  while (fb > fc) {
    const T r = (bx - ax) * (fb - fc);
    const T q = (bx - cx) * (fb - fa);
    const T qdiff = q - r;
    const T denom =
        T{2} * (qdiff >= T{} ? T{1} : T{-1}) * max(abs(qdiff), TINY);
    T u = bx - ((bx - cx) * q - (bx - ax) * r) / denom;
    T ulim = bx + GLIMIT * (cx - bx);
    T fu;

    if ((bx - u) * (u - cx) > T{}) {
      fu = f(u);
      if (fu < fc) {
        ax = bx;
        fa = fb;
        bx = u;
        fb = fu;
        return;
      }
      if (fu > fb) {
        cx = u;
        fc = fu;
        return;
      }
      u = cx + GOLD * (cx - bx);
      fu = f(u);
    } else if ((cx - u) * (u - ulim) > T{}) {
      fu = f(u);
      if (fu < fc) {
        bx = cx;
        fb = fc;
        cx = u;
        fc = fu;
        u = cx + GOLD * (cx - bx);
        fu = f(u);
      }
    } else if ((u - ulim) * (ulim - cx) >= T{}) {
      u = ulim;
      fu = f(u);
    } else {
      u = cx + GOLD * (cx - bx);
      fu = f(u);
    }
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

// ── Shared BFGS core ─────────────────────────────────────────────────────────
// LineSearch(xc, xi, fp, slope) → Point  (the step dx = α·xi)
// EvalGrad(x) → std::pair<T, Eigen::Vector<T,N>>  (value, gradient)
template <typename T, int N, typename EvalGrad, typename LineSearch>
constexpr Eigen::Vector<T, N> bfgs_impl(EvalGrad &eval_grad,
                                        Eigen::Vector<T, N> x, T ftol,
                                        int itmax, LineSearch line_search) {
  using std::abs, std::max;
  using Point = Eigen::Vector<T, N>;
  using Hessian = Eigen::Matrix<T, N, N>;
  constexpr T EPS = std::numeric_limits<T>::epsilon();

  Hessian H = Hessian::Identity();
  auto [fp, g] = eval_grad(x);
  Point xi = -g;

  for (int it = 0; it < itmax; ++it) {
    T slope = g.dot(xi);
    if (slope >= T{}) {
      H = Hessian::Identity();
      xi = -g;
      slope = -g.squaredNorm();
      if (slope == T{}) {
        break;
      }
    }

    const Point dx = line_search(x, xi, fp, slope);
    x += dx;
    auto [fn, g_new] = eval_grad(x);

    const T den = max(abs(fn), T{1});
    if ((g_new.cwiseAbs().array() * x.cwiseAbs().cwiseMax(T{1}).array())
                .maxCoeff() /
            den <
        ftol)
      break;

    const Point dg = g_new - g;
    const Point Hdg = H * dg;
    T fac = dg.dot(dx);
    const T fae = dg.dot(Hdg);

    if (fac > T{} && fac * fac > EPS * dg.squaredNorm() * dx.squaredNorm()) {
      fac = T{1} / fac;
      const T fad = T{1} / fae;
      const Point u = fac * dx - fad * Hdg;
      H += fac * dx * dx.transpose();
      H -= fad * Hdg * Hdg.transpose();
      H += fae * u * u.transpose();
    }

    xi = -(H * g_new);
    g = std::move(g_new);
    fp = fn;
  }
  return x;
}

// Backtracking Armijo line search (for non-CExpression objectives, e.g.
// AugLag).
template <diff::Numeric T, int N, typename EvalGrad>
constexpr Eigen::Vector<T, N>
bfgs_armijo(EvalGrad eval_grad, Eigen::Vector<T, N> x, T ftol, int itmax) {
  using Point = Eigen::Vector<T, N>;
  constexpr T EPS = std::numeric_limits<T>::epsilon();
  constexpr T C1{1e-4};
  auto ls = [&](const Point &xc, const Point &xi, T fp, T slope) {
    T alpha{1};
    for (int k = 0; k < 40 && alpha > EPS; ++k, alpha *= T{0.5}) {
      if (eval_grad(xc + alpha * xi).first <= fp + C1 * alpha * slope) {
        break;
      }
    }
    return alpha * xi;
  };
  return bfgs_impl<T, N>(eval_grad, std::move(x), ftol, itmax, ls);
}

// Brent exact line search (bracket + Brent 1-D minimization along direction).
template <diff::Numeric T, int N, typename EvalGrad>
constexpr Eigen::Vector<T, N>
bfgs_brent(EvalGrad eval_grad, Eigen::Vector<T, N> x, T ftol, int itmax) {
  using Point = Eigen::Vector<T, N>;
  constexpr T ZEPS = std::numeric_limits<T>::epsilon() * T{1e-3};
  auto ls = [&](const Point &xc, const Point &xi, T /*fp*/, T /*slope*/) {
    auto f1d = [&](T t) { return eval_grad(xc + t * xi).first; };
    T ax{0}, bx{1}, cx, fa = f1d(ax), fb = f1d(bx), fc;
    bracket(f1d, ax, bx, cx, fa, fb, fc);
    const T alpha = brent(f1d, ax, bx, cx, ftol, ZEPS);
    return alpha * xi;
  };
  return bfgs_impl<T, N>(eval_grad, std::move(x), ftol, itmax, ls);
}

// Dbrent derivative-aware line search (secant on f′ = ∇f · dir).
template <diff::Numeric T, int N, typename EvalGrad>
constexpr Eigen::Vector<T, N>
bfgs_dbrent(EvalGrad eval_grad, Eigen::Vector<T, N> x, T ftol, int itmax) {
  using Point = Eigen::Vector<T, N>;
  constexpr T ZEPS = std::numeric_limits<T>::epsilon() * T{1e-3};
  auto ls = [&](const Point &xc, const Point &xi, T /*fp*/, T /*slope*/) {
    // Funcd wraps eval_grad to expose operator() and df() for dbrent.
    struct Funcd {
      EvalGrad &eg;
      const Point &xc, &xi;
      T operator()(T t) const { return eg(xc + t * xi).first; }
      T df(T t) const { return eg(xc + t * xi).second.dot(xi); }
    } fc{eval_grad, xc, xi};
    T ax{0}, bx{1}, cx, fa = fc(ax), fb = fc(bx), fc_val;
    bracket(fc, ax, bx, cx, fa, fb, fc_val);
    const T alpha = dbrent(fc, ax, bx, cx, ftol, ZEPS);
    return alpha * xi;
  };
  return bfgs_impl<T, N>(eval_grad, std::move(x), ftol, itmax, ls);
}

} // namespace exprmin::detail
