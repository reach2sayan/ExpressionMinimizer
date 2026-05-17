#pragma once

#include "detail.hpp"
#include "gradient.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>

namespace exprmin {

namespace mp = boost::mp11;

enum class HessianMode { BFGS, ExactAD };

// Powell dogleg trust-region minimizer.
//
// At each iteration builds the dogleg step from three cases:
//   1. Full Newton step inside trust region → take it.
//   2. Cauchy step already reaches trust-region boundary → scale down.
//   3. Otherwise → interpolate Cauchy→Newton to land on the boundary.
//
// HM == BFGS (default): Hessian approximated via rank-2 BFGS updates.
// HM == ExactAD:        exact Hessian computed every iteration via
//                       diff::derivative_tensor<2>(expr).
//
// Trust-region radius is adjusted via ρ = actual/predicted reduction.
template <diff::CExpression Expr, HessianMode HM = HessianMode::BFGS>
struct Dogleg {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr int N = static_cast<int>(mp::mp_size<Syms>::value);
  using Point = Eigen::Vector<value_type, N>;
  using Matrix = Eigen::Matrix<value_type, N, N>;

  value_type fret{};
  int iter{};

private:
  Expr expr;
  value_type tol;
  int itmax;
  value_type trustregion0;
  value_type trustregion_min;

  static constexpr value_type TR_DOWN_FACTOR = value_type{0.1};
  static constexpr value_type TR_DOWN_THRESHOLD = value_type{0.25};
  static constexpr value_type TR_UP_FACTOR = value_type{2.0};
  static constexpr value_type TR_UP_THRESHOLD = value_type{0.75};

  struct StepResult {
    Point step;
    bool at_boundary;
    value_type nn; // Newton-step norm, needed by trust-region shrink policy
  };

  // Three-way dogleg selection: Cauchy / interpolated / Newton.
  constexpr StepResult compute_step(const Point &g, const Matrix &B,
                                    value_type delta) const;

  // libdogleg ρ-based trust-region shrink/grow.
  constexpr void update_trust_region(value_type &delta, value_type rho,
                                     bool at_boundary, value_type nn) const;

  // BFGS rank-2 update of B: B ← B - (Bs)(Bs)ᵀ/(sᵀBs) + yyᵀ/(yᵀs).
  constexpr void update_B(Matrix &B, const Point &step, const Point &dg) const;

public:
  constexpr explicit Dogleg(Expr e, value_type tol_ = value_type{1e-8},
                            int itmax_ = 200,
                            value_type trustregion0_ = value_type{1e3},
                            value_type trustregion_min_ = value_type{1e-12})
      : expr{std::move(e)}, tol{tol_}, itmax{itmax_},
        trustregion0{trustregion0_}, trustregion_min{trustregion_min_} {}

  constexpr value_type operator()(const Point &p) { return eval_at(p); }
  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  constexpr std::pair<value_type, Point> eval_grad(const Point &p) {
    expr.update(Syms{}, p);
    const auto g_arr = diff::gradient<diff::DiffMode::Reverse>(expr);
    return {expr.eval(), Eigen::Map<const Point>(g_arr.data())};
  }

  constexpr Point minimize(Point p);
};

template <diff::CExpression Expr, HessianMode HM>
constexpr typename Dogleg<Expr, HM>::StepResult
Dogleg<Expr, HM>::compute_step(const Point &g, const Matrix &B,
                               value_type delta) const {
  using std::sqrt;

  // Cauchy step: steepest descent scaled to the quadratic-model minimum.
  //   k = -(gᵀg)/(gᵀBg),  p_c = k·g
  const value_type gBg = g.dot(B * g);
  const value_type kc =
      (gBg > value_type{0}) ? (-g.squaredNorm() / gBg) : value_type{-1};
  const Point p_c = kc * g;
  const value_type nc = p_c.norm();

  // Newton step: solve B·p_n = -g.
  const Point p_n = -(B.ldlt().solve(g));
  const value_type nn = p_n.norm();

  if (nn <= delta) {
    // Newton inside trust region — take it
    return {p_n, false, nn};
  } else if (nc >= delta) {
    // scale Cauchy to boundary
    return {(delta / nc) * p_c, true, nn};
  } else {
    // Interpolate p_c→p_n to land exactly on the boundary.
    // Solve: ‖p_c + k·(p_n − p_c)‖² = δ²
    //   l2·k² + 2c·k + (nc² − δ²) = 0,  c = (p_n − p_c)ᵀ p_c
    const Point d = p_n - p_c;
    const value_type l2 = d.squaredNorm();
    const value_type c = d.dot(p_c);
    value_type disc = c * c - l2 * (nc * nc - delta * delta);
    if (disc < value_type{0}) {
      disc = value_type{0};
    }
    return {p_c + ((-c + sqrt(disc)) / l2) * d, true, nn};
  }
}

template <diff::CExpression Expr, HessianMode HM>
constexpr void
Dogleg<Expr, HM>::update_trust_region(value_type &delta, value_type rho,
                                      bool at_boundary, value_type nn) const {
  if (rho < TR_DOWN_THRESHOLD) {
    if (!at_boundary) {
      delta = nn; // drop to Newton size before shrinking
    }
    delta *= TR_DOWN_FACTOR;
  } else if (rho > TR_UP_THRESHOLD && at_boundary) {
    delta *= TR_UP_FACTOR;
  }
}

template <diff::CExpression Expr, HessianMode HM>
constexpr void Dogleg<Expr, HM>::update_B(Matrix &B, const Point &step,
                                          const Point &dg) const {
  const Point Bs = B * step;
  const value_type sBs = step.dot(Bs);
  const value_type ys = dg.dot(step);
  if (ys > value_type{0} && sBs > value_type{0}) {
    B += (dg * dg.transpose()) / ys - (Bs * Bs.transpose()) / sBs;
  }
}

template <diff::CExpression Expr, HessianMode HM>
constexpr typename Dogleg<Expr, HM>::Point Dogleg<Expr, HM>::minimize(Point p) {
  using std::abs, std::max;

  Matrix B = Matrix::Identity();
  value_type delta = trustregion0;
  auto [fp, g] = eval_grad(p);
  for (iter = 0; iter < itmax; ++iter) {
    // Convergence: scaled gradient inf-norm.
    const value_type den = max(abs(fp), value_type{1});
    const value_type gnorm =
        (g.cwiseAbs().array() * p.cwiseAbs().cwiseMax(value_type{1}).array())
            .maxCoeff();
    if (gnorm / den < tol) {
      break;
    }

    if constexpr (HM == HessianMode::ExactAD) {
      expr.update(Syms{}, p);
      const auto H = diff::derivative_tensor<2>(expr);
      B = Eigen::Map<const Eigen::Matrix<value_type, N, N, Eigen::RowMajor>>(
          &H[0][0]);
    }

    const auto [step, at_boundary, nn] = compute_step(g, B, delta);
    const value_type predicted =
        -g.dot(step) - value_type{0.5} * step.dot(B * step);
    const Point p_new = p + step;
    const value_type rho = (predicted > value_type{0})
                               ? (fp - eval_at(p_new)) / predicted
                               : value_type{0};

    update_trust_region(delta, rho, at_boundary, nn);
    if (delta < trustregion_min) {
      break;
    }

    if (rho > value_type{0}) {
      auto [fn, g_new] = eval_grad(p_new);
      if constexpr (HM == HessianMode::BFGS) {
        update_B(B, step, g_new - g);
      }
      if (step.cwiseAbs().maxCoeff() < tol) {
        break;
      }
      p = p_new;
      fp = fn;
      g = std::move(g_new);
    }
  }

  fret = fp;
  return p;
}

template <diff::CExpression Expr> Dogleg(Expr) -> Dogleg<Expr>;
template <diff::CExpression Expr, typename T> Dogleg(Expr, T) -> Dogleg<Expr>;

} // namespace exprmin
