#pragma once

#include "detail.hpp"
#include "expressions.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <ranges>

namespace exprmin {

namespace mp = boost::mp11;

// NR §10.4 — Nelder-Mead downhill simplex method.
//
// Derivative-free N-dim minimizer.  Maintains a simplex of N+1 vertices,
// progressively shrinking it toward the minimum via reflection, expansion,
// and contraction moves.  No line search; no gradient required.
//
// Simplex is stored as an N×(N+1) Eigen matrix; each column is a vertex.
// FVals is an Eigen vector of length N+1 storing f at each vertex.
//
// amotry convention (NR §10.4):  psum = sum of all N+1 vertices; fac=-1
// reflects, fac=2 expands, fac=0.5 contracts.  Derivation:
//   centroid c = (psum − s.col(ihi)) / N
//   ptry = psum·fac1 − s.col(ihi)·fac2,  fac1=(1−fac)/N, fac2=fac1−fac
// which gives ptry = c + fac·(c − s.col(ihi)) for each sign of fac.
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
  constexpr explicit Amoeba(Expr e,
                            value_type ftol_ = static_cast<value_type>(3.0e-8))
      : expr(std::move(e)), ftol(ftol_) {}

  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }
  constexpr value_type operator()(const Point &p) { return eval_at(p); }
  constexpr value_type get_optimal_value() const { return fret; }
  constexpr Point minimize(const Point &p, const value_type &delta) {
    return minimize(detail::make_simplex(p, delta));
  }
  constexpr Point minimize(Simplex s);

private:
  constexpr value_type amotry(Simplex &s, FVals &y, Point &psum,
                              const std::size_t ihi, const value_type &fac) {
    const value_type fac1 = (value_type{1} - fac) / static_cast<value_type>(N);
    const value_type fac2 = fac1 - fac;
    const Point ptry = fac1 * psum - fac2 * s.col(ihi);
    const value_type ytry = eval_at(ptry);
    if (ytry < y[ihi]) {
      psum += ptry - s.col(ihi);
      s.col(ihi) = ptry;
      y[ihi] = ytry;
    }
    return ytry;
  }
};

template <diff::CExpression Expr>
constexpr typename Amoeba<Expr>::Point Amoeba<Expr>::minimize(Simplex s) {
  FVals y;
  for (auto i : std::views::iota(0uz, N + 1))
    y[i] = eval_at(s.col(i));

  Point psum = s.rowwise().sum();

  for (iter = 0; iter < ITMAX; ++iter) {
    // Indices of best (ilo), worst (ihi), second-worst (inhi)
    Eigen::Index ilo_idx;
    y.minCoeff(&ilo_idx);
    const std::size_t ilo = static_cast<std::size_t>(ilo_idx);

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

    value_type ytry = amotry(s, y, psum, ihi, value_type{-1});
    if (ytry <= y[ilo]) {
      amotry(s, y, psum, ihi, value_type{2});
    } else if (ytry >= y[inhi]) {
      const value_type ysave = y[ihi];
      ytry = amotry(s, y, psum, ihi, value_type{0.5});
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
