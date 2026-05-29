#pragma once

#include "amoeba.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/list.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <ranges>

namespace exprmin {

namespace mp = boost::mp11;

/**
 * @brief NR §10.12.2 — Simulated annealing via a modified downhill simplex,
 *        followed by an exact Nelder–Mead cold refinement.
 *
 * **Hot phase**: the simplex is driven by noisy function values
 * @c yy[i] = y[i] + T·log(U),  U ~ Uniform(ε, 1).
 * Because log(U) ≤ 0, noise effectively lowers all stored values; as T → 0
 * this vanishes.  Uphill moves are accepted with probability ∝ exp(−ΔE/T),
 * allowing escape from shallow local minima.  Temperature is decayed by
 * @c cooling every @c epoch_steps iterations.
 *
 * **Cold phase**: once the hot loop ends, a fresh non-degenerate simplex is
 * built at the best SA point (side-length @c cold_delta) and standard Amoeba
 * convergence (no noise, relative spread < @c ftol) takes over.  This matches
 * NR's recommendation to "restart at the claimed minimum" to confirm it.
 *
 * @tparam Expr  A type satisfying the diff::CExpression concept.
 */
template <diff::CExpression Expr, typename Callbacks = callback::NoCallbacks>
struct SimAnneal {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N = mp::mp_size<Syms>::value;
  using Point = Eigen::Vector<value_type, static_cast<int>(N)>;
  using Simplex =
      Eigen::Matrix<value_type, static_cast<int>(N), static_cast<int>(N + 1)>;
  using FVals = Eigen::Vector<value_type, static_cast<int>(N + 1)>;

  static constexpr value_type TINY{1.0e-10};
  static constexpr int NMAX = 200'000;

private:
  Expr expr;
  value_type fret{};
  int iter{};
  value_type temperature;
  const value_type cooling;
  const int epoch_steps;
  const value_type ftol;
  const value_type cold_delta;
  [[no_unique_address]] Callbacks cbs_{};

  /**
   * @brief Samples the Boltzmann noise term T·log(U), U ~ Uniform(ε, 1).
   *
   * The result is always ≤ 0, so it acts as a temperature-scaled downward
   * shift on stored function values.  As T → 0 the noise vanishes and the
   * simplex reverts to deterministic Nelder–Mead behaviour.
   */
  struct BoltzmannSampler {
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<value_type> udist{
        std::numeric_limits<value_type>::epsilon(), value_type{1}};
    /// @brief Returns T·log(U) for the current temperature @p temp.
    constexpr value_type operator()(value_type temp) {
      return temp * std::log(udist(rng));
    }
  } bolt;

public:
  /**
   * @brief Constructs a SimAnneal minimizer.
   * @param e           Expression to minimize.
   * @param T0          Initial temperature (default 1).
   * @param cool        Multiplicative cooling factor per epoch, in (0, 1)
   *                    (default 0.9).
   * @param epoch       Number of simplex moves between temperature drops
   *                    (default 100).
   * @param ftol_       Convergence tolerance for the cold Amoeba phase
   *                    (relative spread between best and worst vertices,
   *                    default 3×10⁻⁸).
   * @param cold_delta_ Side-length of the simplex rebuilt at the SA best point
   *                    before the cold refinement (default 0.1).
   */
  constexpr explicit SimAnneal(
      Expr e, value_type T0 = value_type{1},
      value_type cool = static_cast<value_type>(0.9), int epoch = 100,
      value_type ftol_ = static_cast<value_type>(3.0e-8),
      value_type cold_delta_ = static_cast<value_type>(0.1),
      Callbacks cbs = {})
      : expr(std::move(e)), temperature(T0), cooling(cool), epoch_steps(epoch),
        ftol(ftol_), cold_delta(cold_delta_), cbs_(cbs), bolt{} {}

  /**
   * @brief Builds an initial simplex around @p p and minimizes.
   *
   * Constructs the simplex via exprmin::detail::make_simplex (vertex 0 = p;
   * vertex i+1 = p with component i perturbed by @p delta), then delegates
   * to the Simplex overload.
   *
   * @param p      Initial centre point.
   * @param delta  Side-length of the initial simplex.
   * @return Approximate minimizer after hot + cold phases.
   */
  constexpr Point minimize(const Point &p, const value_type &delta) {
    return minimize(detail::make_simplex(p, delta));
  }

  /**
   * @brief Runs SA hot phase then cold Amoeba refinement on a pre-built
   * simplex.
   *
   * Stores the best function value in fret; retrieve it with
   * get_optimal_value() after the call.
   *
   * @param s  N×(N+1) matrix whose columns are the initial simplex vertices.
   * @return Approximate minimizer.
   */
  constexpr Point minimize(Simplex s) {
    FVals y = FVals::NullaryExpr([&](Eigen::Index i) { return eval_at(s.col(i)); });
    auto &&[ybest, pbest] = HotPhaseSA(std::move(s), y);
    s = detail::make_simplex(pbest, cold_delta);
    std::tie(ybest, pbest) =
        ColdPhaseSA(s, std::move(y), std::move(pbest), std::move(ybest));
    fret = ybest;
    return pbest;
  }

  /// @brief Returns f at the best point found after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fret; }

  /// @brief Callable interface delegating to eval_at (used by amotry_cold).
  constexpr value_type operator()(const Point &p) { return eval_at(p); }

  /**
   * @brief Evaluates the expression at point @p p.
   * @param p  N-dimensional input point.
   * @return   f(p).
   */
  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

private:
  /**
   * @brief Hot SA phase: runs the noisy simplex until T < TINY or NMAX steps.
   *
   * Tracks the globally best (real) function value and its vertex throughout,
   * independent of the noisy @c yy comparison values used for moves.
   *
   * @param s  Initial simplex (consumed).
   * @param y  Initial real function values at each vertex (in/out).
   * @return   {ybest, pbest} — best real value and point seen during the run.
   */
  constexpr std::tuple<value_type, Point> HotPhaseSA(Simplex s, FVals &y);

  /**
   * @brief Cold Amoeba refinement from the best SA point.
   *
   * Rebuilds a fresh, non-degenerate simplex at @p pbest (side-length
   * @c cold_delta) to avoid the collapsed-simplex false-convergence that the
   * hot phase can leave behind, then runs standard Nelder–Mead until the
   * relative spread drops below @c ftol.
   *
   * @param s       Fresh simplex built at pbest (pre-computed by caller).
   * @param y       Function values at the simplex vertices (re-evaluated
   * inside).
   * @param pbest   Best point from the hot phase.
   * @param ybest   Best function value from the hot phase.
   * @return        {ybest_final, pbest_final} after cold convergence.
   */
  constexpr std::tuple<value_type, Point>
  ColdPhaseSA(Simplex s, FVals y, Point pbest, value_type ybest);

  /**
   * @brief Noisy amotry step for the hot SA phase.
   *
   * Reflects vertex @p ihi through the centroid scaled by @p fac, then
   * evaluates the real function value @c ytry_real and adds a Boltzmann noise
   * term to get the noisy comparison value @c ytry.  Updates the simplex and
   * both @c y (real) and @c yy (noisy) only if @c ytry < yy[ihi], so uphill
   * real moves can still be accepted when the noise term is favourable.
   *
   * @param s    Simplex (in/out).
   * @param y    Real function values (in/out).
   * @param yy   Noisy comparison values (in/out).
   * @param psum Column-sum cache (in/out).
   * @param ihi  Index of the worst (highest noisy value) vertex.
   * @param fac  Reflection factor (−1 reflect, 2 expand, 0.5 contract).
   * @return Noisy trial value ytry = f(ptry) + T·log(U).
   */
  constexpr value_type amotry(Simplex &s, FVals &y, FVals &yy, Point &psum,
                              const std::size_t ihi, const value_type &fac) {
    const value_type fac1 = (value_type{1} - fac) / static_cast<value_type>(N);
    const value_type fac2 = fac1 - fac;
    const Point ptry = fac1 * psum - fac2 * s.col(ihi);
    const value_type ytry_real = eval_at(ptry);
    const value_type ytry = ytry_real + bolt(temperature);
    if (ytry < yy[ihi]) {
      psum += ptry - s.col(ihi);
      s.col(ihi) = ptry;
      y[ihi] = ytry_real;
      yy[ihi] = ytry;
    }
    return ytry;
  }

  /**
   * @brief Noiseless amotry step for the cold Amoeba phase.
   *
   * Thin wrapper around detail::amotry_impl using real function values only
   * (no Boltzmann noise).
   *
   * @param s    Simplex (in/out).
   * @param y    Real function values (in/out).
   * @param psum Column-sum cache (in/out).
   * @param ihi  Index of the worst vertex.
   * @param fac  Reflection factor: −1 reflect, 2 expand, 0.5 contract.
   * @return Real trial value f(ptry).
   */
  constexpr value_type amotry_cold(Simplex &s, FVals &y, Point &psum,
                                   const std::size_t ihi,
                                   const value_type &fac) {
    return detail::amotry_impl<value_type, N>(*this, s, y, psum, ihi, fac);
  }
};

/**
 * @brief Hot SA main loop.
 *
 * Each iteration identifies ilo/ihi/inhi using the noisy @c yy values, then
 * performs the standard reflect → expand / contract → shrink sequence, but
 * using amotry (which adds Boltzmann noise) instead of the noiseless version.
 * Every @c epoch_steps iterations the temperature is decayed and @c yy is
 * refreshed with new noise samples at the current T.
 */
template <diff::CExpression Expr, typename Callbacks>
constexpr std::tuple<typename SimAnneal<Expr, Callbacks>::value_type,
                     typename SimAnneal<Expr, Callbacks>::Point>
SimAnneal<Expr, Callbacks>::HotPhaseSA(Simplex s, FVals &y) {

  FVals yy =
      y.unaryExpr([&, this](value_type v) { return v + this->bolt(temperature); });
  Eigen::Index ib_idx;
  y.minCoeff(&ib_idx);
  auto ib = static_cast<std::size_t>(ib_idx);
  value_type ybest = y[ib];
  Point pbest = s.col(ib);
  Point psum = s.rowwise().sum();

  for (iter = 0; iter < NMAX && temperature > TINY; ++iter) {
    if (iter > 0 && iter % epoch_steps == 0) {
      temperature *= cooling;
      yy = FVals::NullaryExpr([&](Eigen::Index i) { return y[i] + bolt(temperature); });
    }

    Eigen::Index ilo_idx;
    yy.minCoeff(&ilo_idx);
    const auto ilo = static_cast<std::size_t>(ilo_idx);
    Eigen::Index ihi_idx;
    yy.maxCoeff(&ihi_idx);
    const auto ihi = static_cast<std::size_t>(ihi_idx);
    auto not_ihi_yy = std::views::iota(0uz, N + 1) |
                      std::views::filter([ihi](auto i) { return i != ihi; });
    const std::size_t inhi = *std::ranges::max_element(
        not_ihi_yy, std::less{}, [&yy](std::size_t i) { return yy[i]; });

    // Track the best *real* value seen, independent of the noisy ranking
    Eigen::Index ibest_idx;
    const value_type ymin = y.minCoeff(&ibest_idx);
    if (ymin < ybest) {
      ybest = ymin;
      pbest = s.col(ibest_idx);
    }
    cbs_.on_anneal_iter(iter, temperature, ybest,
                        std::span<const value_type>(pbest.data(), pbest.size()));

    value_type ytry = amotry(s, y, yy, psum, ihi, value_type{-1});
    if (ytry <= yy[ilo]) {
      amotry(s, y, yy, psum, ihi, value_type{2});
    } else if (ytry >= yy[inhi]) {
      const value_type ysave = yy[ihi];
      ytry = amotry(s, y, yy, psum, ihi, value_type{0.5});
      if (ytry >= ysave) {
        // Contraction failed — shrink whole simplex toward best noisy vertex
        std::ranges::for_each(
            std::views::iota(0uz, N + 1) |
                std::views::filter([ilo](auto i) { return i != ilo; }),
            [&](std::size_t i) {
              s.col(i) = value_type{0.5} * (s.col(i) + s.col(ilo));
              y[i] = eval_at(s.col(i));
              yy[i] = y[i] + bolt(temperature);
            });
        psum = s.rowwise().sum();
      }
    }
  }
  return {ybest, pbest};
}

/**
 * @brief Cold Amoeba refinement loop.
 *
 * Standard noiseless Nelder–Mead on the fresh simplex built at pbest.
 * Convergence criterion: relative spread |y[ihi]−y[ilo]| / (|y[ihi]|+|y[ilo]|)
 * < ftol.  Keeps updating pbest/ybest throughout so the caller always gets the
 * globally best point regardless of where the simplex ends up.
 */
template <diff::CExpression Expr, typename Callbacks>
constexpr std::tuple<typename SimAnneal<Expr, Callbacks>::value_type,
                     typename SimAnneal<Expr, Callbacks>::Point>
SimAnneal<Expr, Callbacks>::ColdPhaseSA(Simplex s, FVals y, Point pbest,
                                        value_type ybest) {

  y = FVals::NullaryExpr([&](Eigen::Index i) { return eval_at(s.col(i)); });
  Point psum = s.rowwise().sum();

  static constexpr value_type ATINY{1.0e-20};
  for (int cold = 0; cold < NMAX; ++cold, ++iter) {
    Eigen::Index ilo_idx;
    y.minCoeff(&ilo_idx);
    const std::size_t ilo = static_cast<std::size_t>(ilo_idx);
    Eigen::Index ihi_idx;
    y.maxCoeff(&ihi_idx);
    const std::size_t ihi = static_cast<std::size_t>(ihi_idx);
    auto not_ihi_y = std::views::iota(0uz, N + 1) |
                     std::views::filter([ihi](auto i) { return i != ihi; });
    const std::size_t inhi = *std::ranges::max_element(
        not_ihi_y, std::less{}, [&y](std::size_t i) { return y[i]; });

    if (y[ilo] < ybest) {
      ybest = y[ilo];
      pbest = s.col(ilo);
    }

    const value_type denom = std::abs(y[ihi]) + std::abs(y[ilo]) + ATINY;
    if (value_type{2} * std::abs(y[ihi] - y[ilo]) / denom < ftol) {
      break;
    }

    value_type ytry = amotry_cold(s, y, psum, ihi, value_type{-1});
    if (ytry <= y[ilo]) {
      amotry_cold(s, y, psum, ihi, value_type{2});
    } else if (ytry >= y[inhi]) {
      const value_type ysave = y[ihi];
      ytry = amotry_cold(s, y, psum, ihi, value_type{0.5});
      if (ytry >= ysave) {
        // Contraction failed — shrink whole simplex toward best vertex
        std::ranges::for_each(
            std::views::iota(0uz, N + 1) |
                std::views::filter([ilo](auto i) { return i != ilo; }),
            [&](std::size_t i) {
              s.col(i) = value_type{0.5} * (s.col(i) + s.col(ilo));
              y[i] = eval_at(s.col(i));
            });
        psum = s.rowwise().sum();
      }
    }
  }

  Eigen::Index ilo_f_idx;
  y.minCoeff(&ilo_f_idx);
  const auto ilo_f = static_cast<std::size_t>(ilo_f_idx);
  if (y[ilo_f] < ybest) {
    ybest = y[ilo_f];
    pbest = s.col(ilo_f);
  }

  return {ybest, pbest};
}

template <diff::CExpression Expr> SimAnneal(Expr) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T>
SimAnneal(Expr, T) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T1, typename T2>
SimAnneal(Expr, T1, T2) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T1, typename T2, typename I>
SimAnneal(Expr, T1, T2, I) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T1, typename T2, typename I,
          typename T3>
SimAnneal(Expr, T1, T2, I, T3) -> SimAnneal<Expr>;
template <diff::CExpression Expr, typename T1, typename T2, typename I,
          typename T3, typename T4>
SimAnneal(Expr, T1, T2, I, T3, T4) -> SimAnneal<Expr>;

} // namespace exprmin
