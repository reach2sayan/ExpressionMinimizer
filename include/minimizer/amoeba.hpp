#pragma once

#include "expressions.hpp"
#include "traits.hpp"
#include "../callback/callback.hpp"
#include <Eigen/Dense>
#include <array>
#include <boost/mp11/list.hpp>
#include <random>
#include <ranges>

namespace exprmin {

namespace mp = boost::mp11;

// ── Free helpers shared by Amoeba and SimAnneal ──────────────────────────────
namespace detail {

/**
 * @brief Constructs an (N+1)-vertex simplex around a centre point.
 *
 * Column 0 is @p p; column i+1 is @p p with component i displaced by
 * @p delta.  The result is an N×(N+1) matrix where each column is a vertex.
 * Shared by Amoeba and SimAnneal.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension of the space.
 * @param p      Centre vertex.
 * @param delta  Side-length of the initial simplex.
 * @return N×(N+1) matrix whose columns are the simplex vertices.
 */
template <typename T, int N>
constexpr Eigen::Matrix<T, N, N + 1> make_simplex(const Eigen::Vector<T, N> &p,
                                                  const T &delta) noexcept {
  Eigen::Matrix<T, N, N + 1> s = p.replicate(1, N + 1);
  s.diagonal(1).array() += delta;
  return s;
}

/**
 * @brief Nelder–Mead trial reflection/expansion/contraction step.
 *
 * Reflects vertex @p ihi through the centroid of the remaining vertices,
 * scaled by @p fac.  Updates @p s, @p y, and @p psum in place when the
 * trial point improves.
 *
 * @tparam T    Scalar type.
 * @tparam N    Simplex dimension.
 * @param ptr   Object exposing eval_at(Point) → T.
 * @param s     N×(N+1) simplex matrix (in/out).
 * @param y     Function values at each vertex (in/out).
 * @param psum  Column-sum cache (in/out).
 * @param ihi   Index of the highest (worst) vertex.
 * @param fac   Reflection factor: −1 = reflect, >1 = expand, 0<fac<1 =
 * contract.
 * @return Function value at the trial point.
 */
template <typename T, std::size_t N>
constexpr T
amotry_impl(auto &&ptr,
            Eigen::Matrix<T, static_cast<int>(N), static_cast<int>(N + 1)> &s,
            Eigen::Vector<T, static_cast<int>(N + 1)> &y,
            Eigen::Vector<T, static_cast<int>(N)> &psum, std::size_t ihi,
            T fac) {
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
}

/**
 * @brief Buffered normal-variate generator.
 *
 * Pre-generates @p BufSize normal-distributed values from @p RNG and serves
 * them from an internal array, refilling only when exhausted.  Amortises the
 * per-call overhead of std::normal_distribution across batches.
 *
 * @tparam T       Scalar type.
 * @tparam RNG     A UniformRandomBitGenerator (default std::mt19937).
 * @tparam BufSize Number of values to pre-generate per refill (default 512).
 */
template <typename T, typename RNG = std::mt19937, std::size_t BufSize = 512>
class RngBuffer {
  RNG rng_;
  std::normal_distribution<T> dist_{};
  std::array<T, BufSize> buf_;
  std::size_t pos_ = BufSize;

  void refill() {
    for (auto &v : buf_)
      v = dist_(rng_);
    pos_ = 0;
  }

public:
  explicit RngBuffer(RNG rng = RNG{std::random_device{}()})
      : rng_{std::move(rng)} {}

  T operator()() {
    if (pos_ == BufSize)
      refill();
    return buf_[pos_++];
  }
};

struct NoRng {}; ///< Zero-size placeholder used when RandomInit = false.

/**
 * @brief Constructs an (N+1)-vertex simplex around a centre point using a
 * random orthonormal basis.
 *
 * Column 0 is @p p; column i+1 is @p p displaced by @p delta along the i-th
 * column of a uniformly random orthonormal basis (obtained via thin QR of a
 * Gaussian random matrix).  This avoids axis-aligned bias and is the
 * initialisation used by GSL's @c nmsimplex2rand.
 *
 * @tparam T         Scalar type.
 * @tparam N         Dimension of the space.
 * @tparam NormalGen A callable returning T-distributed normal variates.
 * @param p          Centre vertex.
 * @param delta      Side-length of the initial simplex.
 * @param gen        Normal-variate source (e.g. an RngBuffer<T>).
 * @return N×(N+1) matrix whose columns are the simplex vertices.
 */
template <typename T, int N, typename NormalGen>
Eigen::Matrix<T, N, N + 1> make_simplex_rand(const Eigen::Vector<T, N> &p,
                                             const T &delta, NormalGen &&gen) {
  Eigen::Matrix<T, N, N> A = Eigen::Matrix<T, N, N>::NullaryExpr(
      [&](Eigen::Index, Eigen::Index) { return gen(); });
  const Eigen::Matrix<T, N, N> Q =
      Eigen::HouseholderQR<Eigen::Matrix<T, N, N>>{A}.householderQ() *
      Eigen::Matrix<T, N, N>::Identity();

  Eigen::Matrix<T, N, N + 1> s;
  s.col(0) = p;
  for (int i = 0; i < N; ++i)
    s.col(i + 1) = p + delta * Q.col(i);
  return s;
}

} // namespace detail

/**
 * @brief NR §10.5 — Nelder–Mead downhill simplex minimizer.
 *
 * Derivative-free N-dimensional minimizer.  Maintains a simplex of N+1
 * vertices, progressively shrinking it toward the minimum via reflection,
 * expansion, and contraction moves.  No line search or gradient required.
 *
 * The simplex is stored as an N×(N+1) Eigen matrix where each column is a
 * vertex; @c FVals is an (N+1)-vector of the corresponding function values.
 *
 * **amotry step convention** (NR §10.5): @c psum is the column-sum of all
 * N+1 vertices.  For reflection factor @c fac:
 * @code
 *   fac1 = (1 − fac) / N,  fac2 = fac1 − fac
 *   ptry = psum·fac1 − s.col(ihi)·fac2
 *        = c − fac·(c − s.col(ihi)),   c = centroid of the remaining N vertices
 * @endcode
 * Canonical values: fac = −1 (reflect), 2 (expand), 0.5 (contract).
 *
 * @tparam Expr        A type satisfying the diff::CExpression concept.
 * @tparam RandomInit  When @c true, minimize(p, delta) builds the initial
 *                     simplex from a random orthonormal basis (GSL
 * nmsimplex2rand style) using a cached detail::RngBuffer member.  When
 *                     @c false (default), the standard axis-aligned simplex is
 *                     used and no RNG state is stored.
 */
template <diff::CExpression Expr, bool RandomInit = false,
          typename Callbacks = callback::NoCallbacks>
struct Amoeba {
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

  using RngType = std::conditional_t<RandomInit, detail::RngBuffer<value_type>,
                                     detail::NoRng>;
  [[no_unique_address]] RngType rng_;
  [[no_unique_address]] Callbacks cbs_{};

public:
  /**
   * @brief Constructs an Amoeba minimizer wrapping the given expression.
   * @param e      Expression to minimize.
   * @param ftol_  Convergence tolerance on the relative function-value spread
   *               between the best and worst simplex vertices (default 3×10⁻⁸).
   */
  constexpr explicit Amoeba(Expr e,
                            value_type ftol_ = static_cast<value_type>(3.0e-8),
                            Callbacks cbs = {})
      : expr{std::move(e)}, ftol{ftol_}, cbs_(cbs) {}

  /**
   * @brief Evaluates the expression at point @p p.
   * @param p  N-dimensional input point.
   * @return   f(p).
   */
  constexpr value_type eval_at(const Point &p) {
    expr.update(Syms{}, p);
    return expr.eval();
  }

  /// @brief Callable interface; lets Amoeba act as a functor for
  /// detail::amotry_impl.
  constexpr value_type operator()(const Point &p) { return eval_at(p); }

  /// @brief Returns f at the best vertex after the last minimize() call.
  constexpr value_type get_optimal_value() const { return fret; }

  /**
   * @brief Builds an initial simplex around @p p and minimizes.
   *
   * When @c RandomInit = false (default), builds an axis-aligned simplex via
   * detail::make_simplex.  When @c RandomInit = true, builds a random
   * orthonormal-basis simplex via detail::make_simplex_rand drawing from the
   * cached member RngBuffer (state is preserved across calls).  Then delegates
   * to the Simplex overload.
   *
   * @param p      Initial centre point.
   * @param delta  Side-length of the initial simplex.
   * @return Approximate minimizer (best vertex on convergence or ITMAX).
   */
  Point minimize(const Point &p, const value_type &delta) {
    if constexpr (RandomInit)
      return minimize(
          detail::make_simplex_rand<value_type, static_cast<int>(N)>(p, delta,
                                                                     rng_));
    else
      return minimize(detail::make_simplex(p, delta));
  }

  /**
   * @brief Runs the Nelder–Mead loop on a pre-built simplex.
   * @param s  N×(N+1) matrix whose columns are the initial vertices.
   * @return Approximate minimizer (best vertex on convergence or ITMAX).
   */
  constexpr Point minimize(Simplex s);
};

/**
 * @brief Nelder–Mead main loop.
 *
 * Each iteration:
 *  1. Identifies the best (ilo), worst (ihi), and second-worst (inhi) vertices.
 *  2. Checks convergence: exits when the relative spread |y[ihi]−y[ilo]| /
 *     (|y[ihi]|+|y[ilo]|) drops below ftol.
 *  3. **Reflect** ihi through the centroid (fac = −1).
 *     - If the reflection beats the current best → **expand** (fac = 2).
 *     - Else if reflection is still worse than second-worst → **contract**
 *       (fac = 0.5).  If contraction also fails → **shrink** every vertex
 *       halfway toward the best.
 *
 * Returns the best vertex found, with the function value stored in fret.
 */
template <diff::CExpression Expr, bool RandomInit, typename Callbacks>
constexpr typename Amoeba<Expr, RandomInit, Callbacks>::Point
Amoeba<Expr, RandomInit, Callbacks>::minimize(Simplex s) {
  // Step 1: evaluate f at every initial vertex.
  FVals y =
      FVals::NullaryExpr([&](Eigen::Index i) { return eval_at(s.col(i)); });

  // Step 2: cache the column-sum; centroid of N+1 vertices = psum / (N+1).
  Point psum = s.rowwise().sum();
  for (iter = 0; iter < ITMAX; ++iter) {
    // Step 3: locate best (ilo), worst (ihi), and second-worst (inhi) vertices.
    Eigen::Index ilo_idx;
    y.minCoeff(&ilo_idx);
    const auto ilo = static_cast<std::size_t>(ilo_idx);

    Eigen::Index ihi_idx;
    y.maxCoeff(&ihi_idx);
    const auto ihi = static_cast<std::size_t>(ihi_idx);
    auto not_ihi = std::views::iota(0uz, N + 1) |
                   std::views::filter([ihi](auto i) { return i != ihi; });
    const std::size_t inhi = *std::ranges::max_element(
        not_ihi, std::less{}, [&y](std::size_t i) { return y[i]; });

    cbs_.on_amoeba_iter(iter, y[ilo], y[ihi]);
    // Step 4: converge when the relative spread between best and worst is
    // small.
    if (value_type{2} * std::abs(y[ihi] - y[ilo]) /
            (std::abs(y[ihi]) + std::abs(y[ilo]) + TINY) <
        ftol) {
      fret = y[ilo];
      return s.col(ilo);
    }

    // Step 5: reflect ihi through the centroid of the remaining N vertices.
    value_type ytry = detail::amotry_impl<value_type, N>(*this, s, y, psum, ihi,
                                                         value_type{-1});
    if (ytry <= y[ilo]) {
      // Step 6a: reflection beat the best — try expanding further.
      detail::amotry_impl<value_type, N>(*this, s, y, psum, ihi, value_type{2});
    } else if (ytry >= y[inhi]) {
      // Step 6b: reflection is still worse than second-worst — try contracting.
      const value_type ysave = y[ihi];
      ytry = detail::amotry_impl<value_type, N>(*this, s, y, psum, ihi,
                                                value_type{0.5});
      if (ytry >= ysave) {
        // Step 6c: contraction failed — shrink every vertex halfway toward ilo.
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
  // Step 7: ITMAX reached without convergence — return the best vertex found.
  Eigen::Index ilo_idx;
  y.minCoeff(&ilo_idx);
  fret = y[ilo_idx];
  return s.col(ilo_idx);
}

template <diff::CExpression Expr> Amoeba(Expr) -> Amoeba<Expr>;
template <diff::CExpression Expr, typename T> Amoeba(Expr, T) -> Amoeba<Expr>;
template <diff::CExpression Expr, typename T>
Amoeba(Expr, T, int) -> Amoeba<Expr>;

/// @brief Factory for the standard axis-aligned simplex variant.
template <diff::CExpression Expr>
auto make_amoeba(Expr e, typename Expr::value_type ftol =
                             static_cast<typename Expr::value_type>(3.0e-8)) {
  return Amoeba<Expr, false>{std::move(e), ftol};
}

/// @brief Factory for the random orthonormal-basis simplex variant.
template <diff::CExpression Expr>
auto make_amoeba_rand(Expr e,
                      typename Expr::value_type ftol =
                          static_cast<typename Expr::value_type>(3.0e-8)) {
  return Amoeba<Expr, true>{std::move(e), ftol};
}

} // namespace exprmin
