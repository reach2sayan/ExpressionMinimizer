#pragma once

#include "gradient.hpp"
#include "traits.hpp"
#include <Eigen/Dense>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <utility>
#include <vector>

namespace exprmin {

namespace mp = boost::mp11;

namespace detail {

// Compile-time indices of each element of SubSyms within AllSyms.
// Returns std::array<std::size_t, mp_size<SubSyms>::value>.
template <typename AllSyms, typename SubSyms> consteval auto sub_indices() {
  constexpr std::size_t NS = mp::mp_size<SubSyms>::value;
  std::array<std::size_t, NS> idx{};
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((idx[I] = mp::mp_find<AllSyms, mp::mp_at_c<SubSyms, I>>::value), ...);
  }(std::make_index_sequence<NS>{});
  return idx;
}

} // namespace detail

/**
 * @brief CRTP base for nonlinear least-squares solvers.
 *
 * Owns the expression and provides:
 * - Compile-time index maps between parameter/input symbols and the full
 *   symbol list (PARAM_IDX, INPUT_IDX).
 * - The DataPoint aggregate for observed data.
 * - make_all_vec — merges a ParamVec + InputVec into the AllSyms-sized
 *   vector the expression update() expects.
 * - eval_rJ — evaluates the residual vector r and Jacobian J over a dataset
 *   using reverse-mode AD for the partial derivatives.
 *
 * Derived classes (LevenbergMarquardt, GaussNewton) add convergence parameters
 * and a fit() method; they do not override any of the above helpers.
 *
 * @tparam Expr      A type satisfying diff::CExpression.
 * @tparam ParamSyms Compile-time list of parameter symbol chars
 *                   (default: all symbols in Expr).
 * @tparam InputSyms Compile-time list of per-data-point input symbol chars
 *                   (default: empty — pure parameter fitting).
 */
template <diff::CExpression Expr,
          typename ParamSyms = diff::extract_symbols_from_expr_t<Expr>,
          typename InputSyms = mp::mp_list<>>
struct LeastSquaresBase {
  using AllSyms = diff::extract_symbols_from_expr_t<Expr>;
  using value_type = typename Expr::value_type;

  static constexpr std::size_t N = mp::mp_size<ParamSyms>::value;
  static constexpr std::size_t K = mp::mp_size<InputSyms>::value;
  static constexpr std::size_t NALL = mp::mp_size<AllSyms>::value;

  using ParamVec = Eigen::Vector<value_type, static_cast<int>(N)>;
  using InputVec = Eigen::Vector<value_type, static_cast<int>(K)>;
  using AllVec = Eigen::Vector<value_type, static_cast<int>(NALL)>;

  static constexpr auto PARAM_IDX = detail::sub_indices<AllSyms, ParamSyms>();
  static constexpr auto INPUT_IDX = detail::sub_indices<AllSyms, InputSyms>();

  /// @brief One observed data point: input predictor, target response, weight
  /// 1/σᵢ.
  struct DataPoint {
    InputVec input;
    value_type target;
    value_type weight{1}; ///< Inverse noise scale 1/σᵢ (default: unweighted).
  };

protected:
  Expr expr;

  /// @brief Constructs the base, taking ownership of the expression.
  constexpr explicit LeastSquaresBase(Expr e) : expr(std::move(e)) {}

  /**
   * @brief Merges @p params and @p input into an AllSyms-sized vector for
   *        expression update().
   */
  constexpr AllVec make_all_vec(const ParamVec &params,
                                const InputVec &input) const;

  /**
   * @brief Evaluates the residual vector r and Jacobian J over @p data.
   *
   * Sign convention:
   * @code
   *   r[i] = wᵢ (yᵢ − f(xᵢ; a)),   J[i,j] = −wᵢ ∂f/∂aⱼ
   * @endcode
   * so the normal equations @c JᵀJ δa = −Jᵀr recover the standard NLS form.
   */
  constexpr auto eval_rJ(const ParamVec &params,
                         const std::vector<DataPoint> &data);
};

template <diff::CExpression Expr, typename ParamSyms, typename InputSyms>
constexpr typename LeastSquaresBase<Expr, ParamSyms, InputSyms>::AllVec
LeastSquaresBase<Expr, ParamSyms, InputSyms>::make_all_vec(
    const ParamVec &params, const InputVec &input) const {
  AllVec v;
  for (std::size_t j = 0; j < N; ++j) {
    v[static_cast<int>(PARAM_IDX[j])] = params[static_cast<int>(j)];
  }
  for (std::size_t k = 0; k < K; ++k) {
    v[static_cast<int>(INPUT_IDX[k])] = input[static_cast<int>(k)];
  }
  return v;
}

template <diff::CExpression Expr, typename ParamSyms, typename InputSyms>
constexpr auto LeastSquaresBase<Expr, ParamSyms, InputSyms>::eval_rJ(
    const ParamVec &params, const std::vector<DataPoint> &data) {
  const int M = static_cast<int>(data.size());
  using DynVec = Eigen::VectorX<value_type>;
  using JMat = Eigen::Matrix<value_type, Eigen::Dynamic, static_cast<int>(N)>;
  DynVec r(M);
  JMat J(M, static_cast<int>(N));

  for (int i = 0; i < M; ++i) {
    expr.update(AllSyms{}, make_all_vec(params, data[i].input));
    r[i] = data[i].weight * (data[i].target - expr);
    const auto g = diff::gradient<diff::DiffMode::Reverse>(expr);
    for (int j = 0; j < static_cast<int>(N); ++j) {
      J(i, j) = -data[i].weight * g[PARAM_IDX[j]];
    }
  }
  return std::pair{r, J};
}

} // namespace exprmin
