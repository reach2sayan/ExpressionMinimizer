#pragma once

/**
 * @file make.hpp
 * @brief Factory helpers that fully deduce all template parameters from
 *        arguments, for optimizers where CTAD alone is insufficient.
 */

#include "../lsq/gaussnewton.hpp"
#include "../lsq/levmar.hpp"
#include "../lsq/nlsdogleg.hpp"
#include "../lsq/subspace2d.hpp"
#include "dogleg.hpp"
#include "frprmn.hpp"
#include "lbfgs.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#ifdef EXPRMIN_LOGGING
#include "../logging/logging_callback.hpp"
#endif

namespace exprmin {

namespace mp = boost::mp11;

// ── NLS residual-packing ────────────────────────────────────────────────────

/**
 * @brief Builds an NLSDogleg solver from a pack of residual expressions.
 * @tparam DV   Dogleg variant (Standard or Double; default Standard).
 * @tparam Rs   Pack of diff::CExpression residuals.
 * @param rs    Residual expressions (passed by value, moved into the system).
 * @return      NLSDogleg<diff::Equation<Rs...>, DV>.
 */
template <DoglegVariant DV = DoglegVariant::Standard, diff::CExpression... Rs>
auto make_nls_dogleg(Rs... rs) {
  return NLSDogleg<diff::Equation<Rs...>, DV>{diff::Equation{std::move(rs)...}};
}

/**
 * @brief Builds a Subspace2D solver from a pack of residual expressions.
 * @tparam Rs  Pack of diff::CExpression residuals.
 * @param rs   Residual expressions (passed by value, moved into the system).
 * @return     Subspace2D<diff::Equation<Rs...>>.
 */
template <diff::CExpression... Rs> auto make_subspace2d(Rs... rs) {
  return Subspace2D<diff::Equation<Rs...>>{diff::Equation{std::move(rs)...}};
}

// ── Curve-fitting symbol-partition ─────────────────────────────────────────
// Specify INPUT symbol chars as template args; parameter symbols are deduced
// as (all symbols in Expr) minus (InputChars).

/**
 * @brief Builds a LevenbergMarquardt fitter with explicit input/parameter
 *        symbol partitioning.
 *
 * @tparam InputChars  Chars naming the per-data-point input symbols.
 * @tparam Expr        A type satisfying diff::CExpression.
 * @param e     Model expression.
 * @param ftol  Convergence tolerance (default 10⁻⁸).
 * @return      LevenbergMarquardt<Expr, ParamSyms, InputSyms>.
 */
template <char... InputChars, diff::CExpression Expr>
auto make_lm(Expr e, typename Expr::value_type ftol = 1e-8) {
  using AllSyms = diff::extract_symbols_from_expr_t<Expr>;
  using InputSyms = mp::mp_list<std::integral_constant<char, InputChars>...>;
  using ParamSyms = mp::mp_set_difference<AllSyms, InputSyms>;
  return LevenbergMarquardt<Expr, ParamSyms, InputSyms>{std::move(e), ftol};
}

/**
 * @brief Builds a GaussNewton fitter with explicit input/parameter symbol
 *        partitioning.
 *
 * @tparam InputChars  Chars naming the per-data-point input symbols.
 * @tparam Expr        A type satisfying diff::CExpression.
 * @param e     Model expression.
 * @param ftol  Convergence tolerance (default 10⁻⁸).
 * @return      GaussNewton<Expr, ParamSyms, InputSyms>.
 */
template <char... InputChars, diff::CExpression Expr>
auto make_gn(Expr e, typename Expr::value_type ftol = 1e-8) {
  using AllSyms = diff::extract_symbols_from_expr_t<Expr>;
  using InputSyms = mp::mp_list<std::integral_constant<char, InputChars>...>;
  using ParamSyms = mp::mp_set_difference<AllSyms, InputSyms>;
  return GaussNewton<Expr, ParamSyms, InputSyms>{std::move(e), ftol};
}

// ── Optimizer policy ────────────────────────────────────────────────────────

/**
 * @brief Builds an LBFGS minimizer with explicit line-search and history-depth
 *        policy.
 *
 * @tparam LS1D  Line-search template (default Brent).
 * @tparam M     History depth — number of (s, y) pairs retained (default 10).
 * @tparam Expr  A type satisfying diff::CExpression.
 * @param e     Expression to minimize.
 * @param gtol  Scaled-gradient convergence tolerance (default 10⁻⁸).
 * @return      LBFGS<Expr, LS1D, M>.
 */
template <template <diff::CExpression> class LS1D = Brent, int M = 10,
          diff::CExpression Expr>
auto make_lbfgs(Expr e, typename Expr::value_type gtol = 1e-8) {
  return LBFGS<Expr, LS1D, M>{std::move(e), gtol};
}

/**
 * @brief Builds a Frprmn (conjugate gradient) minimizer with explicit method
 *        and line-minimizer policy.
 *
 * @tparam Method  CGMethod (default PolakRibiere).
 * @tparam LM      Line-minimizer template (default LinMin).
 * @tparam Expr    A type satisfying diff::CExpression.
 * @param e     Expression to minimize.
 * @param ftol  Convergence tolerance (default 3×10⁻⁸).
 * @return      Frprmn<Expr, Method, LM>.
 */
template <CGMethod Method = CGMethod::PolakRibiere,
          template <diff::CExpression> class LM = LinMin,
          diff::CExpression Expr>
auto make_frprmn(Expr e, typename Expr::value_type ftol = 3e-8) {
  return Frprmn<Expr, Method, LM>{std::move(e), ftol};
}

/**
 * @brief Builds a Dogleg (Powell trust-region) minimizer with explicit Hessian
 *        mode.
 *
 * @tparam HM    Hessian mode (default BFGS).
 * @tparam Expr  A type satisfying diff::CExpression.
 * @param e    Expression to minimize.
 * @param tol  Convergence tolerance (default 10⁻⁸).
 * @return     Dogleg<Expr, HM>.
 */
template <HessianMode HM = HessianMode::BFGS, diff::CExpression Expr>
auto make_dogleg(Expr e, typename Expr::value_type tol = 1e-8) {
  return Dogleg<Expr, HM>{std::move(e), tol};
}

// ── Logged overloads — only when ENABLE_LOGGING is ON ────────────────────────
// Pass a *Signals& after the expression to get a logged solver back.
// The Signals object is stored by pointer (non-owning); it must outlive the
// returned solver.
#ifdef EXPRMIN_LOGGING

template <char... InputChars, diff::CExpression Expr>
auto make_lm(Expr e, logging::LMSignals &signals,
             typename Expr::value_type ftol = 1e-8) {
  using AllSyms = diff::extract_symbols_from_expr_t<Expr>;
  using InputSyms = mp::mp_list<std::integral_constant<char, InputChars>...>;
  using ParamSyms = mp::mp_set_difference<AllSyms, InputSyms>;
  return LevenbergMarquardt<Expr, ParamSyms, InputSyms,
                            logging::LMSignalCallbacks>{
      std::move(e), ftol, 1000, logging::LMSignalCallbacks(&signals)};
}

template <char... InputChars, diff::CExpression Expr>
auto make_gn(Expr e, logging::GNSignals &signals,
             typename Expr::value_type ftol = 1e-8) {
  using AllSyms = diff::extract_symbols_from_expr_t<Expr>;
  using InputSyms = mp::mp_list<std::integral_constant<char, InputChars>...>;
  using ParamSyms = mp::mp_set_difference<AllSyms, InputSyms>;
  return GaussNewton<Expr, ParamSyms, InputSyms, logging::GNSignalCallbacks>{
      std::move(e), ftol, 1000, logging::GNSignalCallbacks(&signals)};
}

template <template <diff::CExpression> class LS1D = Brent, int M = 10,
          diff::CExpression Expr>
auto make_lbfgs(Expr e, logging::QNSignals &signals,
                typename Expr::value_type gtol = 1e-8) {
  return LBFGS<Expr, LS1D, M, logging::QNSignalCallbacks>{
      std::move(e), gtol, logging::QNSignalCallbacks(&signals)};
}

template <HessianMode HM = HessianMode::BFGS, diff::CExpression Expr>
auto make_dogleg(Expr e, logging::TRSignals &signals,
                 typename Expr::value_type tol = 1e-8) {
  return Dogleg<Expr, HM, logging::TRSignalCallbacks>{
      std::move(e), tol, 200, 1e3, 1e-12, logging::TRSignalCallbacks(&signals)};
}

template <DoglegVariant DV = DoglegVariant::Standard, diff::CExpression... Rs>
auto make_nls_dogleg(logging::TRSignals &signals, Rs... rs) {
  return NLSDogleg<diff::Equation<Rs...>, DV, logging::TRSignalCallbacks>{
      diff::Equation{std::move(rs)...},    1e-8, 200, 1e3, 1e-12,
      logging::TRSignalCallbacks(&signals)};
}

#endif // EXPRMIN_LOGGING

} // namespace exprmin
