#pragma once

// Factory helpers that deduce all template parameters from arguments,
// for the optimizers where CTAD alone is insufficient.

#include "dogleg.hpp"
#include "frprmn.hpp"
#include "../lsq/gaussnewton.hpp"
#include "lbfgs.hpp"
#include "../lsq/levmar.hpp"
#include "../lsq/nlsdogleg.hpp"
#include "../lsq/subspace2d.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

namespace exprmin {

namespace mp = boost::mp11;

// ── NLS residual-packing ────────────────────────────────────────────────────

template <DoglegVariant DV = DoglegVariant::Standard, diff::CExpression... Rs>
auto make_nls_dogleg(Rs... rs) {
  return NLSDogleg<diff::Equation<Rs...>, DV>{diff::Equation{std::move(rs)...}};
}

template <diff::CExpression... Rs>
auto make_subspace2d(Rs... rs) {
  return Subspace2D<diff::Equation<Rs...>>{diff::Equation{std::move(rs)...}};
}

// ── Curve-fitting symbol-partition ─────────────────────────────────────────
// Specify INPUT symbol chars as template args; parameter symbols are deduced
// as (all symbols in Expr) minus (InputChars).

template <char... InputChars, diff::CExpression Expr>
auto make_lm(Expr e, typename Expr::value_type ftol = 1e-8) {
  using AllSyms   = diff::extract_symbols_from_expr_t<Expr>;
  using InputSyms = mp::mp_list<std::integral_constant<char, InputChars>...>;
  using ParamSyms = mp::mp_set_difference<AllSyms, InputSyms>;
  return LevenbergMarquardt<Expr, ParamSyms, InputSyms>{std::move(e), ftol};
}

template <char... InputChars, diff::CExpression Expr>
auto make_gn(Expr e, typename Expr::value_type ftol = 1e-8) {
  using AllSyms   = diff::extract_symbols_from_expr_t<Expr>;
  using InputSyms = mp::mp_list<std::integral_constant<char, InputChars>...>;
  using ParamSyms = mp::mp_set_difference<AllSyms, InputSyms>;
  return GaussNewton<Expr, ParamSyms, InputSyms>{std::move(e), ftol};
}

// ── Optimizer policy ────────────────────────────────────────────────────────

template <template <diff::CExpression> class LS1D = Brent, int M = 10,
          diff::CExpression Expr>
auto make_lbfgs(Expr e, typename Expr::value_type gtol = 1e-8) {
  return LBFGS<Expr, LS1D, M>{std::move(e), gtol};
}

template <CGMethod Method = CGMethod::PolakRibiere,
          template <diff::CExpression> class LM = LinMin,
          diff::CExpression Expr>
auto make_frprmn(Expr e, typename Expr::value_type ftol = 3e-8) {
  return Frprmn<Expr, Method, LM>{std::move(e), ftol};
}

template <HessianMode HM = HessianMode::BFGS, diff::CExpression Expr>
auto make_dogleg(Expr e, typename Expr::value_type tol = 1e-8) {
  return Dogleg<Expr, HM>{std::move(e), tol};
}

} // namespace exprmin
