#pragma once

// ── sym.hpp — symbol-system adapter ──────────────────────────────────────────
//
// ExpressionDifferentiator owns the symbol representation.  When it migrates
// from  integral_constant<char, C>  to  symbol_type<FixedString{...}>  (or
// anything else), only this file needs updating.  All ExpressionMinimizer code
// that constructs or partitions symbol lists imports from here.
//
// Current representation: diff::symbol_type<diff::FixedString{...}>  (ExprDiff v1.x)

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <type_traits>

#include <expressions.hpp>

namespace exprmin {

namespace mp = boost::mp11;

// ── sym_t<C> ──────────────────────────────────────────────────────────────────
// The type-list element representing a single symbol labelled C.
// Update this alias when ExpressionDifferentiator changes its symbol type.
//
// ExprDiff v1.x keys symbols by  diff::symbol_type<diff::FixedString{...}>.
// A single-char input label C is widened to the FixedString{ C, '\0' } so the
// produced type is identical to what extract_symbols yields for Variable<…,"C">.
namespace detail {
template <char C> consteval diff::FixedString<2> char_to_fixed_string() {
  const char buf[2] = {C, '\0'};
  return diff::FixedString<2>{buf};
}
} // namespace detail

template <char C>
using sym_t = diff::symbol_type<detail::char_to_fixed_string<C>()>;

// ── sym_list_t<Cs...> ─────────────────────────────────────────────────────────
// An mp_list of sym_t for a pack of char labels.
// Usage in make.hpp:  using InputSyms = sym_list_t<InputChars...>;
template <char... Cs>
using sym_list_t = mp::mp_list<sym_t<Cs>...>;

// ── all_syms_t<Expr> ──────────────────────────────────────────────────────────
// All symbols extracted from an expression — delegated to ExprDiff.
template <diff::CExpression Expr>
using all_syms_t = diff::extract_symbols_from_expr_t<Expr>;

// ── param_syms_t<Expr, InputChars...> ────────────────────────────────────────
// Parameters = all symbols minus the designated input symbols.
// This is the partition used by make_lm / make_gn.
template <diff::CExpression Expr, char... InputChars>
using param_syms_t =
    mp::mp_set_difference<all_syms_t<Expr>, sym_list_t<InputChars...>>;

} // namespace exprmin
