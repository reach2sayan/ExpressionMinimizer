#pragma once

// ── sym.hpp — symbol-system adapter ──────────────────────────────────────────
//
// ExpressionDifferentiator owns the symbol representation.  When it migrates
// from  integral_constant<char, C>  to  symbol_type<FixedString{...}>  (or
// anything else), only this file needs updating.  All ExpressionMinimizer code
// that constructs or partitions symbol lists imports from here.
//
// Current representation: integral_constant<char, C>  (ExprDiff ≤ v1.x)

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <type_traits>

namespace exprmin {

namespace mp = boost::mp11;

// ── sym_t<C> ──────────────────────────────────────────────────────────────────
// The type-list element representing a single symbol labelled C.
// Update this alias when ExpressionDifferentiator changes its symbol type.
template <char C>
using sym_t = std::integral_constant<char, C>;

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
