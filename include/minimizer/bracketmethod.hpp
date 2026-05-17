#pragma once

#include <array>

#include "detail.hpp"
#include "expressions.hpp"
#include "traits.hpp"
#include <boost/mp11/list.hpp>

namespace exprmin {

namespace mp = boost::mp11;

// NR §10.1 — Bracketmethod adapted for expression templates.
//
// Instead of a functor f(x), the expression tree itself is the callable.
// eval_at(x) updates the single Variable in the tree via the existing
// .update(Syms{}, vals) + .eval() protocol, where Syms is deduced at
// compile time from extract_symbols_from_expr_t<Expr>.
template <diff::CExpression Expr> struct Bracketmethod {
  using value_type = typename Expr::value_type;
  using Syms = diff::extract_symbols_from_expr_t<Expr>;
  static constexpr std::size_t N1D = mp::mp_size<Syms>::value;
  Expr expr;
  value_type ax{}, bx{}, cx{};
  value_type fa{}, fb{}, fc{};

  constexpr explicit Bracketmethod(Expr e) : expr(std::move(e)) {}
  constexpr value_type eval_at(value_type x)
    requires(N1D == 1)
  {
    std::array<value_type, 1> v{x};
    expr.update(Syms{}, v);
    return expr.eval();
  }
  constexpr void bracket(const value_type &ax0, const value_type &bx0)
    requires(N1D == 1);
};

template <diff::CExpression Expr>
constexpr void Bracketmethod<Expr>::bracket(const value_type &ax0,
                                            const value_type &bx0)
  requires(N1D == 1)
{
  ax = ax0;
  bx = bx0;
  fa = eval_at(ax0);
  fb = eval_at(bx0);
  auto f = [this](value_type x) { return eval_at(x); };
  detail::bracket(f, ax, bx, cx, fa, fb, fc);
}

template <diff::CExpression Expr> Bracketmethod(Expr) -> Bracketmethod<Expr>;

} // namespace exprmin
