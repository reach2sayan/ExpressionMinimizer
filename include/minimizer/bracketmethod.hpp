#pragma once

#include <array>

#include "detail.hpp"
#include "expressions.hpp"
#include "traits.hpp"
#include <boost/mp11/list.hpp>

namespace exprmin {

namespace mp = boost::mp11;

/**
 * @brief NR §10.1 — Downhill bracket search for a 1-D minimum.
 *
 * Expands [ax, bx] until a bracket (ax, bx, cx) satisfying f(bx) < f(ax)
 * and f(bx) < f(cx) is found.  Uses golden-ratio and parabolic extrapolation
 * steps, capped at GLIMIT×(cx−bx) to prevent runaway.
 *
 * @pre  @p fa = f(ax) and @p fb = f(bx) must be initialised by the caller.
 * @post ax < bx < cx and f(bx) ≤ min(f(ax), f(cx)).
 *
 * @tparam T  Numeric scalar type.
 * @tparam F  Callable with signature T(T).
 * @param f   Objective function.
 * @param ax  Left endpoint (in/out).
 * @param bx  Middle point (in/out).
 * @param cx  Right endpoint (out).
 * @param fa  f(ax) (in/out).
 * @param fb  f(bx) (in/out).
 * @param fc  f(cx) (out).
 */
struct BracketFn {
  template <diff::Numeric T, std::invocable<T> F>
  constexpr void operator()(F &f, T &ax, T &bx, T &cx, T &fa, T &fb,
                            T &fc) const {
    using std::abs;
    using std::max;
    using std::swap;
    constexpr T GOLD = static_cast<T>(std::numbers::phi_v<double>);
    constexpr T GLIMIT{100};
    constexpr T TINY{1.0e-20};

    auto reset = [&f](T &u, T &fu, const T cx, const T bx) {
      u = cx + GOLD * (cx - bx);
      fu = f(u);
    };

    if (fb > fa) {
      swap(ax, bx);
      swap(fa, fb);
    } // swap so A -> B is downhill
    cx = bx + GOLD * (bx - ax); // first guess
    fc = f(cx);

    while (fb > fc) {
      const T r = (bx - ax) * (fb - fc);
      const T q = (bx - cx) * (fb - fa);
      const T qdiff = q - r;
      const T denom =
          T{2} * (qdiff >= T{} ? T{1} : T{-1}) * max(abs(qdiff), TINY);
      T u = bx - ((bx - cx) * q - (bx - ax) * r) / denom;
      T ulim = bx + GLIMIT * (cx - bx);
      T fu = f(u);

      if ((bx - u) * (u - cx) > T{}) { // parabolic u in [b,c]
        fu = f(u);
        if (fu < fc) { // minimum between [b,c]
          ax = bx;
          fa = fb;
          bx = u;
          fb = fu;
          return;
        }
        if (fu > fb) { // minimum between [a,u]
          cx = u;
          fc = fu;
          return;
        }
        reset(u, fu, cx, bx);
      } else if ((cx - u) * (u - ulim) > T{}) { // parabolic between c and limit
        fu = f(u);
        if (fu < fc) {
          bx = cx;
          fb = fc;
          cx = u;
          fc = fu;
          reset(u, fu, cx, bx);
        }
      } else if ((u - ulim) * (ulim - cx) >= T{}) { // parabolic between u and lim
        u = ulim;
        fu = f(u);
      } else { // parabolic step overshot ax; take default GOLD step past cx
        reset(u, fu, cx, bx);
      }
      // eliminate oldest point and continue
      ax = bx;
      bx = cx;
      cx = u;
      fa = fb;
      fb = fc;
      fc = fu;
    }
  }
};

/// Callable object for the NR §10.1 bracket algorithm (see BracketFn).
inline constexpr BracketFn bracket{};

/**
 * @brief NR §10.1 — Bracketmethod adapted for expression templates.
 *
 * Instead of a raw functor f(x), the expression tree itself is the callable.
 * eval_at(x) updates the single Variable in the tree via the existing
 * .update(Syms{}, vals) + .eval() protocol, where Syms is deduced at
 * compile time from extract_symbols_from_expr_t<Expr>.
 */
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

private:
  // Binds this instance's (ax,bx,cx,fa,fb,fc) state to the bracket algorithm.
  template <std::invocable<value_type> F>
  constexpr void bracket_impl(F &f) {
    exprmin::bracket(f, ax, bx, cx, fa, fb, fc);
  }
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
  bracket_impl(f);
}

template <diff::CExpression Expr> Bracketmethod(Expr) -> Bracketmethod<Expr>;

} // namespace exprmin
