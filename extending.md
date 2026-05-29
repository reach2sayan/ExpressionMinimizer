# Extending ExpressionMinimizer

This document explains how to add custom callbacks and custom solver algorithms.

---

## Custom callbacks

Every algorithm accepts a `Callbacks` template parameter (default `NoCallbacks`).
Inherit from `CallbackBase<Derived>` and override only the hooks you care about.
All other hooks are inherited as zero-cost no-ops.

```cpp
#include "callback/callback.hpp"

struct MyCallback : exprmin::CallbackBase<MyCallback> {
    // Called at the top of each outer iteration.
    void on_lm_outer(int iter, double lambda, double chi2) noexcept {
        std::printf("[LM %3d]  λ=%.2e  χ²=%.6g\n", iter, lambda, chi2);
    }

    // Called after each accept/reject decision inside the LM λ-retry loop.
    void on_lm_inner(int iter, double lambda, double chi2_new, bool accepted) noexcept { ... }

    // Called once per outer quasi-Newton step (covers BFGS, L-BFGS).
    void on_qn_iter(int iter, double f, double scaled_grad_inf, double dx_norm) noexcept { ... }

    // Called once per trust-region outer iteration (Dogleg, NLSDogleg).
    void on_tr_iter(int iter, double phi, double gnorm,
                    double delta, double rho, bool accepted) noexcept { ... }

    // Called with the current iterate at every iteration — use for path recording.
    void on_iter_point(int iter, std::span<const double> x) noexcept { ... }

    // on_gn_iter, on_amoeba_iter, on_anneal_iter — same pattern
};
```

### Attaching a callback

Pass it as the second argument to any `make_*` factory (after the expression):

```cpp
auto lm    = exprmin::make_lm<'x'>(model, MyCallback{});
auto lbfgs = exprmin::make_lbfgs(f, MyCallback{});
auto dog   = exprmin::make_dogleg(f, MyCallback{});
exprmin::Amoeba amoeba{f, 3e-8, MyCallback{}};
```

For algorithms that don't have a matching factory overload, pass the callback
directly to the constructor as the last argument:

```cpp
exprmin::BFGS solver{f, /*gtol=*/1e-8, MyCallback{}};
```

### Composing multiple callbacks

`make_callbacks(cb1, cb2, ...)` wraps any number of `CallbackBase`-derived
objects into a single `CompositeCallbacks<Cb1, Cb2, ...>` that dispatches each
hook to all of them in order:

```cpp
auto cbs = exprmin::make_callbacks(
    exprmin::logging::LMSignalCallbacks(&signals),   // spdlog via signals2
    MyCallback{},                                      // custom profiler
    PathCapture{pts}                                   // path recorder
);
auto lm = exprmin::make_lm<'x'>(model, cbs);
```

`CompositeCallbacks` itself satisfies `CallbackType`, so it can be nested.

### The `CallbackType` concept

Any type that publicly inherits `CallbackBase<T>` satisfies `CallbackType`:

```cpp
static_assert(exprmin::CallbackType<MyCallback>);
static_assert(exprmin::CallbackType<exprmin::NoCallbacks>);
```

---

## Custom solvers

All algorithm families follow the same pattern: a CRTP base drives the
outer loop and calls back into the derived class for the problem-specific steps.
Inherit the appropriate base, implement its contract, and the loop is free.

---

### Unconstrained minimizer  (`QuasiNewtonBase`)

`QuasiNewtonBase<Expr, LS1D, Callbacks>` owns the expression, line-search
object, and the `quasi_newton_impl` loop.  Derive from it to supply a custom
direction-update strategy (e.g. a different Hessian approximation):

```cpp
#include "minimizer/bfgs.hpp"

template <diff::CExpression Expr,
          template <diff::CExpression> class LS1D = exprmin::Brent>
struct MyQuasiNewton : exprmin::QuasiNewtonBase<Expr, LS1D> {
    using Base = exprmin::QuasiNewtonBase<Expr, LS1D>;
    using typename Base::Point;
    using typename Base::value_type;

    static constexpr int ITMAX = 200;

    constexpr explicit MyQuasiNewton(Expr e, value_type gtol = 1e-8)
        : Base(std::move(e), gtol) {}

    constexpr Point minimize(Point p) {
        const auto eg  = [this](const Point &q) { return this->eval_grad(q); };
        auto ls_fn     = this->make_line_search_fn();
        MyDirState ds;   // your direction-state type (must have compute/update/reset)
        p = this->quasi_newton_impl(eg, std::move(p), this->gtol, ITMAX, ls_fn, ds,
                                    this->iter);
        this->fret = this->eval_at(p);
        return p;
    }

private:
    // Direction state must implement:
    //   Point compute(const Point &g) const   — returns search direction
    //   void  update(const Point &dx, const Point &dg)
    //   void  reset()
    struct MyDirState { ... };
};
```

Key methods inherited from `QuasiNewtonBase`:

| Method | Purpose |
|--------|---------|
| `eval_at(p)` | Evaluate `f(p)` |
| `eval_grad(p)` | Returns `{f(p), ∇f(p)}` via reverse-mode AD |
| `make_line_search_fn()` | Returns a `(xc, xi, fp, slope) → dx` callable |
| `quasi_newton_impl(...)` | The shared BFGS/L-BFGS outer loop |
| `fret` | Stores `f` at the best point after `minimize` |
| `iter` | Iteration counter (protected) |

---

### Trust-region minimizer  (`TrustRegionBase`)

`TrustRegionBase<Derived, T, N>` drives the Nocedal & Wright Algorithm 4.1
trust-region outer loop.  Derive from it and implement five CRTP methods:

```cpp
#include "minimizer/trustregionbase.hpp"

template <diff::CExpression Expr>
struct MyTrustRegion
    : exprmin::TrustRegionBase<MyTrustRegion<Expr>,
                                typename Expr::value_type,
                                /* N = */ static_cast<int>(
                                    mp::mp_size<diff::extract_symbols_from_expr_t<Expr>>::value)>
{
    using value_type = typename Expr::value_type;
    using Base       = exprmin::TrustRegionBase<MyTrustRegion<Expr>, value_type, N>;
    using ParamVec   = typename Base::ParamVec;
    using NMat       = typename Base::NMat;

    Expr expr;

    constexpr explicit MyTrustRegion(Expr e, value_type tol = 1e-8,
                                      int itmax = 200,
                                      value_type tr0 = 1e3,
                                      value_type trmin = 1e-12)
        : Base{tol, itmax, tr0, trmin}, expr{std::move(e)} {}

    // ── CRTP contract ─────────────────────────────────────────────────────

    // 1. Seed: return {f(p), ∇f(p), B(p)} — B is your Hessian approximation.
    constexpr std::tuple<value_type, ParamVec, NMat> eval_state(const ParamVec &p);

    // 2. Trial evaluation: return f(p_new) and cache state for commit_state.
    constexpr value_type eval_trial(const ParamVec &p_new);

    // 3. Subproblem: return a step with ‖step‖ ≤ delta.
    constexpr ParamVec compute_step(const ParamVec &g, const NMat &B,
                                    value_type delta) const;

    // 4. Radius policy: shrink/expand delta based on rho.
    constexpr void adjust_tr(value_type &delta, value_type rho, bool at_boundary);

    // 5. Promote trial state: return {g_new, B_new} after an accepted step.
    constexpr std::pair<ParamVec, NMat> commit_state(const ParamVec &step,
                                                      const ParamVec &g_old,
                                                      const NMat &B_cur);

    // Optional: refresh B exactly at the start of each iteration (default no-op).
    // constexpr void refresh_hessian(const ParamVec &p, NMat &B) { ... }
};
```

The loop is entirely in `TrustRegionBase::minimize(p)` — you never write it.
See `include/minimizer/dogleg.hpp` for a full worked example.

---

### NLS trust-region minimizer  (`NLSTrustRegionBase`)

`NLSTrustRegionBase<Derived, RExprs...>` specialises `TrustRegionBase` for
systems of residual expressions.  It implements `eval_state`, `eval_trial`,
`adjust_tr`, and `commit_state` using the Gauss-Newton approximation
B ≈ JᵀJ.  You only need to supply `compute_step`:

```cpp
#include "lsq/nlsdogleg.hpp"  // or trustregionbase.hpp directly

template <diff::CExpression... RExprs>
struct MyNLSSolver
    : exprmin::NLSTrustRegionBase<MyNLSSolver<RExprs...>, RExprs...>
{
    using Base     = exprmin::NLSTrustRegionBase<MyNLSSolver<RExprs...>, RExprs...>;
    using ParamVec = typename Base::ParamVec;
    using NMat     = typename Base::NMat;

    using Base::Base;   // inherit NLSTrustRegionBase constructor

    // The only method you must provide: solve the TR subproblem
    // min m(p) = φ + gᵀp + ½pᵀBp  s.t.  ‖p‖ ≤ Δ
    constexpr ParamVec compute_step(const ParamVec &g, const NMat &B,
                                    typename Base::value_type delta) const {
        // your step computation — see NLSDogleg for the dogleg geometry
        // this->current_J() gives the cached Jacobian from eval_state
        ...
    }
};

// Usage:
auto solver = MyNLSSolver{diff::Equation{r1, r2, r3}};
auto p = solver.minimize({1.0, 0.0});
```

See `include/lsq/nlsdogleg.hpp` and `include/lsq/subspace2d.hpp` for
full implementations of `compute_step` using this pattern.

---

### Curve-fitting solver  (`LeastSquaresBase`)

`LeastSquaresBase<Expr, ParamSyms, InputSyms>` handles data-parallel residual
and Jacobian evaluation.  Derive from it to implement a custom fitting loop:

```cpp
#include "lsq/lsq_base.hpp"

template <diff::CExpression Expr,
          typename ParamSyms = diff::extract_symbols_from_expr_t<Expr>,
          typename InputSyms = mp::mp_list<>>
struct MyFitter : exprmin::LeastSquaresBase<Expr, ParamSyms, InputSyms> {
    using Base       = exprmin::LeastSquaresBase<Expr, ParamSyms, InputSyms>;
    using typename Base::DataPoint;
    using typename Base::ParamVec;
    using typename Base::value_type;

    constexpr explicit MyFitter(Expr e, value_type tol = 1e-8)
        : Base{std::move(e)}, tol_{tol} {}

    constexpr ParamVec fit(ParamVec params,
                           const std::vector<DataPoint> &data) {
        for (int iter = 0; iter < 1000; ++iter) {
            // eval_rJ evaluates the weighted residual r and Jacobian J
            // over the entire dataset in one call.
            auto [r, J] = this->eval_rJ(params, data);

            // your update rule here — LM, Gauss-Newton, IRLS, etc.
            const auto step = ...(J, r);
            params += step;

            if (step.norm() < tol_) break;
        }
        return params;
    }

private:
    value_type tol_;
};
```

Key members inherited from `LeastSquaresBase`:

| Member | Type | Purpose |
|--------|------|---------|
| `eval_rJ(params, data)` | `→ {RVec r, JMat J}` | Weighted residuals and Jacobian over the dataset |
| `N` | `constexpr std::size_t` | Number of parameters |
| `M_per_point` | `constexpr std::size_t` | Outputs per data point (1 for scalar models) |
| `DataPoint` | struct | `{InputVec inputs, value_type y_obs, value_type weight}` |
| `ParamVec` | `Eigen::Vector<T, N>` | Parameter vector type |

The symbol partition (`ParamSyms` vs `InputSyms`) is managed automatically
when constructing through `make_lm<'x'>` / `make_gn<'x'>`.  For direct
construction, use the `exprmin::param_syms_t` and `exprmin::sym_list_t`
helpers from `sym.hpp`.

---

## Algorithm policy customization

These algorithms are not CRTP-based but accept policy template parameters that
let you swap line-search strategies, Hessian modes, or step variants without
subclassing anything.

---

### Conjugate gradient  (`Frprmn` / `DFrprmn`)

`Frprmn<Expr, Method, LM>` has two policy knobs:

| Parameter | Type | Default | Options |
|-----------|------|---------|---------|
| `Method`  | `CGMethod` | `PolakRibiere` | `FletcherReeves` — simpler, guaranteed descent with exact line search |
| `LM`      | template | `LinMin` | `DLinMin` — exploits the directional derivative via reverse-mode AD |

```cpp
#include "minimizer/make.hpp"    // make_frprmn
#include "minimizer/frprmn.hpp"  // CGMethod, DFrprmn

// Polak–Ribière (default) — self-restarts when gradient barely changes direction
auto cg_pr  = exprmin::make_frprmn(f);

// Fletcher–Reeves
auto cg_fr  = exprmin::make_frprmn<exprmin::CGMethod::FletcherReeves>(f);

// Derivative-aware line search (DLinMin) — fewer function evaluations per step
auto cg_d   = exprmin::DFrprmn<decltype(f)>{f};   // convenience alias
// or via the factory:
auto cg_d2  = exprmin::make_frprmn<exprmin::CGMethod::PolakRibiere,
                                    exprmin::DLinMin>(f);
```

---

### Trust-region Hessian mode  (`Dogleg`)

`Dogleg<Expr, HM, Callbacks>` selects how the Hessian B is maintained:

| `HessianMode` | Description |
|---------------|-------------|
| `BFGS` (default) | Rank-2 quasi-Newton update — no second derivatives needed |
| `ExactAD`        | Recomputes the exact Hessian at every iteration via reverse-mode AD |

```cpp
#include "minimizer/make.hpp"    // make_dogleg
#include "minimizer/dogleg.hpp"  // HessianMode

auto dog_qn = exprmin::make_dogleg(f);   // BFGS Hessian (default)

auto dog_ex = exprmin::make_dogleg<exprmin::HessianMode::ExactAD>(f);
```

Use `ExactAD` when the expression is cheap to differentiate twice and you
want fast quadratic convergence near the minimum.

---

### L-BFGS history depth and line search  (`LBFGS`)

`LBFGS<Expr, LS1D, M, Callbacks>` exposes two policy parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `LS1D`    | `Brent` | Line-search template; swap `Dbrent` to use directional derivatives |
| `M`       | `10`    | Number of `(s, y)` vector pairs retained (history depth) |

```cpp
#include "minimizer/make.hpp"   // make_lbfgs
#include "minimizer/brent.hpp"  // Dbrent

// Default: Brent line search, 10 history vectors
auto lb1 = exprmin::make_lbfgs(f);

// Derivative-aware line search, 20-vector history
auto lb2 = exprmin::make_lbfgs<exprmin::Dbrent, 20>(f);
```

For smooth problems where evaluating ∇f is cheap, `Dbrent` typically
halves the number of line-search evaluations.

---

### NLS dogleg step variant  (`NLSDogleg`)

`NLSDogleg<System, DV, Callbacks>` selects the step geometry inside the
trust region:

| `DoglegVariant`    | Description |
|--------------------|-------------|
| `Standard` (default) | Classic Powell dogleg: Cauchy → full Gauss-Newton |
| `Double`             | Double-dogleg: adds a scaled intermediate step; smaller trust-region subproblem error |

```cpp
#include "minimizer/make.hpp"    // make_nls_dogleg
#include "lsq/nlsdogleg.hpp"     // DoglegVariant

auto nls_std = exprmin::make_nls_dogleg(r1, r2, r3);   // Standard (default)

auto nls_dbl = exprmin::make_nls_dogleg<exprmin::DoglegVariant::Double>(r1, r2, r3);
```

---

## Constrained minimization  (`AugLag`)

`AugLag<Obj, EqConstraints, IneqConstraints>` minimises `f(x)` subject to
equality constraints `hᵢ(x) = 0` and/or inequality constraints `gⱼ(x) ≤ 0`
using the Birgin–Martínez augmented Lagrangian algorithm.

```cpp
#include "minimizer/auglag.hpp"

using namespace diff::literals;   // 'x'_var, 'y'_var, etc.

auto x = 'x'_var, y = 'y'_var;
auto f = x*x + y*y;              // objective

// Equality constraint: x + y = 1  →  express as h(x,y) = x + y − 1 = 0
auto h = x + y - 1.0;

// Inequality constraint: x² + y² ≤ 1  →  express as g ≤ 0
auto g = x*x + y*y - 1.0;

// Equality only
AugLag al_eq{f, exprmin::make_eq(h), std::tuple{}};

// Inequality only
AugLag al_iq{f, std::tuple{}, exprmin::make_ineq(g)};

// Both
AugLag al{f, exprmin::make_eq(h), exprmin::make_ineq(g)};

// Solve from an initial guess
AugLag::Point x0{0.5, 0.5};
auto x_star = al.minimize(x0);
```

### Constructor parameters

```cpp
AugLag(Obj obj,
       EqConstraints   eq   = {},   // make_eq(...)   or std::tuple{}
       IneqConstraints ineq = {},   // make_ineq(...) or std::tuple{}
       value_type ftol  = 1e-8,    // ICM convergence tolerance
       value_type rho0  = 1.0)     // initial penalty parameter ρ
```

`rho0` is auto-scaled from the initial constraint violation inside
`minimize()`, so the default is usually fine.

### Constraint expression convention

- **Equality**: write each `hᵢ` so that `hᵢ(x) = 0` encodes the constraint.
- **Inequality**: write each `gⱼ` so that `gⱼ(x) ≤ 0` encodes the constraint.

Pass any number of expressions to `make_eq` / `make_ineq`; they are bundled
into a `diff::Equation` and sized at compile time.

### Inspecting the result

```cpp
al.minimize(x0);
double f_opt  = al.fret;   // f(x*) at the solution
int    iters  = al.iter;   // number of outer B&M iterations taken
```

---

## Root finding  (`Broyden`)

`Broyden<FExprs...>` solves the square nonlinear system `F(x) = 0` using
Broyden's rank-1 quasi-Newton method.  The initial inverse-Jacobian is
computed exactly via reverse-mode AD; subsequent updates are Jacobian-free.

```cpp
#include "rootfind/broyden.hpp"

using namespace diff::literals;

auto x = 'x'_var, y = 'y'_var;
auto f1 = x*x + y - 1.0;      // f1(x,y) = 0
auto f2 = x   - y*y;          // f2(x,y) = 0

// CTAD deduces Broyden<decltype(f1), decltype(f2)>
exprmin::Broyden solver{diff::Equation{f1, f2}};

Broyden::Point x0{1.0, 0.0};
auto root = solver.find_root(x0);

double phi = solver.residual_norm();  // ‖F(root)‖ at convergence
int    n   = solver.iter;             // iterations taken
```

### Constructor

```cpp
Broyden(diff::Equation<FExprs...> sys,
        value_type tol   = 1e-10,  // convergence tolerance on ‖F(x)‖
        int        itmax = 200)
```

The system must be **square**: the number of expressions equals the number
of free symbols.  A `static_assert` enforces this at compile time.
