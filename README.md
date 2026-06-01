# ExpressionMinimizer

[![CI](https://github.com/reach2sayan/ExpressionMinimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/reach2sayan/ExpressionMinimizer/actions/workflows/ci.yml)

A header-only C++23 library of numerical minimization algorithms from
*Numerical Recipes* (NR) Ch. 10 & 15, wired to the
[ExpressionDifferentiator](https://github.com/reach2sayan/ExpressionDifferentiator)
expression-template engine. Gradients and Jacobians are obtained for free via
reverse-mode automatic differentiation — no hand-coded derivatives required.

## Algorithms

### Unconstrained minimization

| Class | Reference | Method |
|---|---|---|
| `Bracketmethod` | NR §10.1 | Downhill bracket search |
| `Golden` | NR §10.2 | Golden-section search |
| `Brent` | NR §10.3 | Parabolic interpolation + golden-section fallback |
| `Dbrent` | NR §10.4 | Brent with derivative (secant on f′ via reverse AD) |
| `LinMin` | NR §10.5 | Line minimization along a direction (uses `Brent`) |
| `DLinMin` | NR §10.5 | Line minimization with derivative (uses `Dbrent`) |
| `Amoeba` | NR §10.5 | Nelder–Mead downhill simplex |
| `SimAnneal` | NR §10.12 | Simulated annealing + Amoeba cold refinement |
| `Powell` | NR §10.6 | Powell's conjugate-direction method |
| `Frprmn` / `DFrprmn` | NR §10.6 | Fletcher–Reeves / Polak–Ribière conjugate gradient |
| `BFGS` | NR §10.7 | Quasi-Newton, rank-2 BFGS inverse-Hessian |
| `DFP` | NR §10.7 | Quasi-Newton, rank-2 DFP inverse-Hessian |
| `SR1` | Nocedal §6.2 | Quasi-Newton, rank-1 symmetric inverse-Hessian |
| `DBFGS` / `ABFGS` | — | BFGS + Dbrent / Armijo line search |
| `LBFGS` | Nocedal §7.2 | Limited-memory BFGS (two-loop, circular (s,y) buffer) |
| `Dogleg` | N&W §4.1 | Trust-region dogleg (BFGS or exact-AD Hessian) |

### Nonlinear least squares

| Class | Reference | Method |
|---|---|---|
| `NLSDogleg` | N&W §10.1 | NLS trust-region Powell dogleg (Standard + Double variant) |
| `Subspace2D` | Byrd et al. (1988) | NLS trust-region 2D subspace step (quartic secular equation) |
| `LevenbergMarquardt` | NR §15.5 | Marquardt-damped normal equations |
| `GaussNewton` | N&W §10.2 | Undamped normal equations |

### Constrained minimization

| Class | Reference | Method |
|---|---|---|
| `AugLag` | Birgin & Martínez (2014) | Augmented Lagrangian (equality + inequality constraints) |
| `SimplexLP` | NR §10.10 | Two-phase full-tableau simplex (LP, inequality form) |
| `InteriorPointLP` | NR §10.11 | Primal-dual infeasible interior-point method (LP, standard form) |

### Root finding

| Class | Reference | Method |
|---|---|---|
| `Broyden` | Broyden (1965) | Rank-1 quasi-Newton root finder for F(x) = 0 |

All multi-dimensional optimizers infer the problem dimension at compile time
from the `Variable` symbols in the expression tree.

## Line search policies

`BFGS`, `DFP`, `SR1`, and `LBFGS` accept a line search policy as a
template-template parameter:

| Policy | Description |
|---|---|
| `Brent` (default) | Bracket + parabolic/golden-section 1-D minimization |
| `Dbrent` | Brent with first-derivative (secant on f′) |
| `Armijo` | Backtracking sufficient-decrease (no bracketing required) |

```cpp
auto x = PV(0.0, 'x');
auto y = PV(0.0, 'y');
auto f = (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);

exprmin::BFGS               bfgs{f};                           // Brent (default)
exprmin::DBFGS<decltype(f)> dbfgs{f};                          // BFGS<Expr, Dbrent>
exprmin::ABFGS<decltype(f)> abfgs{f};                          // BFGS<Expr, Armijo>
exprmin::LBFGS              lbfgs{f};                          // Brent (default)
auto albfgs = exprmin::make_lbfgs<exprmin::Armijo>(f);         // Armijo
auto lbfgs5 = exprmin::make_lbfgs<exprmin::Brent, 5>(f);      // M=5 history pairs
```

## Dependencies

| Dependency | How it arrives |
|---|---|
| [ExpressionDifferentiator](https://github.com/reach2sayan/ExpressionDifferentiator) | git submodule (`deps/expression_differentiator`) — initialized automatically by CMake |
| [Eigen 3.4.0](https://eigen.tuxfamily.org) | `FetchContent` |
| [GoogleTest](https://github.com/google/googletest) | `FetchContent` (tests only) |
| Boost.Mp11 | pulled in transitively by ExpressionDifferentiator |

## Building

Requires CMake ≥ 3.20 and a C++23 compiler (tested: Clang 18, GCC 13).

```bash
git clone --recurse-submodules https://github.com/reach2sayan/ExpressionMinimizer.git
cd ExpressionMinimizer
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

The submodule is also initialized automatically during `cmake -B build` if you
cloned without `--recurse-submodules`.

### Running the tests

```bash
cmake --build build --target minimizer_tests
ctest --test-dir build --output-on-failure
```

## Usage

Include the umbrella header (pulls in all algorithms):

```cpp
#include "minimizer.hpp"
#include "expression_differentiator.hpp"
```

### 1-D minimization

```cpp
auto x = PV(0.0, 'x');
auto f = (x - 3.0) * (x - 3.0);

// Brent §10.3 — no derivatives needed
exprmin::Brent b{f};
double xmin = b.minimize(0.0, 5.0);   // bracket from [0, 5], then minimize

// Dbrent §10.4 — uses f′ from reverse-mode AD for faster convergence
exprmin::Dbrent db{f};
double xmin2 = db.minimize(0.0, 5.0);
```

### N-dimensional minimization

```cpp
auto x = PV(0.0, 'x');
auto y = PV(0.0, 'y');
auto f = (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);

// BFGS §10.9 — quasi-Newton, full inverse-Hessian approximation
exprmin::BFGS bfgs{f};
auto p = bfgs.minimize({0.0, 0.0});   // p ≈ {1.0, 2.0}

// DFP / SR1 — alternative quasi-Newton update formulas
exprmin::DFP<decltype(f)> dfp{f};
exprmin::SR1<decltype(f)> sr1{f};

// L-BFGS — same interface, O(M·N) memory instead of O(N²)
exprmin::LBFGS lbfgs{f};
auto p2 = lbfgs.minimize({0.0, 0.0});

// Conjugate gradient — Polak–Ribière (default) or Fletcher–Reeves
exprmin::Frprmn cg{f};
auto p3 = cg.minimize({0.0, 0.0});

auto cg_fr = exprmin::make_frprmn<exprmin::CGMethod::FletcherReeves>(f);

// Derivative-aware variants use Dbrent / DLinMin line search
exprmin::DFrprmn<decltype(f)> dcg{f};
exprmin::DBFGS<decltype(f)>   dbfgs{f};

// Powell §10.6 — no derivatives, conjugate-direction
exprmin::Powell pw{f};
auto p4 = pw.minimize({0.0, 0.0});

// Nelder–Mead §10.5 — derivative-free simplex
exprmin::Amoeba am{f};
auto p5 = am.minimize({0.0, 0.0}, /*delta=*/1.0);
```

### Trust-region dogleg

```cpp
auto x = PV(0.0, 'x');
auto y = PV(0.0, 'y');
auto f = (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);

// Default: BFGS Hessian approximation (rank-2 updates, no second derivatives)
exprmin::Dogleg dl{f};
auto p = dl.minimize({0.0, 0.0});   // p ≈ {1.0, 2.0}
dl.get_optimal_value();              // f at the returned minimum

// ExactAD: full Hessian recomputed every iteration via forward-over-reverse AD
auto dl_exact = exprmin::make_dogleg<exprmin::HessianMode::ExactAD>(f);
auto p2 = dl_exact.minimize({0.0, 0.0});
```

Each iteration picks the longest safe step from three candidates — full Newton
(inside trust region), Cauchy (gradient descent to boundary), or the dogleg
interpolation between them. The trust-region radius is adjusted using the
Powell/libdogleg ρ-ratio policy.

| `HessianMode` | Description |
|---|---|
| `BFGS` (default) | Rank-2 BFGS updates from gradient differences — O(N²) storage, no second derivatives |
| `ExactAD` | Exact Hessian via `diff::derivative_tensor<2>` each iteration — fewer iterations, costs N forward passes |

### Simulated annealing §10.12

```cpp
auto x = PV(0.0, 'x');
auto y = PV(0.0, 'y');
auto f = (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);

// SimAnneal(expr, T0, cooling, epoch_steps)
exprmin::SimAnneal sa{f, 1.0, 0.95, 100};
auto p = sa.minimize({3.0, 0.0}, /*delta=*/1.0);
sa.get_optimal_value();   // f at the returned minimum
```

### Nonlinear least squares (Levenberg–Marquardt §15.5 / Gauss–Newton)

```cpp
auto a = PV(0.0, 'a');
auto b = PV(0.0, 'b');
auto x = PV(0.0, 'x');
auto model = a * exp(-b * x);

// 'x' is the input variable; 'a','b' are inferred as parameters
auto lm = exprmin::make_lm<'x'>(model);   // LM — Marquardt damping
auto gn = exprmin::make_gn<'x'>(model);   // Gauss-Newton — undamped

std::vector<decltype(lm)::DataPoint> data = { /* {InputVec, y_obs, weight} */ };
auto params  = lm.fit(decltype(lm)::ParamVec{1.0, 1.0}, data);
auto params2 = gn.fit(decltype(gn)::ParamVec{1.0, 1.0}, data);
```

`LevenbergMarquardt` adds Marquardt damping (λ·diag(JᵀJ)) and is robust to
poor initial guesses. `GaussNewton` solves the undamped normal equations and
converges quadratically near the solution when the residual is small.

### NLS trust region (NLSDogleg / Subspace2D)

Both minimize ½‖R(θ)‖² where R : ℝᴺ → ℝᴹ (M ≥ N) is given as
`diff::Equation<R1,...,RM>`. The Gauss-Newton Hessian B = JᵀJ is recomputed
from the exact AD Jacobian at each accepted step.

```cpp
auto x = PV(0.0, 'x');
auto y = PV(0.0, 'y');
auto r1 = x * x + y - 3.0;
auto r2 = x + y * y - 3.0;

// NLSDogleg — classical Powell dogleg (Standard variant, default)
auto nd = exprmin::make_nls_dogleg(r1, r2);
auto p = nd.minimize({2.0, 0.0});
nd.get_optimal_value();   // ½‖r‖² at returned point

// Double dogleg — scales GN step by t ∈ [0.2, 1] before interpolation
auto nd2 = exprmin::make_nls_dogleg<exprmin::DoglegVariant::Double>(r1, r2);

// Subspace2D — minimizes the quadratic model over span{Cauchy, GN} ∩ TR ball
auto s2 = exprmin::make_subspace2d(r1, r2);
auto p2 = s2.minimize({2.0, 0.0});
s2.get_optimal_value();
```

| Class | Step selection | Notes |
|---|---|---|
| `NLSDogleg<..., Standard>` | Classical 3-case Powell dogleg | Cauchy / GN / dogleg interpolation |
| `NLSDogleg<..., Double>` | Byrd–Schnabel–Shultz double dogleg | Scales GN by t ∈ [0.2,1] before interpolation |
| `Subspace2D` | 2D subspace optimal step | Solves quartic for λ via companion-matrix eigenvalues |

### Constrained minimization (Augmented Lagrangian)

```cpp
auto x = PV(0.0, 'x');
auto y = PV(0.0, 'y');
auto f = x * x + y * y;
auto h = x + y - 1.0;   // equality:   x + y = 1
auto g = -x;             // inequality: x ≥ 0

exprmin::AugLag al{f, exprmin::make_eq(h), exprmin::make_ineq(g)};
auto p = al.minimize({0.5, 0.5});
```

The outer loop follows Birgin & Martínez Algorithm 3.1: auto-scaled initial ρ,
BFGS+Armijo inner solves, dual updates for λ (equality) and μ (inequality),
and adaptive penalty growth when constraint progress stalls.

### Linear programming

```cpp
// SimplexLP — two-phase tableau simplex (inequality form: Ax ≤ b, x ≥ 0)
exprmin::SimplexLP<> lp;
auto x = lp.solve(A, b, c);   // min cᵀx s.t. Ax ≤ b, x ≥ 0
// lp.status == SimplexLP<>::Status::Optimal

// InteriorPointLP — primal-dual interior-point (standard form: Ax = b, x ≥ 0)
exprmin::InteriorPointLP<> ip;
auto x2 = ip.solve(A_eq, b_eq, c_eq);
// ip.status == InteriorPointLP<>::Status::Optimal
```

Both solvers set a `status` member (`Optimal`, `Infeasible`, `Unbounded`,
`MaxIter`) and store the objective value in `fret` after each `solve()`.

### Broyden root finding

```cpp
auto x = PV(0.0, 'x');
auto y = PV(0.0, 'y');
auto f1 = x * x + y * y - 4.0;   // x² + y² = 4
auto f2 = x - y;                   // x = y  →  root: {√2, √2}

exprmin::Broyden br{diff::Equation{f1, f2}};
auto p = br.find_root({1.0, 0.5});   // p ≈ {√2, √2}
// br.residual_norm() < tol
```

Uses the exact reverse-mode AD Jacobian (instead of finite differences) for the
initial H₀ = −J(x₀)⁻¹ and for fallback refreshes after a failed line search.
Each iteration applies a rank-1 Broyden update to H ≈ −J⁻¹, with Hebden
backtracking on ‖F‖.

## Factory helpers

`make.hpp` (included via `minimizer.hpp`) provides factory functions that deduce
all template parameters from their arguments — useful when CTAD alone is
insufficient (non-default policies, `HessianMode`, CG method, LBFGS history
size, NLS residual packing, or the curve-fitting symbol partition).

| Factory | Equivalent |
|---|---|
| `make_nls_dogleg(r1, r2, ...)` | `NLSDogleg<Equation<...>>` |
| `make_nls_dogleg<DoglegVariant::Double>(r1, ...)` | `NLSDogleg<Equation<...>, Double>` |
| `make_subspace2d(r1, r2, ...)` | `Subspace2D<Equation<...>>` |
| `make_lm<'x'>(model)` | `LevenbergMarquardt<Expr, ParamSyms, InputSyms>` |
| `make_gn<'x'>(model)` | `GaussNewton<Expr, ParamSyms, InputSyms>` |
| `make_lbfgs<Armijo>(f)` | `LBFGS<Expr, Armijo>` |
| `make_lbfgs<Brent, 5>(f)` | `LBFGS<Expr, Brent, 5>` |
| `make_frprmn<CGMethod::FletcherReeves>(f)` | `Frprmn<Expr, FletcherReeves>` |
| `make_dogleg<HessianMode::ExactAD>(f)` | `Dogleg<Expr, ExactAD>` |

## Extending

See **[extending.md](extending.md)** for a guide on:

- Writing **custom callbacks** (CRTP `CallbackBase`, `CompositeCallbacks`)
- Implementing **custom solvers** by deriving from `QuasiNewtonBase`,
  `TrustRegionBase`, `NLSTrustRegionBase`, or `LeastSquaresBase`
- Tuning **algorithm policies** — CG method, Hessian mode, L-BFGS history
  depth, NLS dogleg variant — without subclassing
- Setting up **constrained optimization** with `AugLag` and `make_eq` /
  `make_ineq`
- Using **Broyden root finding** for square nonlinear systems

## License

Distributed under the [Boost Software License 1.0](LICENSE.txt).
