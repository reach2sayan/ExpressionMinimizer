# ExpressionMinimizer

A header-only C++23 library of numerical minimization algorithms from
*Numerical Recipes* (NR) Ch. 10 & 15, wired to the
[ExpressionDifferentiator](https://github.com/reach2sayan/ExpressionDifferentiator)
expression-template engine. Gradients and Jacobians are obtained for free via
reverse-mode automatic differentiation — no hand-coded derivatives required.

## Algorithms

| Class | NR ref | Method |
|---|---|---|
| `Bracketmethod` | §10.1 | Downhill bracket search |
| `Golden` | §10.2 | Golden-section search |
| `Brent` | §10.3 | Parabolic interpolation + golden fallback |
| `Dbrent` | §10.4 | Brent with derivative (secant on f′ via reverse AD) |
| `LinMin` | §10.5 | Line minimization along a direction (uses `Brent`) |
| `DLinMin` | §10.5 | Line minimization with derivative (uses `Dbrent`) |
| `Powell` | §10.5 | Powell's conjugate-direction method |
| `Frprmn` | §10.6 | Fletcher-Reeves / Polak-Ribière conjugate gradient |
| `DFrprmn` | §10.6 | Conjugate gradient with `DLinMin` line search |
| `BFGS` | §10.7 | Quasi-Newton with full N×N inverse-Hessian |
| `DBFGS` | §10.7 | BFGS with derivative line search (`Dbrent`) |
| `ABFGS` | §10.7 | BFGS with Armijo backtracking line search |
| `LBFGS` | Nocedal §7.2 | Limited-memory BFGS (two-loop, circular (s,y) buffer) |
| `Amoeba` | §10.4 | Nelder-Mead downhill simplex |
| `SimAnneal` | §10.12 | Simulated annealing + Amoeba cold refinement |
| `Dogleg` | Powell (1970) | Trust-region dogleg with BFGS Hessian approximation |
| `NLSDogleg` | Powell (1970) | NLS trust-region Powell dogleg (classical + double variant) for ½‖R(θ)‖² |
| `Subspace2D` | Kaufman (1999) | NLS trust-region 2D subspace step (quartic Lagrange multiplier) for ½‖R(θ)‖² |
| `LevenbergMarquardt` | §15.5 | Nonlinear least-squares fitting with Marquardt damping |
| `GaussNewton` | — | Nonlinear least-squares (undamped normal equations) |
| `AugLag` | — | Augmented Lagrangian constrained minimization |
| `Broyden` | Broyden (1965) | Rank-1 quasi-Newton root finder for F(x) = 0 |

All multi-dimensional optimizers infer the problem dimensionality at compile
time from the set of `Variable` symbols in the expression tree.

## Line search policies

`BFGS` and `LBFGS` accept a line search policy as a template-template
parameter:

| Policy | Description |
|---|---|
| `Brent` (default) | Bracket + parabolic/golden-section 1D minimization |
| `Dbrent` | Brent with first-derivative (secant on f′) |
| `Armijo` | Backtracking sufficient-decrease (no bracketing required) |

```cpp
exprmin::BFGS<decltype(f)>                     bfgs{f};   // Brent (default)
exprmin::BFGS<decltype(f), exprmin::Dbrent>    dbfgs{f};  // == DBFGS
exprmin::BFGS<decltype(f), exprmin::Armijo>    abfgs{f};  // == ABFGS
exprmin::LBFGS<decltype(f)>                    lbfgs{f};  // Brent (default)
exprmin::LBFGS<decltype(f), exprmin::Armijo>   albfgs{f}; // Armijo
exprmin::LBFGS<decltype(f), exprmin::Brent, 5> lbfgs5{f}; // M=5 history pairs
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

All headers are included via the umbrella header:

```cpp
#include "minimizer/minimizer.hpp"
#include "expression_differentiator.hpp"
```

### 1D minimization

```cpp
auto x = diff::Variable<double, 'x'>{0.0};
auto f = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0});

// Brent §10.3 — no derivatives needed
exprmin::Brent b{f};
double xmin = b.minimize(0.0, 5.0); // bracket from [0, 5], then minimize
// b.xmin == 3.0,  b.fmin == 0.0

// Dbrent §10.4 — uses f′ from reverse-mode AD for faster convergence
exprmin::Dbrent db{f};
double xmin2 = db.minimize(0.0, 5.0);
```

### N-dimensional minimization

```cpp
auto x = diff::Variable<double, 'x'>{0.0};
auto y = diff::Variable<double, 'y'>{0.0};
auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
       + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

// BFGS §10.7 — quasi-Newton, full inverse-Hessian approximation
exprmin::BFGS bfgs{f};
auto p = bfgs.minimize({0.0, 0.0}); // p ≈ {1.0, 2.0}

// L-BFGS — same interface, O(M·N) memory instead of O(N²)
exprmin::LBFGS lbfgs{f};
auto p2 = lbfgs.minimize({0.0, 0.0});

// Conjugate gradient — Polak-Ribière (default) or Fletcher-Reeves
exprmin::Frprmn cg{f};
auto p3 = cg.minimize({0.0, 0.0});

exprmin::Frprmn<decltype(f), exprmin::CGMethod::FletcherReeves> cg_fr{f};

// Derivative-aware variants use Dbrent line search
exprmin::DFrprmn<decltype(f)> dcg{f};
exprmin::DBFGS<decltype(f)>   dbfgs{f};

// Powell §10.5 — no derivatives, conjugate-direction
exprmin::Powell pw{f};
auto p4 = pw.minimize({0.0, 0.0});
```

### Trust-region dogleg

```cpp
auto x = diff::Variable<double, 'x'>{0.0};
auto y = diff::Variable<double, 'y'>{0.0};
auto f = (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);

// Default: BFGS Hessian approximation (rank-2 updates, no second derivatives)
exprmin::Dogleg dl{f};
auto p = dl.minimize({0.0, 0.0}); // p ≈ {1.0, 2.0}
dl.get_optimal_value();            // f at the returned minimum

// ExactAD: full Hessian recomputed every iteration via forward-over-reverse AD
exprmin::Dogleg<decltype(f), exprmin::HessianMode::ExactAD> dl_exact{f};
auto p2 = dl_exact.minimize({0.0, 0.0});
```

Each iteration picks the longest safe step from three candidates — full Newton
(inside trust region), Cauchy (gradient descent to boundary), or the dogleg
interpolation between them. The trust-region radius is adjusted using the
Powell (libdogleg) ρ-ratio policy.

A compile-time `HessianMode` template parameter controls how the Hessian is supplied:

| Mode | Description |
|---|---|
| `HessianMode::BFGS` (default) | Rank-2 BFGS updates from gradient differences — O(N²) storage, no second derivatives |
| `HessianMode::ExactAD` | Exact Hessian via `diff::derivative_tensor<2>` each iteration — fewer iterations, costs N forward passes |

### Nonlinear least-squares (Gauss-Newton)

```cpp
auto a = diff::Variable<double, 'a'>{0.0};
auto b = diff::Variable<double, 'b'>{0.0};
auto x = diff::Variable<double, 'x'>{0.0};
auto model = a * exp(-b * x);

using ParamSyms = boost::mp11::mp_list<std::integral_constant<char,'a'>,
                                       std::integral_constant<char,'b'>>;
using InputSyms = boost::mp11::mp_list<std::integral_constant<char,'x'>>;

exprmin::GaussNewton<decltype(model), ParamSyms, InputSyms> gn{model};

std::vector<decltype(gn)::DataPoint> data = { /* {InputVec, y_obs, weight} */ };
auto params = gn.fit(decltype(gn)::ParamVec{1.0, 1.0}, data);
```

Solves the undamped normal equations `(JᵀJ)·step = Jᵀr` each iteration.
Converges quadratically near the solution; prefer `LevenbergMarquardt` when
the initial guess may be far from the optimum.

### Constrained minimization (Augmented Lagrangian)

```cpp
auto x = diff::Variable<double, 'x'>{0.0};
auto y = diff::Variable<double, 'y'>{0.0};
auto f  = x * x + y * y;
auto h  = x + y - diff::Constant<double>{1.0};  // equality: x + y = 1
auto g  = -x;                                    // inequality: x ≥ 0

using EqC   = diff::Equation<decltype(h)>;
using IneqC = diff::Equation<decltype(g)>;

exprmin::AugLag<decltype(f), EqC, IneqC> al{f, EqC{h}, IneqC{g}};
auto p = al.minimize({0.5, 0.5});
```

### Nonlinear least-squares (Levenberg-Marquardt §15.5 / Gauss-Newton)

```cpp
auto a = diff::Variable<double, 'a'>{0.0};
auto b = diff::Variable<double, 'b'>{0.0};
auto x = diff::Variable<double, 'x'>{0.0};
auto model = a * exp(-b * x);

using ParamSyms = boost::mp11::mp_list<std::integral_constant<char,'a'>,
                                       std::integral_constant<char,'b'>>;
using InputSyms = boost::mp11::mp_list<std::integral_constant<char,'x'>>;

// LM — Marquardt damping, robust to poor initial guesses
exprmin::LevenbergMarquardt<decltype(model), ParamSyms, InputSyms> lm{model};

std::vector<decltype(lm)::DataPoint> data = { /* {InputVec, y_obs, weight} */ };
auto params = lm.fit(decltype(lm)::ParamVec{1.0, 1.0}, data);

// Gauss-Newton — undamped, quadratic convergence near the solution
exprmin::GaussNewton<decltype(model), ParamSyms, InputSyms> gn{model};
auto params2 = gn.fit(decltype(gn)::ParamVec{1.0, 1.0}, data);
```

### Simulated annealing §10.12

```cpp
// SimAnneal(expr, T0, cooling, epoch_steps)
exprmin::SimAnneal sa{f, 1.0, 0.95, 100};
auto p = sa.minimize({3.0, 0.0}, /*delta=*/1.0);
sa.get_optimal_value(); // f at the returned minimum
```

### Broyden root finding

```cpp
auto x = diff::Variable<double, 'x'>{0.0};
auto y = diff::Variable<double, 'y'>{0.0};
auto f1 = x * x + y * y - 4.0;  // x² + y² = 4
auto f2 = x - y;                  // x = y  →  root: {√2, √2}

exprmin::Broyden br{diff::Equation{f1, f2}};
auto p = br.find_root({1.0, 0.5}); // p ≈ {√2, √2}
// br.residual_norm() < tol
```

Uses exact reverse-mode AD Jacobian (instead of finite differences as in GSL) for initialization and the fallback Jacobian recomputation. Each iteration applies a rank-1 Broyden update to the approximate inverse Jacobian H ≈ −J⁻¹, with Hebden backtracking on ‖F‖.

### NLS trust-region (NLSDogleg / Subspace2D)

Both minimize ½‖R(θ)‖² where R : ℝᴺ → ℝᴹ (M ≥ N) is given as `diff::Equation<R1,...,RM>`.
The Gauss-Newton Hessian B = JᵀJ is recomputed from the exact AD Jacobian at each accepted step.

```cpp
auto x = PV(0.0, 'x');  auto y = PV(0.0, 'y');
auto r1 = x * x + y - 3.0;
auto r2 = x + y * y - 3.0;

// NLSDogleg — classical Powell dogleg (Standard variant, default)
exprmin::NLSDogleg<diff::Equation<decltype(r1), decltype(r2)>> nd{
    diff::Equation{r1, r2}};
auto p = nd.minimize({2.0, 0.0});
nd.get_optimal_value();  // ½‖r‖² at returned point

// Double dogleg variant (Dennis & Mei 1979) — scales GN step before interpolation
exprmin::NLSDogleg<diff::Equation<decltype(r1), decltype(r2)>,
                   exprmin::DoglegVariant::Double> nd2{diff::Equation{r1, r2}};

// Subspace2D — minimizes quadratic model over span{Cauchy, GN} ∩ TR ball
// Optimal subspace step found via degree-4 polynomial in Lagrange multiplier λ
exprmin::Subspace2D<diff::Equation<decltype(r1), decltype(r2)>> s2{
    diff::Equation{r1, r2}};
auto p2 = s2.minimize({2.0, 0.0});
s2.get_optimal_value();
```

| Class | Step selection | Notes |
|---|---|---|
| `NLSDogleg<..., DoglegVariant::Standard>` | Classical 3-case Powell dogleg | Cauchy / GN / dogleg interpolation |
| `NLSDogleg<..., DoglegVariant::Double>` | Dennis & Mei (1979) double dogleg | Scales GN by t before interpolation |
| `Subspace2D` | 2D subspace optimal step | Solves quartic for λ via companion-matrix eigenvalues |

## Factory helpers

`#include "minimizer/make.hpp"` provides factory functions that deduce all template parameters when CTAD alone is insufficient.

### NLS residual packing

```cpp
// Pack residuals without wrapping them in diff::Equation by hand.
auto nd  = exprmin::make_nls_dogleg(r1, r2);            // Standard variant
auto nd2 = exprmin::make_nls_dogleg<exprmin::DoglegVariant::Double>(r1, r2);
auto s2  = exprmin::make_subspace2d(r1, r2);
```

### Curve-fitting symbol partition

Specify the *input* variable characters as template arguments; parameter symbols are deduced as the complement.

```cpp
auto x = PV(0.0, 'x');
auto a = PV(0.0, 'a');
auto b = PV(0.0, 'b');
auto model = a * exp(-b * x);

// 'x' is the input; 'a','b' are inferred as parameters
auto lm = exprmin::make_lm<'x'>(model);
auto gn = exprmin::make_gn<'x'>(model);

std::vector<decltype(lm)::DataPoint> data = { /* {InputVec, y_obs, weight} */ };
auto params = lm.fit(decltype(lm)::ParamVec{1.0, 1.0}, data);
```

### Optimizer policy helpers

```cpp
// L-BFGS with explicit line-search policy and history size
auto lbfgs5 = exprmin::make_lbfgs<exprmin::Armijo, 5>(f);

// Conjugate gradient with explicit CG method and line-minimizer
auto cg = exprmin::make_frprmn<exprmin::CGMethod::FletcherReeves>(f);

// Dogleg with explicit Hessian mode
auto dl = exprmin::make_dogleg<exprmin::HessianMode::ExactAD>(f);
```

## License

Distributed under the [Boost Software License 1.0](LICENSE.txt).
