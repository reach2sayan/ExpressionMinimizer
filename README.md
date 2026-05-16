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
| `LevenbergMarquardt` | §15.5 | Nonlinear least-squares fitting |
| `AugLag` | — | Augmented Lagrangian constrained minimization |

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

### Nonlinear least-squares (Levenberg-Marquardt §15.5)

```cpp
auto a = diff::Variable<double, 'a'>{0.0};
auto b = diff::Variable<double, 'b'>{0.0};
auto x = diff::Variable<double, 'x'>{0.0};
auto model = a * exp(-b * x);

using ParamSyms = boost::mp11::mp_list<std::integral_constant<char,'a'>,
                                       std::integral_constant<char,'b'>>;
using InputSyms = boost::mp11::mp_list<std::integral_constant<char,'x'>>;

exprmin::LevenbergMarquardt<decltype(model), ParamSyms, InputSyms> lm{model};

std::vector<decltype(lm)::DataPoint> data = { /* {InputVec, y_obs, weight} */ };
auto params = lm.fit(decltype(lm)::ParamVec{1.0, 1.0}, data);
```

### Simulated annealing §10.12

```cpp
// SimAnneal(expr, T0, cooling, epoch_steps)
exprmin::SimAnneal sa{f, 1.0, 0.95, 100};
auto p = sa.minimize({3.0, 0.0}, /*delta=*/1.0);
// sa.fret holds f at the returned minimum
```

## License

Distributed under the [Boost Software License 1.0](LICENSE.txt).
