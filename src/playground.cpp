#include <nlohmann/json.hpp>

#include "../minimizer.hpp"
#include "callback/callback.hpp"
#include "expression_differentiator.hpp"

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <span>
#include <string>

using json = nlohmann::json;

struct PathCapture : exprmin::CallbackBase<PathCapture> {
  json *pts; // non-owning pointer to a json array
  explicit PathCapture(json &arr) : pts(&arr) {}
  void on_iter_point(int /*iter*/, std::span<const double> x) noexcept {
    if (x.size() >= 2)
      pts->push_back({x[0], x[1]});
  }
};

template <typename Algo>
json make_grid(Algo &algo, double xlo, double xhi, int nx, double ylo,
               double yhi, int ny) {
  using Point = typename Algo::Point;
  json z = json::array();
  for (int j = 0; j < ny; ++j) {
    const double yj = ylo + j * (yhi - ylo) / (ny - 1);
    for (int i = 0; i < nx; ++i) {
      const double xi = xlo + i * (xhi - xlo) / (nx - 1);
      z.push_back(algo.eval_at(Point{xi, yj}));
    }
  }
  return {{"xlo", xlo}, {"xhi", xhi}, {"nx", nx},         {"ylo", ylo},
          {"yhi", yhi}, {"ny", ny},   {"z", std::move(z)}};
}
template <typename URBG>
Eigen::Vector2d random_point(double xlo, double xhi, double ylo, double yhi,
                             URBG &rng) {
  return {std::uniform_real_distribution<double>(xlo, xhi)(rng),
          std::uniform_real_distribution<double>(ylo, yhi)(rng)};
}
// =============================================================================
//  PLAYGROUND — change the expression and starting point, then:
//
//    cmake --build build --target playground && ./build/playground
//    python3 scripts/plot_paths.py build/optimization_data.json
// =============================================================================
int main() {
  // ── 1. Define the objective function ─────────────────────────────────────
  // Himmelblau: f(x,y) = (x²+y−11)² + (x+y²−7)²
  // Four equal minima at: (3,2), (−2.805,3.131), (−3.779,−3.283),
  // (3.584,−1.848). From p0=(0,0) each algorithm races a different route to its
  // nearest minimum, making path geometry differences visually obvious.
  //
  // Swap in any other expression here — bounds, start, and title are all you
  // need to change.  The grid is evaluated by C++; Python plots blindly.
  const std::string title = "Himmelblau";

  std::mt19937 rng(std::random_device{}());
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto f = (x * x + y - 11.0) * (x * x + y - 11.0) +
           (x + y * y - 7.0) * (x + y * y - 7.0);

  // ── Other expressions to try — just swap the block below ─────────────────
  //
  // Rosenbrock (banana valley)
  //   global min f=0 at (1,1); notoriously difficult curved valley
  //   bounds: x∈[-2,2], y∈[-1,3], p0=(−1,1)
  //
  //   const std::string title = "Rosenbrock";
  //   auto f = (1.0 - x)*(1.0 - x) + 100.0*(y - x*x)*(y - x*x);
  //   constexpr double xlo=-2, xhi=2, ylo=-1, yhi=3;
  //
  // Beale
  //   global min f=0 at (3, 0.5); sharp ridges at boundary
  //   bounds: x∈[-4.5,4.5], y∈[-4.5,4.5], p0=(1,1)
  //
  //   const std::string title = "Beale";
  //   auto f = (1.5   - x + x*y)*(1.5   - x + x*y)
  //          + (2.25  - x + x*y*y)*(2.25  - x + x*y*y)
  //          + (2.625 - x + x*y*y*y)*(2.625 - x + x*y*y*y);
  //   constexpr double xlo=-4.5, xhi=4.5, ylo=-4.5, yhi=4.5;
  //
  // Booth
  //   global min f=0 at (1,3); smooth single-minimum quadratic
  //   bounds: x∈[-10,10], y∈[-10,10], p0=(0,0)
  //
  //   const std::string title = "Booth";
  //   auto f = (x + 2.0*y - 7.0)*(x + 2.0*y - 7.0)
  //          + (2.0*x + y - 5.0)*(2.0*x + y - 5.0);
  //   constexpr double xlo=-10, xhi=10, ylo=-10, yhi=10;
  //
  // Matyas
  //   global min f=0 at (0,0); shallow elliptical bowl
  //   bounds: x∈[-10,10], y∈[-10,10], p0=(5,5)
  //
  //   const std::string title = "Matyas";
  //   auto f = 0.26*(x*x + y*y) - 0.48*x*y;
  //   constexpr double xlo=-10, xhi=10, ylo=-10, yhi=10;
  //
  // Six-hump camelback
  //   two global minima f≈−1.0316 at ±(0.0898,−0.7126); six local minima
  //   bounds: x∈[-3,3], y∈[-2,2], p0=(−2,1)
  //
  //   const std::string title = "Six-hump camelback";
  //   auto f = (4.0 - 2.1*x*x + x*x*x*x/3.0)*x*x
  //          + x*y
  //          + (-4.0 + 4.0*y*y)*y*y;
  //   constexpr double xlo=-3, xhi=3, ylo=-2, yhi=2;
  //
  // McCormick  (requires sin — use `using std::sin;` in scope or rely on ADL)
  //   global min f≈−1.9133 at (−0.5472,−1.5472)
  //   bounds: x∈[-1.5,4], y∈[-3,4], p0=(0,0)
  //
  //   const std::string title = "McCormick";
  //   auto f = sin(x + y) + (x - y)*(x - y) - 1.5*x + 2.5*y + 1.0;
  //   constexpr double xlo=-1.5, xhi=4, ylo=-3, yhi=4;
  //
  // Easom  (requires cos and exp)
  //   global min f=−1 at (π,π); extremely sharp spike, easy for gradient methods
  //   bounds: x∈[-10,10], y∈[-10,10], p0=(2,2)
  //
  //   const std::string title = "Easom";
  //   auto pi = std::numbers::pi;
  //   auto f = -cos(x)*cos(y)*exp(-(x - pi)*(x - pi) - (y - pi)*(y - pi));
  //   constexpr double xlo=-10, xhi=10, ylo=-10, yhi=10;
  //
  // Goldstein–Price
  //   global min f=3 at (0,−1); four local minima
  //   bounds: x∈[-2,2], y∈[-2,2], p0=(0,−1.5)
  //
  //   const std::string title = "Goldstein-Price";
  //   auto A = 1.0 + (x + y + 1.0)*(x + y + 1.0)
  //                *(19.0 - 14.0*x + 3.0*x*x - 14.0*y + 6.0*x*y + 3.0*y*y);
  //   auto B = 30.0 + (2.0*x - 3.0*y)*(2.0*x - 3.0*y)
  //                 *(18.0 - 32.0*x + 12.0*x*x + 48.0*y - 36.0*x*y + 27.0*y*y);
  //   auto f = A * B;
  //   constexpr double xlo=-2, xhi=2, ylo=-2, yhi=2;
  //
  // ── 2. Grid bounds (for the contour plot) and starting point ─────────────
  constexpr double xlo = -5.0, xhi = 5.0;
  constexpr double ylo = -5.0, yhi = 5.0;
  constexpr int nx = 250, ny = 250;

  // const Eigen::Vector2d p0{0.0, 0.0}; // equidistant from all four minima
  auto p0 = random_point(xlo, xhi, ylo, yhi, rng);
  // ── 3. Evaluate grid (once — used for the Python contour) ────────────────
  std::printf("=== %s  start=(%.2f, %.2f) ===\n\n", title.c_str(), p0[0],
              p0[1]);

  auto grid_eval = exprmin::make_lbfgs(f); // plain solver, no callback
  json root;
  root["title"] = title;
  root["grid"] = make_grid(grid_eval, xlo, xhi, nx, ylo, yhi, ny);
  root["start"] = {p0[0], p0[1]};
  root["paths"] = json::array();

  // ── 4. Run algorithms and record paths ────────────────────────────────────
  auto record = [&](const std::string &name, auto run_fn) {
    json pts = json::array();
    pts.push_back({p0[0], p0[1]}); // all paths start from the same p0
    auto [r, fopt] = run_fn(PathCapture{pts});
    std::printf("%-14s [%.6f, %.6f]  f=%.2e  iters=%zu\n", (name + ":").c_str(),
                r[0], r[1], fopt, pts.size());
    p0 = random_point(xlo, xhi, ylo, yhi, rng);
    root["paths"].push_back({
        {"name", name},
        {"start", {p0[0], p0[1]}},
        {"final", {r[0], r[1]}},
        {"points", std::move(pts)},
    });
  };

  record("L-BFGS", [&](auto cb) {
    auto s = exprmin::make_lbfgs<exprmin::Brent, 10>(f, std::move(cb));
    auto r = s.minimize(p0);
    return std::pair{r, s.get_optimal_value()};
  });

  record("BFGS/Armijo", [&](auto cb) {
    auto s = exprmin::make_lbfgs<exprmin::Armijo, 200>(f, std::move(cb));
    auto r = s.minimize(p0);
    return std::pair{r, s.get_optimal_value()};
  });

  record("Dogleg", [&](auto cb) {
    auto s = exprmin::make_dogleg(f, std::move(cb));
    auto r = s.minimize(p0);
    return std::pair{r, s.get_optimal_value()};
  });

  record("Amoeba", [&](auto cb) {
    exprmin::Amoeba s{f, 3e-8, std::move(cb)};
    auto r = s.minimize(p0, 0.5);
    return std::pair{r, s.get_optimal_value()};
  });

  // ── 5. Write JSON data file ───────────────────────────────────────────────
  constexpr const char *out_path = "optimization_data.json";
  std::ofstream(out_path) << root.dump(2);
  std::printf("\nWrote  %s\n", out_path);
  std::printf("Plot:  python3 scripts/plot_paths.py %s\n", out_path);

  return 0;
}
