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
