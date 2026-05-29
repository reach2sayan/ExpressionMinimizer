#include <nlohmann/json.hpp>

#include "../minimizer.hpp"
#include "callback/callback.hpp"
#include "expression_differentiator.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <span>
#include <string>

using json = nlohmann::json;

struct PathCapture : exprmin::CallbackBase<PathCapture> {
  json *pts;
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

// slug: "Some Title" -> "some_title"
static std::string to_slug(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::replace(s.begin(), s.end(), ' ', '_');
  std::replace(s.begin(), s.end(), '-', '_');
  return s;
}

// =============================================================================
//  PLAYGROUND — cycles through all supported 2-D benchmark functions,
//  starting from a fresh random point each run, and emits one PNG per function.
//
//    cmake --build build --target playground && ./build/playground
// =============================================================================
int main() {
  constexpr int nx = 250, ny = 250;
  std::mt19937 rng(std::random_device{}());

  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');

  // run_func: evaluate grid, run all algos from p0, write JSON, call Python.
  auto run_func = [&](const std::string &title, auto f, double xlo, double xhi,
                      double ylo, double yhi, Eigen::Vector2d p0) {
    std::printf("\n=== %s  start=(%.3f, %.3f) ===\n", title.c_str(), p0[0],
                p0[1]);

    auto grid_eval = exprmin::make_lbfgs(f);
    json root;
    root["title"] = title;
    root["grid"] = make_grid(grid_eval, xlo, xhi, nx, ylo, yhi, ny);
    root["start"] = {p0[0], p0[1]};
    root["paths"] = json::array();

    auto record = [&](const std::string &name, auto run_fn) {
      json pts = json::array();
      pts.push_back({p0[0], p0[1]});
      auto [r, fopt] = run_fn(PathCapture{pts});
      std::printf("  %-14s [%.6f, %.6f]  f=%.2e  iters=%zu\n",
                  (name + ":").c_str(), r[0], r[1], fopt, pts.size());
      root["paths"].push_back({{"name", name},
                                {"start", {p0[0], p0[1]}},
                                {"final", {r[0], r[1]}},
                                {"points", std::move(pts)}});
    };

    record("L-BFGS", [&](auto cb) {
      auto s = exprmin::make_lbfgs<exprmin::Brent, 10>(f, std::move(cb));
      return std::pair{s.minimize(p0), s.get_optimal_value()};
    });
    record("BFGS/Armijo", [&](auto cb) {
      auto s = exprmin::make_lbfgs<exprmin::Armijo, 200>(f, std::move(cb));
      return std::pair{s.minimize(p0), s.get_optimal_value()};
    });
    record("Dogleg", [&](auto cb) {
      auto s = exprmin::make_dogleg(f, std::move(cb));
      return std::pair{s.minimize(p0), s.get_optimal_value()};
    });
    record("Amoeba", [&](auto cb) {
      exprmin::Amoeba s{f, 3e-8, std::move(cb)};
      return std::pair{s.minimize(p0, 0.5), s.get_optimal_value()};
    });

    const std::string json_file = to_slug(title) + "_data.json";
    std::ofstream(json_file) << root.dump(2);
    std::printf("  -> %s\n", json_file.c_str());

    const std::string cmd =
        "python3 scripts/plot_paths.py " + json_file;
    [[maybe_unused]] int rc = std::system(cmd.c_str());
  };

  // ── Himmelblau ───────────────────────────────────────────────────────────
  // Four equal minima at: (3,2), (−2.805,3.131), (−3.779,−3.283), (3.584,−1.848)
  run_func("Himmelblau",
           (x * x + y - 11.0) * (x * x + y - 11.0) +
               (x + y * y - 7.0) * (x + y * y - 7.0),
           -5, 5, -5, 5, random_point(-5, 5, -5, 5, rng));

  // ── Rosenbrock ───────────────────────────────────────────────────────────
  // Global min f=0 at (1,1); notoriously difficult curved valley
  run_func("Rosenbrock",
           (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x),
           -2, 2, -1, 3, random_point(-2, 2, -1, 3, rng));

  // ── Beale ────────────────────────────────────────────────────────────────
  // Global min f=0 at (3, 0.5); sharp ridges at boundary
  run_func("Beale",
           (1.5 - x + x * y) * (1.5 - x + x * y) +
               (2.25 - x + x * y * y) * (2.25 - x + x * y * y) +
               (2.625 - x + x * y * y * y) * (2.625 - x + x * y * y * y),
           -4.5, 4.5, -4.5, 4.5, random_point(-4.5, 4.5, -4.5, 4.5, rng));

  // ── Booth ────────────────────────────────────────────────────────────────
  // Global min f=0 at (1,3); smooth single-minimum quadratic
  run_func("Booth",
           (x + 2.0 * y - 7.0) * (x + 2.0 * y - 7.0) +
               (2.0 * x + y - 5.0) * (2.0 * x + y - 5.0),
           -10, 10, -10, 10, random_point(-10, 10, -10, 10, rng));

  // ── Matyas ───────────────────────────────────────────────────────────────
  // Global min f=0 at (0,0); shallow elliptical bowl
  run_func("Matyas", 0.26 * (x * x + y * y) - 0.48 * x * y, -10, 10, -10, 10,
           random_point(-10, 10, -10, 10, rng));

  // ── Six-hump camelback ───────────────────────────────────────────────────
  // Two global minima f≈−1.0316 at ±(0.0898,−0.7126); six local minima
  run_func("Six-hump camelback",
           (4.0 - 2.1 * x * x + x * x * x * x / 3.0) * x * x + x * y +
               (-4.0 + 4.0 * y * y) * y * y,
           -3, 3, -2, 2, random_point(-3, 3, -2, 2, rng));

  // ── Goldstein-Price ──────────────────────────────────────────────────────
  // Global min f=3 at (0,−1); four local minima
  {
    auto A = 1.0 + (x + y + 1.0) * (x + y + 1.0) *
                       (19.0 - 14.0 * x + 3.0 * x * x - 14.0 * y +
                        6.0 * x * y + 3.0 * y * y);
    auto B = 30.0 + (2.0 * x - 3.0 * y) * (2.0 * x - 3.0 * y) *
                        (18.0 - 32.0 * x + 12.0 * x * x + 48.0 * y -
                         36.0 * x * y + 27.0 * y * y);
    run_func("Goldstein-Price", A * B, -2, 2, -2, 2,
             random_point(-2, 2, -2, 2, rng));
  }

  return 0;
}
