#include <gtest/gtest.h>
#include <cmath>
#include <numbers>

#include <Eigen/Dense>
#include "expression_differentiator.hpp"
#include "minimizer/minimizer.hpp"

// Tolerance matching NR's default for Golden (3e-8 * |xmin|)
static constexpr double kTol = 1e-5;

// ─────────────────────────────────────────────────────────────
// Bracketmethod tests
// ─────────────────────────────────────────────────────────────

TEST(Bracketmethod, QuadraticBrackets) {
    // f(x) = (x - 3)^2  — minimum at x = 3
    auto x   = diff::Variable<double, 'x'>{0.0};
    auto f   = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0});

    exprmin::Bracketmethod bm{f};
    bm.bracket(0.0, 1.0);

    // After bracketing, bx must be strictly between ax and cx with f(bx) < f(ax) and f(bx) < f(cx)
    EXPECT_LT(bm.fb, bm.fa);
    EXPECT_LT(bm.fb, bm.fc);
    // The minimum (x=3) must lie within [ax, cx]
    double lo = std::min(bm.ax, bm.cx);
    double hi = std::max(bm.ax, bm.cx);
    EXPECT_LE(lo, 3.0);
    EXPECT_GE(hi, 3.0);
}

TEST(Bracketmethod, SingleCycleConverges) {
    // f(x) = x^4 - 14x^3 + 60x^2 - 70x — multiple features; bracket from (0,1)
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = x * x * x * x
           - diff::Constant<double>{14.0} * x * x * x
           + diff::Constant<double>{60.0} * x * x
           - diff::Constant<double>{70.0} * x;

    exprmin::Bracketmethod bm{f};
    EXPECT_NO_THROW(bm.bracket(0.0, 1.0));
    EXPECT_LT(bm.fb, bm.fa);
    EXPECT_LT(bm.fb, bm.fc);
}

// ─────────────────────────────────────────────────────────────
// Golden section search tests
// ─────────────────────────────────────────────────────────────

TEST(Golden, QuadraticMinimum) {
    // f(x) = (x - 2)^2  =>  xmin = 2, fmin = 0
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{2.0}) * (x - diff::Constant<double>{2.0});

    exprmin::Golden golden{f};
    double xmin = golden.minimize(0.0, 5.0);

    EXPECT_NEAR(xmin,         2.0, kTol);
    EXPECT_NEAR(golden.xmin,  2.0, kTol);
    EXPECT_NEAR(golden.fmin,  0.0, kTol * kTol);
}

TEST(Golden, SineMinimum) {
    // sin(x) local minimum near 3π/2 ≈ 4.71238898.
    // bracket() is an unbounded search; set the triplet manually to confine
    // golden section to the [3, 6] bowl around the local minimum.
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = sin(y);

    exprmin::Golden g{f};
    g.ax = 3.0;
    g.bx = std::numbers::pi * 1.5;   // known minimum — satisfies f(bx) < f(ax,cx)
    g.cx = 6.0;
    g.fa = g.eval_at(g.ax);
    g.fb = g.eval_at(g.bx);
    g.fc = g.eval_at(g.cx);
    double xmin = g.minimize();

    EXPECT_NEAR(xmin, 3.0 * std::numbers::pi / 2.0, kTol);
    EXPECT_NEAR(g.fmin, -1.0, kTol);
}

TEST(Golden, NegativeQuadratic) {
    // f(x) = -(x - 1)^2 + 4  =>  this is a maximum, minimum is at the bracket edges.
    // Instead test f(x) = (x + 1)^2, minimum at x = -1
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x + diff::Constant<double>{1.0}) * (x + diff::Constant<double>{1.0});

    exprmin::Golden g{f};
    double xmin = g.minimize(-3.0, 2.0);

    EXPECT_NEAR(xmin,  -1.0, kTol);
    EXPECT_NEAR(g.fmin, 0.0, kTol * kTol);
}

TEST(Golden, CustomTolerance) {
    // Tighter tolerance: 1e-10
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{7.5}) * (x - diff::Constant<double>{7.5});

    exprmin::Golden g{f, 1.0e-10};
    double xmin = g.minimize(5.0, 10.0);

    EXPECT_NEAR(xmin, 7.5, 1e-7);
}

TEST(Golden, ManualBracketThenMinimize) {
    // Set bracket manually, then call minimize() without ax0/bx0
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{4.0}) * (x - diff::Constant<double>{4.0});

    exprmin::Golden g{f};
    g.bracket(2.0, 3.0);   // bracket first
    double xmin = g.minimize();  // then minimize

    EXPECT_NEAR(xmin, 4.0, kTol);
}

// ─────────────────────────────────────────────────────────────
// Brent tests
// ─────────────────────────────────────────────────────────────

TEST(Brent, QuadraticMinimum) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{2.0}) * (x - diff::Constant<double>{2.0});

    exprmin::Brent b{f};
    double xmin = b.minimize(0.0, 5.0);

    EXPECT_NEAR(xmin,   2.0, kTol);
    EXPECT_NEAR(b.fmin, 0.0, kTol * kTol);
}

TEST(Brent, SineMinimum) {
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = sin(y);

    exprmin::Brent b{f};
    b.ax = 3.0; b.bx = std::numbers::pi * 1.5; b.cx = 6.0;
    b.fa = b.eval_at(b.ax); b.fb = b.eval_at(b.bx); b.fc = b.eval_at(b.cx);
    double xmin = b.minimize();

    EXPECT_NEAR(xmin,   3.0 * std::numbers::pi / 2.0, kTol);
    EXPECT_NEAR(b.fmin, -1.0, kTol);
}

TEST(Brent, QuarticMinimum) {
    // f(x) = (x-1)^4  — flat near minimum, good stress test for parabolic interpolation
    auto x = diff::Variable<double, 'x'>{0.0};
    auto d = x - diff::Constant<double>{1.0};
    auto f = d * d * d * d;

    exprmin::Brent b{f};
    double xmin = b.minimize(0.0, 3.0);

    EXPECT_NEAR(xmin,   1.0, kTol);
    EXPECT_NEAR(b.fmin, 0.0, kTol * kTol);
}

// ─────────────────────────────────────────────────────────────
// LinMin tests
// ─────────────────────────────────────────────────────────────

TEST(LinMin, AxisDirection) {
    // f(x,y) = (x-3)^2 + (y-4)^2; start (0,0), dir (1,0)
    // minimum along x-axis at t=3 → p=(3,0)
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0})
           + (y - diff::Constant<double>{4.0}) * (y - diff::Constant<double>{4.0});

    exprmin::LinMin lm{f};
    Eigen::Vector2d p{0.0, 0.0};
    Eigen::Vector2d dir{1.0, 0.0};
    lm.minimize(p, dir);

    EXPECT_NEAR(p[0],   3.0, kTol);
    EXPECT_NEAR(p[1],   0.0, kTol);   // y unchanged
    EXPECT_NEAR(lm.fret, 16.0, kTol); // (3-3)^2 + (0-4)^2 = 16
}

TEST(LinMin, DiagonalDirection) {
    // f(x,y) = (x-3)^2 + (y-4)^2; start (0,0), dir (1,1)
    // minimise (t-3)^2 + (t-4)^2 → t=3.5 → p=(3.5, 3.5)
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0})
           + (y - diff::Constant<double>{4.0}) * (y - diff::Constant<double>{4.0});

    exprmin::LinMin lm{f};
    Eigen::Vector2d p{0.0, 0.0};
    Eigen::Vector2d dir{1.0, 1.0};
    lm.minimize(p, dir);

    EXPECT_NEAR(p[0],   3.5, kTol);
    EXPECT_NEAR(p[1],   3.5, kTol);
    EXPECT_NEAR(lm.fret, 0.5, kTol); // (3.5-3)^2 + (3.5-4)^2 = 0.5
}

TEST(LinMin, DirScaledByStep) {
    // dir should be multiplied by xmin after minimize
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0})
           + (y - diff::Constant<double>{4.0}) * (y - diff::Constant<double>{4.0});

    exprmin::LinMin lm{f};
    Eigen::Vector2d p{0.0, 0.0};
    Eigen::Vector2d dir{1.0, 1.0};
    lm.minimize(p, dir);

    // dir = xmin * original_dir; p = original_p + dir
    EXPECT_NEAR(dir[0], p[0], kTol);
    EXPECT_NEAR(dir[1], p[1], kTol);
}

// ─────────────────────────────────────────────────────────────
// Powell tests
// ─────────────────────────────────────────────────────────────

TEST(Powell, Bowl2D) {
    // f(x,y) = (x-1)^2 + (y-2)^2  — minimum at (1, 2)
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::Powell pw{f};
    auto p = pw.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],   1.0, kTol);
    EXPECT_NEAR(p[1],   2.0, kTol);
    EXPECT_NEAR(pw.fret, 0.0, kTol * kTol);
}

TEST(Powell, Rosenbrock) {
    // f(x,y) = (1-x)^2 + 100*(y-x^2)^2  — minimum at (1,1), fmin=0
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    exprmin::Powell pw{f, 1e-10};
    auto p = pw.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],    1.0, 1e-4);
    EXPECT_NEAR(p[1],    1.0, 1e-4);
    EXPECT_NEAR(pw.fret, 0.0, 1e-6);
}

// ─────────────────────────────────────────────────────────────
// Frprmn (Conjugate Gradient) tests
// ─────────────────────────────────────────────────────────────

TEST(Frprmn, Bowl2D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::Frprmn cg{f};
    auto p = cg.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],    1.0, kTol);
    EXPECT_NEAR(p[1],    2.0, kTol);
    EXPECT_NEAR(cg.fret, 0.0, kTol * kTol);
}

TEST(Frprmn, Rosenbrock) {
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    exprmin::Frprmn cg{f, 1e-10};
    auto p = cg.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],    1.0, 1e-4);
    EXPECT_NEAR(p[1],    1.0, 1e-4);
    EXPECT_NEAR(cg.fret, 0.0, 1e-6);
}

TEST(Frprmn, Quadratic3D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    exprmin::Frprmn cg{f};
    auto p = cg.minimize({3.0, 3.0, 3.0});

    EXPECT_NEAR(p[0],    0.0, kTol);
    EXPECT_NEAR(p[1],    0.0, kTol);
    EXPECT_NEAR(p[2],    0.0, kTol);
    EXPECT_NEAR(cg.fret, 0.0, kTol * kTol);
}

TEST(Frprmn, FletcherReeves) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::Frprmn<decltype(f), exprmin::CGMethod::FletcherReeves> cg{f};
    auto p = cg.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0], 1.0, kTol);
    EXPECT_NEAR(p[1], 2.0, kTol);
}

// ─────────────────────────────────────────────────────────────
// BFGS tests
// ─────────────────────────────────────────────────────────────

TEST(BFGS, Bowl2D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::BFGS bfgs{f};
    auto p = bfgs.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],      1.0, kTol);
    EXPECT_NEAR(p[1],      2.0, kTol);
    EXPECT_NEAR(bfgs.fret, 0.0, kTol * kTol);
}

TEST(BFGS, Rosenbrock) {
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    exprmin::BFGS bfgs{f, 1e-10};
    auto p = bfgs.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],      1.0, 1e-4);
    EXPECT_NEAR(p[1],      1.0, 1e-4);
    EXPECT_NEAR(bfgs.fret, 0.0, 1e-6);
}

TEST(BFGS, Quadratic3D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    exprmin::BFGS bfgs{f};
    auto p = bfgs.minimize({3.0, 3.0, 3.0});

    EXPECT_NEAR(p[0],      0.0, kTol);
    EXPECT_NEAR(p[1],      0.0, kTol);
    EXPECT_NEAR(p[2],      0.0, kTol);
    EXPECT_NEAR(bfgs.fret, 0.0, kTol * kTol);
}

// ─────────────────────────────────────────────────────────────
// Amoeba (Nelder-Mead) tests
// ─────────────────────────────────────────────────────────────

TEST(Amoeba, Bowl2D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::Amoeba am{f};
    // (0,0)+delta=1 puts vertices on the same f=5 contour → false convergence;
    // (2,0) avoids that degeneracy.
    auto p = am.minimize({2.0, 0.0}, 1.0);

    EXPECT_NEAR(p[0],    1.0, 1e-4);
    EXPECT_NEAR(p[1],    2.0, 1e-4);
    EXPECT_NEAR(am.fret, 0.0, 1e-6);
}

TEST(Amoeba, Rosenbrock) {
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    exprmin::Amoeba am{f, 1e-8};
    auto p = am.minimize({-1.0, 1.0}, 0.5);

    EXPECT_NEAR(p[0],    1.0, 1e-3);
    EXPECT_NEAR(p[1],    1.0, 1e-3);
    EXPECT_NEAR(am.fret, 0.0, 1e-4);
}

TEST(Amoeba, Quadratic3D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    exprmin::Amoeba am{f};
    auto p = am.minimize({3.0, 3.0, 3.0}, 1.0);

    EXPECT_NEAR(p[0],    0.0, 1e-4);
    EXPECT_NEAR(p[1],    0.0, 1e-4);
    EXPECT_NEAR(p[2],    0.0, 1e-4);
    EXPECT_NEAR(am.fret, 0.0, 1e-6);
}

// ─────────────────────────────────────────────────────────────
// SimAnneal (Simulated Annealing) tests
//
// SA is stochastic: tolerances are generous (1e-2) and test parameters
// are chosen to give reliable convergence across random seeds.
// ─────────────────────────────────────────────────────────────

TEST(SimAnneal, Bowl2D) {
    // f(x,y) = (x-1)^2 + (y-2)^2 — minimum at (1,2), f=0
    // Slow schedule so the cold Amoeba phase can converge.
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::SimAnneal sa{f, 1.0, 0.95, 20};
    auto p = sa.minimize({3.0, 0.0}, 1.0);

    EXPECT_NEAR(p[0],    1.0, 1e-3);
    EXPECT_NEAR(p[1],    2.0, 1e-3);
    EXPECT_NEAR(sa.fret, 0.0, 1e-5);
}

TEST(SimAnneal, Rosenbrock) {
    // f(x,y) = (1-x)^2 + 100(y-x^2)^2 — minimum at (1,1), f=0
    // Higher T0 enables broad exploration; slow cooling lets it settle.
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    exprmin::SimAnneal sa{f, 10.0, 0.97, 30};
    auto p = sa.minimize({-1.0, 1.0}, 0.5);

    EXPECT_NEAR(p[0],    1.0, 1e-2);
    EXPECT_NEAR(p[1],    1.0, 1e-2);
    EXPECT_NEAR(sa.fret, 0.0, 1e-3);
}

TEST(SimAnneal, Quadratic3D) {
    // f(x,y,z) = x^2 + 2y^2 + 3z^2 — minimum at origin
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    exprmin::SimAnneal sa{f, 2.0, 0.95, 30};
    auto p = sa.minimize({3.0, 3.0, 3.0}, 1.0);

    EXPECT_NEAR(p[0],    0.0, 1e-3);
    EXPECT_NEAR(p[1],    0.0, 1e-3);
    EXPECT_NEAR(p[2],    0.0, 1e-3);
    EXPECT_NEAR(sa.fret, 0.0, 1e-5);
}

TEST(SimAnneal, BestPointTracked) {
    // fret must equal f(returned point) — verifies best-tracking bookkeeping
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::SimAnneal sa{f, 1.0, 0.9, 50};
    auto p = sa.minimize({0.0, 0.0}, 1.0);

    const double fx = sa.eval_at(p);
    EXPECT_NEAR(sa.fret, fx, 1e-10);
}

// ─────────────────────────────────────────────────────────────
// Compile-time / trait tests
// ─────────────────────────────────────────────────────────────

TEST(GoldenTraits, SymbolAutoDeduced) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = x * x;

    // Symbol extracted at compile time — no char template arg needed on Golden
    using Syms = diff::extract_symbols_from_expr_t<decltype(f)>;
    constexpr bool one_var = boost::mp11::mp_size<Syms>::value == 1;
    EXPECT_TRUE(one_var);
}

// ─────────────────────────────────────────────────────────────
// Dbrent tests (NR §10.4 — derivative-aware 1D minimizer)
// ─────────────────────────────────────────────────────────────

TEST(Dbrent, QuadraticMinimum) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto f = (x - diff::Constant<double>{2.0}) * (x - diff::Constant<double>{2.0});

    exprmin::Dbrent db{f};
    double xmin = db.minimize(0.0, 5.0);

    EXPECT_NEAR(xmin,   2.0, kTol);
    EXPECT_NEAR(db.fmin, 0.0, kTol * kTol);
}

TEST(Dbrent, SineMinimum) {
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = sin(y);

    exprmin::Dbrent db{f};
    db.ax = 3.0; db.bx = std::numbers::pi * 1.5; db.cx = 6.0;
    db.fa = db.eval_at(db.ax); db.fb = db.eval_at(db.bx); db.fc = db.eval_at(db.cx);
    double xmin = db.minimize();

    EXPECT_NEAR(xmin,    3.0 * std::numbers::pi / 2.0, kTol);
    EXPECT_NEAR(db.fmin, -1.0, kTol);
}

TEST(Dbrent, QuarticMinimum) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto d = x - diff::Constant<double>{1.0};
    auto f = d * d * d * d;

    exprmin::Dbrent db{f};
    double xmin = db.minimize(0.0, 3.0);

    EXPECT_NEAR(xmin,    1.0, kTol);
    EXPECT_NEAR(db.fmin, 0.0, kTol * kTol);
}

// ─────────────────────────────────────────────────────────────
// DFrprmn (CG with derivative line search) tests
// ─────────────────────────────────────────────────────────────

TEST(DFrprmn, Bowl2D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::DFrprmn<decltype(f)> cg{f};
    auto p = cg.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],    1.0, kTol);
    EXPECT_NEAR(p[1],    2.0, kTol);
    EXPECT_NEAR(cg.fret, 0.0, kTol * kTol);
}

TEST(DFrprmn, Rosenbrock) {
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    exprmin::DFrprmn<decltype(f)> cg{f, 1e-10};
    auto p = cg.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],    1.0, 1e-4);
    EXPECT_NEAR(p[1],    1.0, 1e-4);
    EXPECT_NEAR(cg.fret, 0.0, 1e-6);
}

TEST(DFrprmn, Quadratic3D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    exprmin::DFrprmn<decltype(f)> cg{f};
    auto p = cg.minimize({3.0, 3.0, 3.0});

    EXPECT_NEAR(p[0],    0.0, kTol);
    EXPECT_NEAR(p[1],    0.0, kTol);
    EXPECT_NEAR(p[2],    0.0, kTol);
    EXPECT_NEAR(cg.fret, 0.0, kTol * kTol);
}

TEST(DFrprmn, FletcherReeves) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::DFrprmn<decltype(f), exprmin::CGMethod::FletcherReeves> cg{f};
    auto p = cg.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0], 1.0, kTol);
    EXPECT_NEAR(p[1], 2.0, kTol);
}

// ─────────────────────────────────────────────────────────────
// DBFGS (BFGS with derivative line search) tests
// ─────────────────────────────────────────────────────────────

TEST(DBFGS, Bowl2D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{1.0}) * (x - diff::Constant<double>{1.0})
           + (y - diff::Constant<double>{2.0}) * (y - diff::Constant<double>{2.0});

    exprmin::DBFGS<decltype(f)> bfgs{f};
    auto p = bfgs.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],      1.0, kTol);
    EXPECT_NEAR(p[1],      2.0, kTol);
    EXPECT_NEAR(bfgs.fret, 0.0, kTol * kTol);
}

TEST(DBFGS, Rosenbrock) {
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = diff::Constant<double>{1.0} - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;

    exprmin::DBFGS<decltype(f)> bfgs{f, 1e-10};
    auto p = bfgs.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],      1.0, 1e-4);
    EXPECT_NEAR(p[1],      1.0, 1e-4);
    EXPECT_NEAR(bfgs.fret, 0.0, 1e-6);
}

TEST(DBFGS, Quadratic3D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    exprmin::DBFGS<decltype(f)> bfgs{f};
    auto p = bfgs.minimize({3.0, 3.0, 3.0});

    EXPECT_NEAR(p[0],      0.0, kTol);
    EXPECT_NEAR(p[1],      0.0, kTol);
    EXPECT_NEAR(p[2],      0.0, kTol);
    EXPECT_NEAR(bfgs.fret, 0.0, kTol * kTol);
}

// ─────────────────────────────────────────────────────────────
// LevenbergMarquardt tests
// ─────────────────────────────────────────────────────────────

// Helper: mp_list of integral_constant<char,C> — matches the symbol list type
// used by extract_symbols_from_expr_t.
template <char... Cs>
using sym_list = boost::mp11::mp_list<std::integral_constant<char, Cs>...>;

// Build a DataPoint for a 1D input model.
template <typename LMT>
typename LMT::DataPoint make_pt(double x_val, double y_val, double w = 1.0) {
    return {typename LMT::InputVec{x_val}, y_val, w};
}

TEST(LevenbergMarquardt, LinearModel) {
    // f(x; a, b) = a*x + b  — true params a=2, b=3
    auto a = diff::Variable<double, 'a'>{0.0};
    auto b = diff::Variable<double, 'b'>{0.0};
    auto x = diff::Variable<double, 'x'>{0.0};
    auto model = a * x + b;

    using ParamSyms = sym_list<'a', 'b'>;
    using InputSyms = sym_list<'x'>;
    exprmin::LevenbergMarquardt<decltype(model), ParamSyms, InputSyms> lm{model};

    std::vector<decltype(lm)::DataPoint> data;
    for (int i = 0; i < 10; ++i) {
        double xi = static_cast<double>(i);
        data.push_back(make_pt<decltype(lm)>(xi, 2.0 * xi + 3.0));
    }

    decltype(lm)::ParamVec p0{1.0, 1.0};
    auto p = lm.fit(p0, data);

    EXPECT_NEAR(p[0], 2.0, 1e-6); // a
    EXPECT_NEAR(p[1], 3.0, 1e-6); // b
}

TEST(LevenbergMarquardt, ExponentialDecay) {
    // f(x; a, b) = a * exp(-b*x)  — true params a=5, b=0.5
    auto a = diff::Variable<double, 'a'>{0.0};
    auto b = diff::Variable<double, 'b'>{0.0};
    auto x = diff::Variable<double, 'x'>{0.0};
    auto model = a * exp(-b * x);

    using ParamSyms = sym_list<'a', 'b'>;
    using InputSyms = sym_list<'x'>;
    exprmin::LevenbergMarquardt<decltype(model), ParamSyms, InputSyms> lm{model};

    std::vector<decltype(lm)::DataPoint> data;
    for (int i = 0; i < 15; ++i) {
        double xi = 0.2 * i;
        data.push_back(make_pt<decltype(lm)>(xi, 5.0 * std::exp(-0.5 * xi)));
    }

    decltype(lm)::ParamVec p0{3.0, 1.0}; // perturbed initial guess
    auto p = lm.fit(p0, data);

    EXPECT_NEAR(p[0], 5.0, 1e-5); // a
    EXPECT_NEAR(p[1], 0.5, 1e-5); // b
}

TEST(LevenbergMarquardt, Gaussian) {
    // f(x; a, b, c) = a * exp(-(x-b)^2 / (2*c^2))  true: a=4, b=2, c=1
    auto a = diff::Variable<double, 'a'>{0.0};
    auto b = diff::Variable<double, 'b'>{0.0};
    auto c = diff::Variable<double, 'c'>{0.0};
    auto x = diff::Variable<double, 'x'>{0.0};
    auto two  = diff::Constant<double>{2.0};
    auto model = a * exp(-(x - b) * (x - b) / (two * c * c));

    using ParamSyms = sym_list<'a', 'b', 'c'>;
    using InputSyms = sym_list<'x'>;
    exprmin::LevenbergMarquardt<decltype(model), ParamSyms, InputSyms> lm{model};

    std::vector<decltype(lm)::DataPoint> data;
    for (int i = 0; i < 20; ++i) {
        double xi = -3.0 + 0.3 * i;
        data.push_back(make_pt<decltype(lm)>(
            xi, 4.0 * std::exp(-(xi - 2.0) * (xi - 2.0) / 2.0)));
    }

    decltype(lm)::ParamVec p0{3.0, 1.5, 0.8};
    auto p = lm.fit(p0, data);

    EXPECT_NEAR(p[0], 4.0, 1e-4); // a
    EXPECT_NEAR(p[1], 2.0, 1e-4); // b (centre)
    EXPECT_NEAR(p[2], 1.0, 1e-4); // c (width)
}

TEST(LevenbergMarquardt, NoInputVarSingleParam) {
    // f(a) = a — single parameter, no input variable.
    // Multiple data points: {3,4,5,6,7}; LM should converge to a = mean = 5.
    auto a = diff::Variable<double, 'a'>{0.0};
    auto model = a + diff::Constant<double>{0.0}; // wrap to ensure CExpression

    using LMT = exprmin::LevenbergMarquardt<decltype(model)>;
    LMT lm{model};

    std::vector<LMT::DataPoint> data;
    for (double target : {3.0, 4.0, 5.0, 6.0, 7.0})
        data.push_back({LMT::InputVec{}, target, 1.0});

    auto p = lm.fit(LMT::ParamVec{0.0}, data);

    EXPECT_NEAR(p[0], 5.0, 1e-6); // mean of {3,4,5,6,7}
}

// ─────────────────────────────────────────────────────────────
// AugLag (Augmented Lagrangian) tests
// ─────────────────────────────────────────────────────────────

TEST(AugLag, EqualityConstrained) {
    // min x²+y²  s.t.  x+y−1 = 0
    // Analytic: x=y=0.5, f=0.5
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = x * x + y * y;
    auto h = x + y - diff::Constant<double>{1.0};

    exprmin::AugLag al{f, exprmin::make_eq(h), std::tuple{}};
    auto p = al.minimize({2.0, 2.0});

    EXPECT_NEAR(p[0],    0.5, 1e-4);
    EXPECT_NEAR(p[1],    0.5, 1e-4);
    EXPECT_NEAR(al.fret, 0.5, 1e-4);
}

TEST(AugLag, InequalityConstrained) {
    // min (x−3)²+(y−3)²  s.t.  x+y−4 ≤ 0
    // Unconstrained min at (3,3) violates x+y≤4; active-constraint min at x=y=2, f=2
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - diff::Constant<double>{3.0}) * (x - diff::Constant<double>{3.0})
           + (y - diff::Constant<double>{3.0}) * (y - diff::Constant<double>{3.0});
    auto g = x + y - diff::Constant<double>{4.0};

    exprmin::AugLag al{f, std::tuple{}, exprmin::make_ineq(g)};
    auto p = al.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],    2.0, 1e-3);
    EXPECT_NEAR(p[1],    2.0, 1e-3);
    EXPECT_NEAR(al.fret, 2.0, 1e-3);
}

TEST(AugLag, BothConstraints) {
    // min x²+y²  s.t.  x−y = 0  (equality)  and  1−x−y ≤ 0  (inequality: x+y≥1)
    // Analytic: x=y=0.5, f=0.5
    auto x = PV(0.0,'x');
    auto y = PV(0.0,'y');
    auto f  = x * x + y * y;
    diff::Equation h  = x - y;                                                   // x=y
    diff::Equation g  = 1.0 - x - y;                    // x+y≥1

    exprmin::AugLag al{f, h, g};
    //exprmin::AugLag al{f, h, g};
    auto p = al.minimize({2.0, 0.0});

    EXPECT_NEAR(p[0],    0.5, 1e-3);
    EXPECT_NEAR(p[1],    0.5, 1e-3);
    EXPECT_NEAR(al.fret, 0.5, 1e-3);
}

// ─────────────────────────────────────────────────────────────
// LBFGS (limited-memory BFGS) tests
// ─────────────────────────────────────────────────────────────

TEST(LBFGS, Bowl2D) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto f = (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);

    exprmin::LBFGS lbfgs{f};
    auto p = lbfgs.minimize({0.0, 0.0});

    EXPECT_NEAR(p[0],       1.0, kTol);
    EXPECT_NEAR(p[1],       2.0, kTol);
    EXPECT_NEAR(lbfgs.fret, 0.0, kTol * kTol);
}

TEST(LBFGS, Rosenbrock) {
    auto x  = diff::Variable<double, 'x'>{0.0};
    auto y  = diff::Variable<double, 'y'>{0.0};
    auto t1 = 1.0 - x;
    auto t2 = y - x * x;
    auto f  = t1 * t1 + 100.0 * t2 * t2;

    exprmin::LBFGS lbfgs{f, 1e-10};
    auto p = lbfgs.minimize({-1.0, 1.0});

    EXPECT_NEAR(p[0],       1.0, 1e-4);
    EXPECT_NEAR(p[1],       1.0, 1e-4);
    EXPECT_NEAR(lbfgs.fret, 0.0, 1e-6);
}

TEST(LBFGS, Quadratic3DWithArmijo) {
    auto x = diff::Variable<double, 'x'>{0.0};
    auto y = diff::Variable<double, 'y'>{0.0};
    auto z = diff::Variable<double, 'z'>{0.0};
    auto f = x * x
           + diff::Constant<double>{2.0} * y * y
           + diff::Constant<double>{3.0} * z * z;

    exprmin::LBFGS<decltype(f), exprmin::Armijo> lbfgs{f};
    auto p = lbfgs.minimize({3.0, 3.0, 3.0});

    EXPECT_NEAR(p[0],       0.0, kTol);
    EXPECT_NEAR(p[1],       0.0, kTol);
    EXPECT_NEAR(p[2],       0.0, kTol);
    EXPECT_NEAR(lbfgs.fret, 0.0, kTol * kTol);
}
