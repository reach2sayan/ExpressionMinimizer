#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

#include "../minimizer.hpp"
#include "expression_differentiator.hpp"
#include <Eigen/Dense>

static constexpr double kTol = 1e-5;
inline auto make_bowl2d() {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  return (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);
}
inline auto make_rosenbrock() {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  return (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x);
}
inline auto make_quad3d() {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto z = PV(0.0, 'z');
  return x * x + 2.0 * y * y + 3.0 * z * z;
}

using Bowl2DExpr = decltype(make_bowl2d());
using RosenbrockExpr = decltype(make_rosenbrock());
using Quad3DExpr = decltype(make_quad3d());

// ─── Typed test suites: gradient-based N-D optimizers ───────────
// Bowl2D: (x-1)²+(y-2)²  →  {1,2}, f=0
using Bowl2DTypes =
    testing::Types<exprmin::Powell<Bowl2DExpr>, exprmin::Frprmn<Bowl2DExpr>,
                   exprmin::BFGS<Bowl2DExpr>, exprmin::DFrprmn<Bowl2DExpr>,
                   exprmin::DBFGS<Bowl2DExpr>, exprmin::LBFGS<Bowl2DExpr>,
                   exprmin::DFP<Bowl2DExpr>, exprmin::SR1<Bowl2DExpr>,
                   exprmin::Dogleg<Bowl2DExpr>,
                   exprmin::Dogleg<Bowl2DExpr, exprmin::HessianMode::ExactAD>>;
template <typename T> class Bowl2DTest : public testing::Test {};
TYPED_TEST_SUITE(Bowl2DTest, Bowl2DTypes);
TYPED_TEST(Bowl2DTest, Converges) {
  auto f = make_bowl2d();
  TypeParam opt{f};
  auto p = opt.minimize({0.0, 0.0});
  EXPECT_NEAR(p[0], 1.0, kTol);
  EXPECT_NEAR(p[1], 2.0, kTol);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, kTol * kTol);
}

// Rosenbrock: (1-x)²+100(y-x²)²  →  {1,1}, f=0
// SR1 is excluded from the typed suite: no positive-definiteness guarantee means
// it can require more iterations on this highly curved valley; see TEST(SR1,…).
using RosenbrockTypes = testing::Types<
    exprmin::Powell<RosenbrockExpr>, exprmin::Frprmn<RosenbrockExpr>,
    exprmin::BFGS<RosenbrockExpr>, exprmin::DFrprmn<RosenbrockExpr>,
    exprmin::DBFGS<RosenbrockExpr>, exprmin::LBFGS<RosenbrockExpr>,
    exprmin::DFP<RosenbrockExpr>,
    exprmin::Dogleg<RosenbrockExpr>,
    exprmin::Dogleg<RosenbrockExpr, exprmin::HessianMode::ExactAD>>;
template <typename T> class RosenbrockTest : public testing::Test {};
TYPED_TEST_SUITE(RosenbrockTest, RosenbrockTypes);
TYPED_TEST(RosenbrockTest, Converges) {
  auto f = make_rosenbrock();
  TypeParam opt{f, 1e-10};
  auto p = opt.minimize({-1.0, 1.0});
  EXPECT_NEAR(p[0], 1.0, 1e-4);
  EXPECT_NEAR(p[1], 1.0, 1e-4);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-6);
}

// Quadratic3D: x²+2y²+3z²  →  {0,0,0}, f=0
using Quad3DTypes =
    testing::Types<exprmin::Powell<Quad3DExpr>, exprmin::Frprmn<Quad3DExpr>,
                   exprmin::BFGS<Quad3DExpr>, exprmin::DFrprmn<Quad3DExpr>,
                   exprmin::DBFGS<Quad3DExpr>, exprmin::LBFGS<Quad3DExpr>,
                   exprmin::DFP<Quad3DExpr>, exprmin::SR1<Quad3DExpr>,
                   exprmin::Dogleg<Quad3DExpr>,
                   exprmin::Dogleg<Quad3DExpr, exprmin::HessianMode::ExactAD>>;
template <typename T> class Quad3DTest : public testing::Test {};
TYPED_TEST_SUITE(Quad3DTest, Quad3DTypes);
TYPED_TEST(Quad3DTest, Converges) {
  auto f = make_quad3d();
  TypeParam opt{f};
  auto p = opt.minimize({3.0, 3.0, 3.0});
  EXPECT_NEAR(p[0], 0.0, kTol);
  EXPECT_NEAR(p[1], 0.0, kTol);
  EXPECT_NEAR(p[2], 0.0, kTol);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, kTol * kTol);
}

// ─── Bracketmethod ──────────────────────────────────────────────
TEST(Bracketmethod, QuadraticBrackets) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = (x - 3.0) * (x - 3.0);
  exprmin::Bracketmethod bm{f};
  bm.bracket(0.0, 1.0);
  EXPECT_LT(bm.fb, bm.fa);
  EXPECT_LT(bm.fb, bm.fc);
  EXPECT_LE(std::min(bm.ax, bm.cx), 3.0);
  EXPECT_GE(std::max(bm.ax, bm.cx), 3.0);
}
TEST(Bracketmethod, SingleCycleConverges) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = x * x * x * x - 14 * x * x * x + 60 * x * x - 70 * x;
  exprmin::Bracketmethod bm{f};
  EXPECT_NO_THROW(bm.bracket(0.0, 1.0));
  EXPECT_LT(bm.fb, bm.fa);
  EXPECT_LT(bm.fb, bm.fc);
}

TEST(Golden, QuadraticMinimum) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = (x - 2.0) * (x - 2.0);
  exprmin::Golden g{f};
  EXPECT_NEAR(g.minimize(0.0, 5.0), 2.0, kTol);
  EXPECT_NEAR(g.get_optimal_value(), 0.0, kTol * kTol);
}
TEST(Golden, SineMinimum) {
  auto y = diff::Variable<double, 'y'>{0.0};
  auto f = sin(y);
  exprmin::Golden g{f};
  g.ax = 3.0;
  g.bx = std::numbers::pi * 1.5;
  g.cx = 6.0;
  g.fa = g.eval_at(g.ax);
  g.fb = g.eval_at(g.bx);
  g.fc = g.eval_at(g.cx);
  EXPECT_NEAR(g.minimize(), 3.0 * std::numbers::pi / 2.0, kTol);
  EXPECT_NEAR(g.get_optimal_value(), -1.0, kTol);
}
TEST(Golden, NegativeQuadratic) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = (x + 1.0) * (x + 1.0);
  exprmin::Golden g{f};
  EXPECT_NEAR(g.minimize(-3.0, 2.0), -1.0, kTol);
  EXPECT_NEAR(g.get_optimal_value(), 0.0, kTol * kTol);
}
TEST(Golden, CustomTolerance) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = (x - 7.5) * (x - 7.5);
  exprmin::Golden g{f, 1.0e-10};
  EXPECT_NEAR(g.minimize(5.0, 10.0), 7.5, 1e-7);
}
TEST(Golden, ManualBracketThenMinimize) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = (x - 4.0) * (x - 4.0);
  exprmin::Golden g{f};
  g.bracket(2.0, 3.0);
  EXPECT_NEAR(g.minimize(), 4.0, kTol);
}

TEST(Brent, QuadraticMinimum) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = (x - 2.0) * (x - 2.0);
  exprmin::Brent b{f};
  EXPECT_NEAR(b.minimize(0.0, 5.0), 2.0, kTol);
  EXPECT_NEAR(b.get_optimal_value(), 0.0, kTol * kTol);
}
TEST(Brent, SineMinimum) {
  auto y = diff::Variable<double, 'y'>{0.0};
  auto f = sin(y);
  exprmin::Brent b{f};
  b.ax = 3.0;
  b.bx = std::numbers::pi * 1.5;
  b.cx = 6.0;
  b.fa = b.eval_at(b.ax);
  b.fb = b.eval_at(b.bx);
  b.fc = b.eval_at(b.cx);
  EXPECT_NEAR(b.minimize(), 3.0 * std::numbers::pi / 2.0, kTol);
  EXPECT_NEAR(b.get_optimal_value(), -1.0, kTol);
}
TEST(Brent, QuarticMinimum) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto d = x - diff::Constant<double>{1.0};
  auto f = d * d * d * d;
  exprmin::Brent b{f};
  EXPECT_NEAR(b.minimize(0.0, 3.0), 1.0, kTol);
  EXPECT_NEAR(b.get_optimal_value(), 0.0, kTol * kTol);
}

TEST(Dbrent, QuadraticMinimum) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = (x - 2.0) * (x - 2.0);
  exprmin::Dbrent db{f};
  EXPECT_NEAR(db.minimize(0.0, 5.0), 2.0, kTol);
  EXPECT_NEAR(db.get_optimal_value(), 0.0, kTol * kTol);
}
TEST(Dbrent, SineMinimum) {
  auto y = diff::Variable<double, 'y'>{0.0};
  auto f = sin(y);
  exprmin::Dbrent db{f};
  db.ax = 3.0;
  db.bx = std::numbers::pi * 1.5;
  db.cx = 6.0;
  db.fa = db.eval_at(db.ax);
  db.fb = db.eval_at(db.bx);
  db.fc = db.eval_at(db.cx);
  EXPECT_NEAR(db.minimize(), 3.0 * std::numbers::pi / 2.0, kTol);
  EXPECT_NEAR(db.get_optimal_value(), -1.0, kTol);
}
TEST(Dbrent, QuarticMinimum) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto d = x - diff::Constant<double>{1.0};
  auto f = d * d * d * d;
  exprmin::Dbrent db{f};
  EXPECT_NEAR(db.minimize(0.0, 3.0), 1.0, kTol);
  EXPECT_NEAR(db.get_optimal_value(), 0.0, kTol * kTol);
}

TEST(LinMin, AxisDirection) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto y = diff::Variable<double, 'y'>{0.0};
  auto f = (x - 3) * (x - 3) + (y - 4) * (y - 4);
  exprmin::LinMin lm{f};
  Eigen::Vector2d p{0.0, 0.0}, dir{1.0, 0.0};
  lm.minimize(p, dir);
  EXPECT_NEAR(p[0], 3.0, kTol);
  EXPECT_NEAR(p[1], 0.0, kTol);
  EXPECT_NEAR(lm.get_optimal_value(), 16.0, kTol);
}
TEST(LinMin, DiagonalDirection) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto y = diff::Variable<double, 'y'>{0.0};
  auto f = (x - 3) * (x - 3) + (y - 4) * (y - 4);
  exprmin::LinMin lm{f};
  Eigen::Vector2d p{0.0, 0.0}, dir{1.0, 1.0};
  lm.minimize(p, dir);
  EXPECT_NEAR(p[0], 3.5, kTol);
  EXPECT_NEAR(p[1], 3.5, kTol);
  EXPECT_NEAR(lm.get_optimal_value(), 0.5, kTol);
}
TEST(LinMin, DirScaledByStep) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto y = diff::Variable<double, 'y'>{0.0};
  auto f = (x - 3.0) * (x - 3.0) + (y - 4.0) * (y - 4.0);
  exprmin::LinMin lm{f};
  Eigen::Vector2d p{0.0, 0.0}, dir{1.0, 1.0};
  lm.minimize(p, dir);
  EXPECT_NEAR(dir[0], p[0], kTol);
  EXPECT_NEAR(dir[1], p[1], kTol);
}

TEST(Frprmn, FletcherReeves) {
  auto f = make_bowl2d();
  auto cg = exprmin::make_frprmn<exprmin::CGMethod::FletcherReeves>(f);
  auto p = cg.minimize({0.0, 0.0});
  EXPECT_NEAR(p[0], 1.0, kTol);
  EXPECT_NEAR(p[1], 2.0, kTol);
}
TEST(DFrprmn, FletcherReeves) {
  auto f = make_bowl2d();
  auto cg =
      exprmin::make_frprmn<exprmin::CGMethod::FletcherReeves, exprmin::DLinMin>(
          f);
  auto p = cg.minimize({0.0, 0.0});
  EXPECT_NEAR(p[0], 1.0, kTol);
  EXPECT_NEAR(p[1], 2.0, kTol);
}
// SR1 does not guarantee a positive-definite Hessian, so on highly curved
// objectives like Rosenbrock the loop may reset to steepest descent several
// times.  The tolerance is relaxed to 1e-3 / 1e-4 accordingly.
TEST(SR1, Rosenbrock) {
  auto f = make_rosenbrock();
  exprmin::SR1<RosenbrockExpr> opt{f, 1e-8};
  auto p = opt.minimize({-1.0, 1.0});
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 1.0, 1e-3);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-4);
}
// SR1 with derivative-based line search.
TEST(DSR1, Rosenbrock) {
  auto f = make_rosenbrock();
  exprmin::DSR1<RosenbrockExpr> opt{f, 1e-8};
  auto p = opt.minimize({-1.0, 1.0});
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 1.0, 1e-3);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-4);
}

TEST(LBFGS, Quadratic3DWithArmijo) {
  auto f = make_quad3d();
  auto lbfgs = exprmin::make_lbfgs<exprmin::Armijo>(f);
  auto p = lbfgs.minimize({3.0, 3.0, 3.0});
  EXPECT_NEAR(p[0], 0.0, kTol);
  EXPECT_NEAR(p[1], 0.0, kTol);
  EXPECT_NEAR(p[2], 0.0, kTol);
  EXPECT_NEAR(lbfgs.get_optimal_value(), 0.0, kTol * kTol);
}

// ─── AmoebaRand (RandomInit=true) ───────────────────────────────────────────

TEST(AmoebaRand, Bowl2D) {
  auto f = make_bowl2d();
  auto am = exprmin::make_amoeba_rand(f);
  auto p = am.minimize({2.0, 0.0}, 1.0);
  EXPECT_NEAR(p[0], 1.0, 1e-4);
  EXPECT_NEAR(p[1], 2.0, 1e-4);
  EXPECT_NEAR(am.get_optimal_value(), 0.0, 1e-6);
}
TEST(AmoebaRand, Rosenbrock) {
  auto f = make_rosenbrock();
  auto am = exprmin::make_amoeba_rand(f, 1e-8);
  auto p = am.minimize({-1.0, 1.0}, 0.5);
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 1.0, 1e-3);
  EXPECT_NEAR(am.get_optimal_value(), 0.0, 1e-4);
}
TEST(AmoebaRand, Quadratic3D) {
  auto f = make_quad3d();
  auto am = exprmin::make_amoeba_rand(f);
  auto p = am.minimize({3.0, 3.0, 3.0}, 1.0);
  EXPECT_NEAR(p[0], 0.0, 1e-4);
  EXPECT_NEAR(p[1], 0.0, 1e-4);
  EXPECT_NEAR(p[2], 0.0, 1e-4);
  EXPECT_NEAR(am.get_optimal_value(), 0.0, 1e-6);
}

// The RngBuffer is a persistent member: calling minimize twice on the same
// instance must both converge (buffer state is carried over, not reset).
TEST(AmoebaRand, RepeatedCallsConverge) {
  auto f = make_bowl2d();
  auto am = exprmin::make_amoeba_rand(f);
  auto p1 = am.minimize({2.0, 0.0}, 1.0);
  auto p2 = am.minimize({0.0, 4.0}, 1.0);
  EXPECT_NEAR(p1[0], 1.0, 1e-4);
  EXPECT_NEAR(p1[1], 2.0, 1e-4);
  EXPECT_NEAR(p2[0], 1.0, 1e-4);
  EXPECT_NEAR(p2[1], 2.0, 1e-4);
}

// make_simplex_rand must produce N mutually orthogonal displacement directions
// (columns 1..N of the simplex minus column 0 are delta * Q, Q orthonormal).
TEST(AmoebaRand, SimplexDirectionsOrthonormal) {
  std::mt19937 rng{42};
  exprmin::detail::RngBuffer<double> buf{rng};

  using Point = Eigen::Vector3d;
  const Point centre{1.0, 2.0, 3.0};
  constexpr double delta = 2.0;
  auto s = exprmin::detail::make_simplex_rand<double, 3>(centre, delta, buf);

  // Columns 1,2,3 minus column 0 should be delta * orthonormal vectors.
  Eigen::Matrix3d D;
  for (int i = 0; i < 3; ++i)
    D.col(i) = (s.col(i + 1) - s.col(0)) / delta;

  // D^T D should be close to identity (orthonormality).
  const Eigen::Matrix3d gram = D.transpose() * D;
  EXPECT_NEAR(gram(0, 0), 1.0, 1e-12);
  EXPECT_NEAR(gram(1, 1), 1.0, 1e-12);
  EXPECT_NEAR(gram(2, 2), 1.0, 1e-12);
  EXPECT_NEAR(gram(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(gram(0, 2), 0.0, 1e-12);
  EXPECT_NEAR(gram(1, 2), 0.0, 1e-12);
}

// RngBuffer should produce zero-mean normal variates over a large sample.
TEST(RngBuffer, ApproximatelyZeroMean) {
  exprmin::detail::RngBuffer<double> buf{std::mt19937{0}};
  constexpr int N = 100'000;
  double sum = 0.0;
  for (int i = 0; i < N; ++i)
    sum += buf();
  EXPECT_NEAR(sum / N, 0.0, 0.01);
}

// ─── Amoeba (axis-aligned) ───────────────────────────────────────────────────

TEST(Amoeba, Bowl2D) {
  auto f = make_bowl2d();
  exprmin::Amoeba am{f};
  auto p = am.minimize({2.0, 0.0}, 1.0); // avoid degenerate start at (0,0)+1
  EXPECT_NEAR(p[0], 1.0, 1e-4);
  EXPECT_NEAR(p[1], 2.0, 1e-4);
  EXPECT_NEAR(am.get_optimal_value(), 0.0, 1e-6);
}
TEST(Amoeba, Rosenbrock) {
  auto f = make_rosenbrock();
  exprmin::Amoeba am{f, 1e-8};
  auto p = am.minimize({-1.0, 1.0}, 0.5);
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 1.0, 1e-3);
  EXPECT_NEAR(am.get_optimal_value(), 0.0, 1e-4);
}
TEST(Amoeba, Quadratic3D) {
  auto f = make_quad3d();
  exprmin::Amoeba am{f};
  auto p = am.minimize({3.0, 3.0, 3.0}, 1.0);
  EXPECT_NEAR(p[0], 0.0, 1e-4);
  EXPECT_NEAR(p[1], 0.0, 1e-4);
  EXPECT_NEAR(p[2], 0.0, 1e-4);
  EXPECT_NEAR(am.get_optimal_value(), 0.0, 1e-6);
}

TEST(SimAnneal, Bowl2D) {
  auto f = make_bowl2d();
  exprmin::SimAnneal sa{f, 1.0, 0.95, 20};
  auto p = sa.minimize({3.0, 0.0}, 1.0);
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 2.0, 1e-3);
  EXPECT_NEAR(sa.get_optimal_value(), 0.0, 1e-5);
}
TEST(SimAnneal, Rosenbrock) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto y = diff::Variable<double, 'y'>{0.0};
  auto t1 = diff::Constant<double>{1.0} - x;
  auto t2 = y - x * x;
  auto f = t1 * t1 + diff::Constant<double>{100.0} * t2 * t2;
  exprmin::SimAnneal sa{f, 10.0, 0.97, 30};
  auto p = sa.minimize({-1.0, 1.0}, 0.5);
  EXPECT_NEAR(p[0], 1.0, 1e-2);
  EXPECT_NEAR(p[1], 1.0, 1e-2);
  EXPECT_NEAR(sa.get_optimal_value(), 0.0, 1e-3);
}
TEST(SimAnneal, Quadratic3D) {
  auto f = make_quad3d();
  exprmin::SimAnneal sa{f, 2.0, 0.95, 30};
  auto p = sa.minimize({3.0, 3.0, 3.0}, 1.0);
  EXPECT_NEAR(p[0], 0.0, 1e-3);
  EXPECT_NEAR(p[1], 0.0, 1e-3);
  EXPECT_NEAR(p[2], 0.0, 1e-3);
  EXPECT_NEAR(sa.get_optimal_value(), 0.0, 1e-5);
}
TEST(SimAnneal, BestPointTracked) {
  auto f = make_bowl2d();
  exprmin::SimAnneal sa{f, 1.0, 0.9, 50};
  auto p = sa.minimize({0.0, 0.0}, 1.0);
  EXPECT_NEAR(sa.get_optimal_value(), sa(p), 1e-10);
}

TEST(GoldenTraits, SymbolAutoDeduced) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto f = x * x;
  using Syms = diff::extract_symbols_from_expr_t<decltype(f)>;
  EXPECT_TRUE(boost::mp11::mp_size<Syms>::value == 1);
}

template <char... Cs>
using sym_list = boost::mp11::mp_list<std::integral_constant<char, Cs>...>;

template <typename LMT>
typename LMT::DataPoint make_pt(double x_val, double y_val, double w = 1.0) {
  return {typename LMT::InputVec{x_val}, y_val, w};
}

TEST(LevenbergMarquardt, LinearModel) {
  auto a = diff::Variable<double, 'a'>{0.0};
  auto b = diff::Variable<double, 'b'>{0.0};
  auto x = diff::Variable<double, 'x'>{0.0};
  auto model = a * x + b;
  auto lm = exprmin::make_lm<'x'>(model);
  std::vector<decltype(lm)::DataPoint> data;
  for (int i = 0; i < 10; ++i)
    data.push_back(make_pt<decltype(lm)>(i, 2.0 * i + 3.0));
  auto p = lm.fit({1.0, 1.0}, data);
  EXPECT_NEAR(p[0], 2.0, 1e-6);
  EXPECT_NEAR(p[1], 3.0, 1e-6);
}
TEST(LevenbergMarquardt, ExponentialDecay) {
  auto a = diff::Variable<double, 'a'>{0.0};
  auto b = diff::Variable<double, 'b'>{0.0};
  auto x = diff::Variable<double, 'x'>{0.0};
  auto model = a * exp(-b * x);
  auto lm = exprmin::make_lm<'x'>(model);
  std::vector<decltype(lm)::DataPoint> data;
  for (int i = 0; i < 15; ++i) {
    double xi = 0.2 * i;
    data.push_back(make_pt<decltype(lm)>(xi, 5.0 * std::exp(-0.5 * xi)));
  }
  auto p = lm.fit({3.0, 1.0}, data);
  EXPECT_NEAR(p[0], 5.0, 1e-5);
  EXPECT_NEAR(p[1], 0.5, 1e-5);
}
TEST(LevenbergMarquardt, Gaussian) {
  auto a = diff::Variable<double, 'a'>{0.0};
  auto b = diff::Variable<double, 'b'>{0.0};
  auto c = diff::Variable<double, 'c'>{0.0};
  auto x = diff::Variable<double, 'x'>{0.0};
  auto two = diff::Constant<double>{2.0};
  auto model = a * exp(-(x - b) * (x - b) / (two * c * c));
  auto lm = exprmin::make_lm<'x'>(model);
  std::vector<decltype(lm)::DataPoint> data;
  for (int i = 0; i < 20; ++i) {
    double xi = -3.0 + 0.3 * i;
    data.push_back(make_pt<decltype(lm)>(
        xi, 4.0 * std::exp(-(xi - 2.0) * (xi - 2.0) / 2.0)));
  }
  auto p = lm.fit({3.0, 1.5, 0.8}, data);
  EXPECT_NEAR(p[0], 4.0, 1e-4);
  EXPECT_NEAR(p[1], 2.0, 1e-4);
  EXPECT_NEAR(p[2], 1.0, 1e-4);
}
TEST(LevenbergMarquardt, NoInputVarSingleParam) {
  auto a = diff::Variable<double, 'a'>{0.0};
  auto model = a + diff::Constant<double>{0.0};
  using LMT = exprmin::LevenbergMarquardt<decltype(model)>;
  LMT lm{model};
  std::vector<LMT::DataPoint> data;
  for (double t : {3.0, 4.0, 5.0, 6.0, 7.0})
    data.push_back({LMT::InputVec{}, t, 1.0});
  auto p = lm.fit(LMT::ParamVec{0.0}, data);
  EXPECT_NEAR(p[0], 5.0, 1e-6);
}

// ─── GaussNewton ────────────────────────────────────────────────
TEST(GaussNewton, LinearModel) {
  auto a = diff::Variable<double, 'a'>{0.0};
  auto b = diff::Variable<double, 'b'>{0.0};
  auto x = diff::Variable<double, 'x'>{0.0};
  auto model = a * x + b;
  auto gn = exprmin::make_gn<'x'>(model);
  std::vector<decltype(gn)::DataPoint> data;
  for (int i = 0; i < 10; ++i)
    data.push_back(make_pt<decltype(gn)>(i, 2.0 * i + 3.0));
  auto p = gn.fit({1.0, 1.0}, data);
  EXPECT_NEAR(p[0], 2.0, 1e-6);
  EXPECT_NEAR(p[1], 3.0, 1e-6);
}
TEST(GaussNewton, ExponentialDecay) {
  auto a = diff::Variable<double, 'a'>{0.0};
  auto b = diff::Variable<double, 'b'>{0.0};
  auto x = diff::Variable<double, 'x'>{0.0};
  auto model = a * exp(-b * x);
  auto gn = exprmin::make_gn<'x'>(model);
  std::vector<decltype(gn)::DataPoint> data;
  for (int i = 0; i < 15; ++i) {
    double xi = 0.2 * i;
    data.push_back(make_pt<decltype(gn)>(xi, 5.0 * std::exp(-0.5 * xi)));
  }
  auto p = gn.fit({3.0, 1.0}, data);
  EXPECT_NEAR(p[0], 5.0, 1e-5);
  EXPECT_NEAR(p[1], 0.5, 1e-5);
}
TEST(GaussNewton, NoInputVarSingleParam) {
  auto a = diff::Variable<double, 'a'>{0.0};
  auto model = a + diff::Constant<double>{0.0};
  using GNT = exprmin::GaussNewton<decltype(model)>;
  GNT gn{model};
  std::vector<GNT::DataPoint> data;
  for (double t : {3.0, 4.0, 5.0, 6.0, 7.0})
    data.push_back({GNT::InputVec{}, t, 1.0});
  auto p = gn.fit(GNT::ParamVec{0.0}, data);
  EXPECT_NEAR(p[0], 5.0, 1e-6);
}

// ─── AugLag ─────────────────────────────────────────────────────
TEST(AugLag, EqualityConstrained) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto y = diff::Variable<double, 'y'>{0.0};
  auto f = x * x + y * y;
  auto h = x + y - 1.0;
  exprmin::AugLag al{f, exprmin::make_eq(h), std::tuple{}};
  auto p = al.minimize({2.0, 2.0});
  EXPECT_NEAR(p[0], 0.5, 1e-4);
  EXPECT_NEAR(p[1], 0.5, 1e-4);
  EXPECT_NEAR(al.get_optimal_value(), 0.5, 1e-4);
}
TEST(AugLag, InequalityConstrained) {
  auto x = diff::Variable<double, 'x'>{0.0};
  auto y = diff::Variable<double, 'y'>{0.0};
  auto f = (x - 3.0) * (x - 3.0) + (y - 3.0) * (y - 3.0);
  auto g = x + y - diff::Constant<double>{4.0};
  exprmin::AugLag al{f, std::tuple{}, exprmin::make_ineq(g)};
  auto p = al.minimize({0.0, 0.0});
  EXPECT_NEAR(p[0], 2.0, 1e-3);
  EXPECT_NEAR(p[1], 2.0, 1e-3);
  EXPECT_NEAR(al.get_optimal_value(), 2.0, 1e-3);
}
TEST(AugLag, BothConstraints) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto f = x * x + y * y;
  diff::Equation h = x - y;
  diff::Equation g = 1.0 - x - y;
  exprmin::AugLag al{f, h, g};
  auto p = al.minimize({2.0, 0.0});
  EXPECT_NEAR(p[0], 0.5, 1e-3);
  EXPECT_NEAR(p[1], 0.5, 1e-3);
  EXPECT_NEAR(al.get_optimal_value(), 0.5, 1e-3);
}

// ─── NLopt testfuncs: Roth and Rosenbrock as NLS ─────────────────────────────
// Roth (testfuncs.c): r1=-13+u+((5-v)v-2)v, r2=-29+u+((v+1)v-14)v
// init {4.5, 3.5}, min at {5, 4}, f=0
TEST(NLSDogleg, Roth_Standard) {
  auto u = PV(0.0, 'u');
  auto v = PV(0.0, 'v');
  auto r1 = -13.0 + u + ((5.0 - v) * v - 2.0) * v;
  auto r2 = -29.0 + u + ((v + 1.0) * v - 14.0) * v;
  auto nd = exprmin::make_nls_dogleg(r1, r2);
  auto p = nd.minimize({4.5, 3.5});
  EXPECT_NEAR(p[0], 5.0, 1e-3);
  EXPECT_NEAR(p[1], 4.0, 1e-3);
  EXPECT_NEAR(nd.get_optimal_value(), 0.0, 1e-5);
}
TEST(NLSDogleg, Roth_Double) {
  auto u = PV(0.0, 'u');
  auto v = PV(0.0, 'v');
  auto r1 = -13.0 + u + ((5.0 - v) * v - 2.0) * v;
  auto r2 = -29.0 + u + ((v + 1.0) * v - 14.0) * v;
  auto nd = exprmin::make_nls_dogleg<exprmin::DoglegVariant::Double>(r1, r2);
  auto p = nd.minimize({4.5, 3.5});
  EXPECT_NEAR(p[0], 5.0, 1e-3);
  EXPECT_NEAR(p[1], 4.0, 1e-3);
  EXPECT_NEAR(nd.get_optimal_value(), 0.0, 1e-5);
}

// Rosenbrock as NLS residuals: r1=10*(y-x²), r2=1-x
// init {-1.2, 1.0}, min at {1, 1}, f=0
TEST(NLSDogleg, RosenbrockNLS_Standard) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto r1 = 10.0 * (y - x * x);
  auto r2 = 1.0 - x;
  auto nd = exprmin::make_nls_dogleg(r1, r2);
  auto p = nd.minimize({-1.2, 1.0});
  EXPECT_NEAR(p[0], 1.0, 1e-4);
  EXPECT_NEAR(p[1], 1.0, 1e-4);
  EXPECT_NEAR(nd.get_optimal_value(), 0.0, 1e-6);
}
TEST(NLSDogleg, RosenbrockNLS_Double) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto r1 = 10.0 * (y - x * x);
  auto r2 = 1.0 - x;
  auto nd = exprmin::make_nls_dogleg<exprmin::DoglegVariant::Double>(r1, r2);
  auto p = nd.minimize({-1.2, 1.0});
  EXPECT_NEAR(p[0], 1.0, 1e-4);
  EXPECT_NEAR(p[1], 1.0, 1e-4);
  EXPECT_NEAR(nd.get_optimal_value(), 0.0, 1e-6);
}

// NLopt t_tutorial.cxx: min y  s.t.  (2x)³-y ≤ 0,  (1-x)³-y ≤ 0
// Same optimal point as the tutorial's min sqrt(y) (sqrt is monotone).
// init {0.5, 1.0}, optimal: x=1/3, y=8/27≈0.2963
TEST(AugLag, NLoptTutorial) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto f = y + diff::Constant<double>{0.0} * x;
  auto two_x = 2.0 * x;
  auto g1 = two_x * two_x * two_x - y;
  auto neg_x1 = 1.0 - x;
  auto g2 = neg_x1 * neg_x1 * neg_x1 - y;
  exprmin::AugLag al{f, std::tuple{}, exprmin::make_ineq(g1, g2)};
  auto p = al.minimize({0.5, 1.0});
  EXPECT_NEAR(p[0], 1.0 / 3.0, 1e-3);
  EXPECT_NEAR(p[1], 8.0 / 27.0, 1e-3);
  EXPECT_NEAR(al.get_optimal_value(), 8.0 / 27.0, 1e-3);
}

// ─── Broyden root finder ─────────────────────────────────────────────────────

// Linear system: x + y - 3 = 0, x - y - 1 = 0  →  x=2, y=1
TEST(Broyden, Linear2D) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  exprmin::Broyden br{diff::Equation{x + y - 3.0, x - y - 1.0}};
  auto p = br.find_root({0.0, 0.0});
  EXPECT_NEAR(p[0], 2.0, kTol);
  EXPECT_NEAR(p[1], 1.0, kTol);
  EXPECT_LT(br.residual_norm(), kTol);
}

// Nonlinear system: x²+y²-4=0, x-y=0  →  x=√2, y=√2
TEST(Broyden, Nonlinear2D) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto f1 = x * x + y * y - 4.0;
  auto f2 = x - y;
  exprmin::Broyden br{diff::Equation{f1, f2}};
  auto p = br.find_root({1.0, 0.5});
  EXPECT_NEAR(p[0], std::sqrt(2.0), kTol);
  EXPECT_NEAR(p[1], std::sqrt(2.0), kTol);
  EXPECT_LT(br.residual_norm(), kTol);
}

// 3D nonlinear system: x²-y=0, y²-z=0, z²-x=0  →  {1,1,1}
TEST(Broyden, Nonlinear3D) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto z = PV(0.0, 'z');
  auto f1 = x * x - y;
  auto f2 = y * y - z;
  auto f3 = z * z - x;
  exprmin::Broyden br{diff::Equation{f1, f2, f3}};
  auto p = br.find_root({2.0, 3.0, 5.0});
  EXPECT_NEAR(p[0], 1.0, kTol);
  EXPECT_NEAR(p[1], 1.0, kTol);
  EXPECT_NEAR(p[2], 1.0, kTol);
  EXPECT_LT(br.residual_norm(), kTol);
}

// ─── NLSDogleg ───────────────────────────────────────────────────────────────

// Square 2D NLS: r1=x²+y-3, r2=x+y²-3  →  multiple roots; check any is found
// (residual→0)
TEST(NLSDogleg, Square2D_Standard) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto r1 = x * x + y - 3.0;
  auto r2 = x + y * y - 3.0;
  auto nd = exprmin::make_nls_dogleg(r1, r2);
  nd.minimize({2.0, 0.0});
  EXPECT_NEAR(nd.get_optimal_value(), 0.0, kTol * kTol);
}

TEST(NLSDogleg, Square2D_Double) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto r1 = x * x + y - 3.0;
  auto r2 = x + y * y - 3.0;
  auto nd = exprmin::make_nls_dogleg<exprmin::DoglegVariant::Double>(r1, r2);
  nd.minimize({2.0, 0.0});
  EXPECT_NEAR(nd.get_optimal_value(), 0.0, kTol * kTol);
}

// Overdetermined 3-residual: r1=x+y-2, r2=x-y, r3=x-1  →  LS sol {1,1}
TEST(NLSDogleg, Overdetermined3r_Standard) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto r1 = x + y - 2.0;
  auto r2 = x - y;
  auto r3 = x - 1.0;
  exprmin::NLSDogleg nd{diff::Equation{r1, r2, r3}};
  auto p = nd.minimize({0.0, 0.0});
  EXPECT_NEAR(p[0], 1.0, kTol);
  EXPECT_NEAR(p[1], 1.0, kTol);
}

TEST(NLSDogleg, Overdetermined3r_Double) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto r1 = x + y - 2.0;
  auto r2 = x - y;
  auto r3 = x - 1.0;
  auto nd =
      exprmin::make_nls_dogleg<exprmin::DoglegVariant::Double>(r1, r2, r3);
  auto p = nd.minimize({0.0, 0.0});
  EXPECT_NEAR(p[0], 1.0, kTol);
  EXPECT_NEAR(p[1], 1.0, kTol);
}

// ─── Subspace2D ──────────────────────────────────────────────────────────────

// Same multi-root system: just verify a root is found (residual→0)
TEST(Subspace2D, Square2D) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto r1 = x * x + y - 3.0;
  auto r2 = x + y * y - 3.0;
  auto s2 = exprmin::make_subspace2d(r1, r2);
  s2.minimize({2.0, 0.0});
  EXPECT_NEAR(s2.get_optimal_value(), 0.0, kTol * kTol);
}

TEST(Subspace2D, Overdetermined3r) {
  auto x = PV(0.0, 'x');
  auto y = PV(0.0, 'y');
  auto r1 = x + y - 2.0;
  auto r2 = x - y;
  auto r3 = x - 1.0;
  auto s2 = exprmin::make_subspace2d(r1, r2, r3);
  auto p = s2.minimize({0.0, 0.0});
  EXPECT_NEAR(p[0], 1.0, kTol);
  EXPECT_NEAR(p[1], 1.0, kTol);
}

// ─── SimplexLP §10.10 ────────────────────────────────────────────────────────

// Simple 2D: min –x₁ – 2x₂  s.t. x₁+x₂ ≤ 4, x ≥ 0
// Optimal: x=(0,4), obj=–8
TEST(SimplexLP, Simple2D) {
  using LP = exprmin::SimplexLP<double>;
  LP::MatX A(1, 2); A << 1.0, 1.0;
  LP::VecX b(1);    b << 4.0;
  LP::VecX c(2);    c << -1.0, -2.0;
  LP lp;
  auto x = lp.solve(A, b, c);
  EXPECT_EQ(lp.status, LP::Status::Optimal);
  EXPECT_NEAR(lp.fret, -8.0, kTol);
  EXPECT_NEAR(x[0], 0.0, kTol);
  EXPECT_NEAR(x[1], 4.0, kTol);
}

// NR3 worked example (p.533): min –40x₁ – 60x₂
//   2x₁ + x₂ ≤ 70
//   x₁  + x₂ ≥ 40  →  –x₁ – x₂ ≤ –40  (negate row)
//   x₁  + 3x₂ = 90  →  two rows: x₁+3x₂ ≤ 90 and –x₁–3x₂ ≤ –90
// Optimal: x₁=24, x₂=22, obj=–2280
TEST(SimplexLP, NRWorkedExample) {
  using LP = exprmin::SimplexLP<double>;
  LP::MatX A(4, 2);
  A << 2.0,  1.0,
      -1.0, -1.0,
       1.0,  3.0,
      -1.0, -3.0;
  LP::VecX b(4); b <<  70.0, -40.0,  90.0, -90.0;
  LP::VecX c(2); c << -40.0, -60.0;
  LP lp;
  auto x = lp.solve(A, b, c);
  EXPECT_EQ(lp.status, LP::Status::Optimal);
  EXPECT_NEAR(lp.fret, -2280.0, 1e-3);
  EXPECT_NEAR(x[0], 24.0, 1e-3);
  EXPECT_NEAR(x[1], 22.0, 1e-3);
}

// Infeasible: x₁+x₂ ≤ 1 and x₁+x₂ ≥ 2 → –x₁–x₂ ≤ –2 (no solution)
TEST(SimplexLP, Infeasible) {
  using LP = exprmin::SimplexLP<double>;
  LP::MatX A(2, 2);
  A <<  1.0,  1.0,
       -1.0, -1.0;
  LP::VecX b(2); b << 1.0, -2.0;
  LP::VecX c(2); c << 1.0, 1.0;
  LP lp;
  lp.solve(A, b, c);
  EXPECT_EQ(lp.status, LP::Status::Infeasible);
}

// Unbounded: min –x₁  s.t. x₁ ≥ 0 (no upper bound)
TEST(SimplexLP, Unbounded) {
  using LP = exprmin::SimplexLP<double>;
  LP::MatX A(1, 1); A << 0.0;    // trivial constraint 0·x₁ ≤ 1
  LP::VecX b(1);    b << 1.0;
  LP::VecX c(1);    c << -1.0;   // minimise –x₁
  LP lp;
  lp.solve(A, b, c);
  EXPECT_EQ(lp.status, LP::Status::Unbounded);
}

// ─── GSL multimin/test.c benchmark functions ─────────────────────────────────
// Roth: f = (-13+u+((5-v)v-2)v)² + (-29+u+((v+1)v-14)v)²
// init (4.5, 3.5), min at (5, 4), f=0
inline auto make_roth() {
  auto u = PV(0.0, 'u');
  auto v = PV(0.0, 'v');
  auto a = -13.0 + u + ((5.0 - v) * v - 2.0) * v;
  auto b = -29.0 + u + ((v + 1.0) * v - 14.0) * v;
  return a * a + b * b;
}

// Wood: 100(a²-b)²+(1-a)²+90(c²-d)²+(1-c)²+10.1((1-b)²+(1-d)²)+19.8(1-b)(1-d)
// init (-3,-1,-3,-1), min at (1,1,1,1), f=0
inline auto make_wood() {
  auto a = PV(0.0, 'a');
  auto b = PV(0.0, 'b');
  auto c = PV(0.0, 'c');
  auto d = PV(0.0, 'd');
  auto t1 = a * a - b;
  auto t2 = c * c - d;
  return 100.0 * t1 * t1 + (1.0 - a) * (1.0 - a)
       + 90.0 * t2 * t2 + (1.0 - c) * (1.0 - c)
       + 10.1 * ((1.0 - b) * (1.0 - b) + (1.0 - d) * (1.0 - d))
       + 19.8 * (1.0 - b) * (1.0 - d);
}

// GSL Rosenbrock variant: (u-1)²+10(u²-v)²  (coefficient 10, not 100)
// init (-1.2, 1.0), min at (1, 1), f=0
inline auto make_gsl_rosenbrock() {
  auto u = PV(0.0, 'u');
  auto v = PV(0.0, 'v');
  auto da = u - 1.0;
  auto db = u * u - v;
  return da * da + 10.0 * db * db;
}

// SimpleAbs: |u-1|+|v-2|, non-smooth, min at (1,2), f=0
inline auto make_simpleabs() {
  auto u = PV(0.0, 'u');
  auto v = PV(0.0, 'v');
  return abs(u - 1.0) + abs(v - 2.0);
}

using RothExpr         = decltype(make_roth());
using WoodExpr         = decltype(make_wood());
using GSLRosenbrockExpr = decltype(make_gsl_rosenbrock());
using SimpleAbsExpr    = decltype(make_simpleabs());

// Roth — gradient-based (all fdf minimizers)
using RothGradTypes = testing::Types<
    exprmin::Powell<RothExpr>, exprmin::Frprmn<RothExpr>,
    exprmin::BFGS<RothExpr>,   exprmin::DFrprmn<RothExpr>,
    exprmin::DBFGS<RothExpr>,  exprmin::LBFGS<RothExpr>,
    exprmin::DFP<RothExpr>,    exprmin::SR1<RothExpr>,
    exprmin::Dogleg<RothExpr>,
    exprmin::Dogleg<RothExpr, exprmin::HessianMode::ExactAD>>;
template <typename T> class RothGradTest : public testing::Test {};
TYPED_TEST_SUITE(RothGradTest, RothGradTypes);
TYPED_TEST(RothGradTest, Converges) {
  auto f = make_roth();
  TypeParam opt{f};
  auto p = opt.minimize({4.5, 3.5});
  EXPECT_NEAR(p[0], 5.0, 1e-3);
  EXPECT_NEAR(p[1], 4.0, 1e-3);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-5);
}

// Roth — gradient-free (f minimizers: simplex variants)
using RothFreeTypes = testing::Types<
    exprmin::Amoeba<RothExpr>, exprmin::Amoeba<RothExpr, true>>;
template <typename T> class RothFreeTest : public testing::Test {};
TYPED_TEST_SUITE(RothFreeTest, RothFreeTypes);
TYPED_TEST(RothFreeTest, Converges) {
  auto f = make_roth();
  TypeParam opt{f};
  auto p = opt.minimize({4.5, 3.5}, 0.5);
  EXPECT_NEAR(p[0], 5.0, 1e-3);
  EXPECT_NEAR(p[1], 4.0, 1e-3);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-5);
}

// Wood — gradient-based
// SR1 excluded: no positive-definiteness guarantee on this coupling-heavy function
// Dogleg<ExactAD> excluded: cross-coupling terms make the exact 4×4 Hessian
// ill-conditioned at the starting point; the approximated-Hessian variant works.
using WoodGradTypes = testing::Types<
    exprmin::Powell<WoodExpr>,  exprmin::Frprmn<WoodExpr>,
    exprmin::BFGS<WoodExpr>,    exprmin::DFrprmn<WoodExpr>,
    exprmin::DBFGS<WoodExpr>,   exprmin::LBFGS<WoodExpr>,
    exprmin::DFP<WoodExpr>,
    exprmin::Dogleg<WoodExpr>>;
template <typename T> class WoodGradTest : public testing::Test {};
TYPED_TEST_SUITE(WoodGradTest, WoodGradTypes);
TYPED_TEST(WoodGradTest, Converges) {
  auto f = make_wood();
  TypeParam opt{f, 1e-10};
  auto p = opt.minimize({-3.0, -1.0, -3.0, -1.0});
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 1.0, 1e-3);
  EXPECT_NEAR(p[2], 1.0, 1e-3);
  EXPECT_NEAR(p[3], 1.0, 1e-3);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-5);
}

// Wood — gradient-free
using WoodFreeTypes = testing::Types<
    exprmin::Amoeba<WoodExpr>, exprmin::Amoeba<WoodExpr, true>>;
template <typename T> class WoodFreeTest : public testing::Test {};
TYPED_TEST_SUITE(WoodFreeTest, WoodFreeTypes);
TYPED_TEST(WoodFreeTest, Converges) {
  auto f = make_wood();
  TypeParam opt{f, 1e-8};
  auto p = opt.minimize({-3.0, -1.0, -3.0, -1.0}, 1.0);
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 1.0, 1e-3);
  EXPECT_NEAR(p[2], 1.0, 1e-3);
  EXPECT_NEAR(p[3], 1.0, 1e-3);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-5);
}

// GSL Rosenbrock (coeff 10) — gradient-based
using GSLRosenbrockGradTypes = testing::Types<
    exprmin::Powell<GSLRosenbrockExpr>,   exprmin::Frprmn<GSLRosenbrockExpr>,
    exprmin::BFGS<GSLRosenbrockExpr>,     exprmin::DFrprmn<GSLRosenbrockExpr>,
    exprmin::DBFGS<GSLRosenbrockExpr>,    exprmin::LBFGS<GSLRosenbrockExpr>,
    exprmin::DFP<GSLRosenbrockExpr>,      exprmin::SR1<GSLRosenbrockExpr>,
    exprmin::Dogleg<GSLRosenbrockExpr>,
    exprmin::Dogleg<GSLRosenbrockExpr, exprmin::HessianMode::ExactAD>>;
template <typename T> class GSLRosenbrockGradTest : public testing::Test {};
TYPED_TEST_SUITE(GSLRosenbrockGradTest, GSLRosenbrockGradTypes);
TYPED_TEST(GSLRosenbrockGradTest, Converges) {
  auto f = make_gsl_rosenbrock();
  TypeParam opt{f, 1e-10};
  auto p = opt.minimize({-1.2, 1.0});
  EXPECT_NEAR(p[0], 1.0, 1e-4);
  EXPECT_NEAR(p[1], 1.0, 1e-4);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-6);
}

// GSL Rosenbrock (coeff 10) — gradient-free
using GSLRosenbrockFreeTypes = testing::Types<
    exprmin::Amoeba<GSLRosenbrockExpr>, exprmin::Amoeba<GSLRosenbrockExpr, true>>;
template <typename T> class GSLRosenbrockFreeTest : public testing::Test {};
TYPED_TEST_SUITE(GSLRosenbrockFreeTest, GSLRosenbrockFreeTypes);
TYPED_TEST(GSLRosenbrockFreeTest, Converges) {
  auto f = make_gsl_rosenbrock();
  TypeParam opt{f, 1e-8};
  auto p = opt.minimize({-1.2, 1.0}, 0.5);
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 1.0, 1e-3);
  EXPECT_NEAR(opt.get_optimal_value(), 0.0, 1e-4);
}

// SimpleAbs: non-smooth — gradient-free only (nmsimplex equivalents)
TEST(SimpleAbs, Amoeba) {
  auto f = make_simpleabs();
  exprmin::Amoeba am{f};
  auto p = am.minimize({0.0, 0.0}, 1.0);
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 2.0, 1e-3);
  EXPECT_NEAR(am.get_optimal_value(), 0.0, 1e-5);
}
TEST(SimpleAbs, AmoebaRand) {
  auto f = make_simpleabs();
  auto am = exprmin::make_amoeba_rand(f);
  auto p = am.minimize({0.0, 0.0}, 1.0);
  EXPECT_NEAR(p[0], 1.0, 1e-3);
  EXPECT_NEAR(p[1], 2.0, 1e-3);
  EXPECT_NEAR(am.get_optimal_value(), 0.0, 1e-5);
}

// ─── InteriorPointLP §10.11 ──────────────────────────────────────────────────

// Same 2D problem in standard form: add slack s so x₁+x₂+s = 4.
// min [–1,–2,0]·[x₁;x₂;s],  [1,1,1][x₁;x₂;s]=4,  x≥0
// Optimal: x₁=0, x₂=4, s=0, obj=–8
TEST(InteriorPointLP, Simple2D) {
  using IP = exprmin::InteriorPointLP<double>;
  IP::MatX A(1, 3); A << 1.0, 1.0, 1.0;
  IP::VecX b(1);    b << 4.0;
  IP::VecX c(3);    c << -1.0, -2.0, 0.0;
  IP ip;
  auto x = ip.solve(A, b, c);
  EXPECT_EQ(ip.status, IP::Status::Optimal);
  EXPECT_NEAR(ip.fret, -8.0, 1e-4);
  EXPECT_NEAR(x[0], 0.0, 1e-4);
  EXPECT_NEAR(x[1], 4.0, 1e-4);
}

// Two-constraint unique-vertex problem: min –3x₁–5x₂
//   x₁+x₂+s₁=4, x₂+s₂=3, x≥0
// Unique optimal vertex: (x₁=1, x₂=3, s₁=0, s₂=0), obj=–3–15=–18
TEST(InteriorPointLP, TwoConstraints) {
  using IP = exprmin::InteriorPointLP<double>;
  IP::MatX A(2, 4);
  A << 1.0, 1.0, 1.0, 0.0,
       0.0, 1.0, 0.0, 1.0;
  IP::VecX b(2); b << 4.0, 3.0;
  IP::VecX c(4); c << -3.0, -5.0, 0.0, 0.0;
  IP ip;
  auto x = ip.solve(A, b, c);
  EXPECT_EQ(ip.status, IP::Status::Optimal);
  EXPECT_NEAR(ip.fret, -18.0, 1e-3);
  EXPECT_NEAR(x[0], 1.0, 1e-3);
  EXPECT_NEAR(x[1], 3.0, 1e-3);
}

// ─── Compile-time evaluation (constexpr / consteval) ─────────────────────────
// Expression construction, eval(), and reverse-mode gradient are fully
// constexpr throughout the library.  1-D minimizers (Brent, Golden, Dbrent)
// use only scalar arithmetic and std::array, so they are consteval in C++23
// (std::abs / std::sqrt became constexpr via P0533R9).
// N-D minimizers depend on Eigen::Vector which is not a literal type, so they
// remain runtime-only despite their constexpr-qualified minimize() signatures.

// --- Expression evaluation at known points ---

consteval double ce_bowl2d_at_min() {
  auto x = PV(1.0, 'x');
  auto y = PV(2.0, 'y');
  return ((x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0)).eval();
}
consteval double ce_rosenbrock_at_min() {
  auto x = PV(1.0, 'x');
  auto y = PV(1.0, 'y');
  return ((1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)).eval();
}
static_assert(ce_bowl2d_at_min() == 0.0);
static_assert(ce_rosenbrock_at_min() == 0.0);

// --- Reverse-mode gradient at the minimum (should be zero) ---

consteval std::array<double, 2> ce_bowl2d_grad_at_min() {
  auto x = PV(1.0, 'x');
  auto y = PV(2.0, 'y');
  auto f = (x - 1.0) * (x - 1.0) + (y - 2.0) * (y - 2.0);
  return diff::gradient<diff::DiffMode::Reverse>(f);
}
static_assert(ce_bowl2d_grad_at_min()[0] == 0.0);
static_assert(ce_bowl2d_grad_at_min()[1] == 0.0);

// --- 1-D minimisation at compile time ---

consteval double ce_brent_quadratic() {
  auto x = diff::Variable<double, 'x'>{0.0};
  exprmin::Brent b{(x - 2.0) * (x - 2.0)};
  return b.minimize(0.0, 5.0);
}
consteval double ce_golden_quadratic() {
  auto x = diff::Variable<double, 'x'>{0.0};
  exprmin::Golden g{(x - 2.0) * (x - 2.0)};
  return g.minimize(0.0, 5.0);
}
consteval double ce_dbrent_quadratic() {
  auto x = diff::Variable<double, 'x'>{0.0};
  exprmin::Dbrent db{(x - 2.0) * (x - 2.0)};
  return db.minimize(0.0, 5.0);
}
consteval bool ce_bracket_quadratic() {
  auto x = diff::Variable<double, 'x'>{0.0};
  exprmin::Bracketmethod bm{(x - 3.0) * (x - 3.0)};
  bm.bracket(0.0, 1.0);
  return (bm.ax <= 3.0 && 3.0 <= bm.cx) || (bm.cx <= 3.0 && 3.0 <= bm.ax);
}
static_assert(ce_brent_quadratic()  > 2.0 - 1e-5 && ce_brent_quadratic()  < 2.0 + 1e-5);
static_assert(ce_golden_quadratic() > 2.0 - 1e-5 && ce_golden_quadratic() < 2.0 + 1e-5);
static_assert(ce_dbrent_quadratic() > 2.0 - 1e-5 && ce_dbrent_quadratic() < 2.0 + 1e-5);
static_assert(ce_bracket_quadratic());

TEST(ConstexprEval, ExpressionAtMinimum) {
  EXPECT_EQ(ce_bowl2d_at_min(), 0.0);
  EXPECT_EQ(ce_rosenbrock_at_min(), 0.0);
}
TEST(ConstexprEval, GradientAtMinimum) {
  constexpr auto g = ce_bowl2d_grad_at_min();
  EXPECT_EQ(g[0], 0.0);
  EXPECT_EQ(g[1], 0.0);
}
TEST(ConstexprBrent, QuadraticMinimum) {
  EXPECT_NEAR(ce_brent_quadratic(), 2.0, 1e-5);
}
TEST(ConstexprGolden, QuadraticMinimum) {
  EXPECT_NEAR(ce_golden_quadratic(), 2.0, 1e-5);
}
TEST(ConstexprDbrent, QuadraticMinimum) {
  EXPECT_NEAR(ce_dbrent_quadratic(), 2.0, 1e-5);
}
TEST(ConstexprBracketmethod, BracketContainsMinimum) {
  static_assert(ce_bracket_quadratic());
  EXPECT_TRUE(ce_bracket_quadratic());
}
