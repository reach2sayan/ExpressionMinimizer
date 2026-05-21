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
using RosenbrockTypes = testing::Types<
    exprmin::Powell<RosenbrockExpr>, exprmin::Frprmn<RosenbrockExpr>,
    exprmin::BFGS<RosenbrockExpr>, exprmin::DFrprmn<RosenbrockExpr>,
    exprmin::DBFGS<RosenbrockExpr>, exprmin::LBFGS<RosenbrockExpr>,
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
TEST(LBFGS, Quadratic3DWithArmijo) {
  auto f = make_quad3d();
  auto lbfgs = exprmin::make_lbfgs<exprmin::Armijo>(f);
  auto p = lbfgs.minimize({3.0, 3.0, 3.0});
  EXPECT_NEAR(p[0], 0.0, kTol);
  EXPECT_NEAR(p[1], 0.0, kTol);
  EXPECT_NEAR(p[2], 0.0, kTol);
  EXPECT_NEAR(lbfgs.get_optimal_value(), 0.0, kTol * kTol);
}

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
