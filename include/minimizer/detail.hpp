#pragma once

#include "expressions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace exprmin::detail {

/**
 * @brief Constructs an (N+1)-vertex simplex around a centre point.
 *
 * Column 0 is @p p; column i+1 is @p p with component i displaced by
 * @p delta.  The result is an N×(N+1) matrix where each column is a vertex.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension of the space.
 * @param p      Centre vertex.
 * @param delta  Side-length of the initial simplex.
 * @return N×(N+1) matrix whose columns are the simplex vertices.
 */
template <typename T, int N>
Eigen::Matrix<T, N, N + 1> constexpr make_simplex(const Eigen::Vector<T, N> &p,
                                                  const T &delta) noexcept {
  Eigen::Matrix<T, N, N + 1> s = p.replicate(1, N + 1);
  s.diagonal(1).array() += delta;
  return s;
}





/**
 * @brief Nelder–Mead trial reflection/expansion/contraction step.
 *
 * Reflects vertex @p ihi through the centroid of the remaining vertices,
 * scaled by @p fac.  Updates the simplex @p s, function-value vector @p y,
 * and the column-sum cache @p psum in place when the trial point improves.
 *
 * @tparam T    Scalar type.
 * @tparam N    Simplex dimension.
 * @param ptr   Object exposing eval_at(Point) → T.
 * @param s     N×(N+1) simplex matrix (columns are vertices).
 * @param y     Function values at each vertex.
 * @param psum  Sum of all vertex columns (cached centroid numerator).
 * @param ihi   Index of the highest (worst) vertex to reflect.
 * @param fac   Reflection factor: −1 = reflect, >1 = expand, 0<fac<1 = contract.
 * @return Function value at the trial point.
 */
template <typename T, std::size_t N>
constexpr T
amotry_impl(auto &&ptr,
            Eigen::Matrix<T, static_cast<int>(N), static_cast<int>(N + 1)> &s,
            Eigen::Vector<T, static_cast<int>(N + 1)> &y,
            Eigen::Vector<T, static_cast<int>(N)> &psum, const std::size_t ihi,
            const T &fac) {
  const T fac1 = (T{1} - fac) / static_cast<T>(N);
  const T fac2 = fac1 - fac;
  const Eigen::Vector<T, static_cast<int>(N)> ptry =
      fac1 * psum - fac2 * s.col(ihi);
  const T ytry = ptr.eval_at(ptry);
  if (ytry < y[ihi]) {
    psum += ptry - s.col(ihi);
    s.col(ihi) = ptry;
    y[ihi] = ytry;
  }
  return ytry;
};

} // namespace exprmin::detail
