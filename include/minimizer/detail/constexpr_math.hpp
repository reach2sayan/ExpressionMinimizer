#pragma once

#include <type_traits>

namespace exprmin::detail {

// std::abs is not constexpr for floating-point types on MSVC (even in C++23).
// This wrapper provides a constexpr-safe absolute value for use in consteval
// and constexpr contexts across all minimizer implementations.
template <typename T>
  requires std::is_arithmetic_v<T>
[[nodiscard]] constexpr T abs_for_constexpr(T value) noexcept {
  return value < T{} ? -value : value;
}

} // namespace exprmin::detail
