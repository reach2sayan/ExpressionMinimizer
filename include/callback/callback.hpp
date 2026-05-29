#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

// ── Callback infrastructure — no external dependencies
// ────────────────────────
//
// Pattern for custom callbacks:
//   struct MyCallback : CallbackBase<MyCallback> {
//       void on_lm_outer(int iter, double lambda, double chi2) noexcept { ... }
//       // on_lm_inner, on_gn_iter, ... inherited as no-ops
//   };
//
// To compose multiple callbacks pass them to make_callbacks():
//   auto cbs = make_callbacks(MyLogger{}, MyProfiler{}, MyWatchdog{});
//   auto lm  = make_lm<'x'>(expr, cbs);

namespace exprmin::callback {

// ── CallbackBase<Derived>
// ───────────────────────────────────────────────────── CRTP base providing
// constexpr no-op defaults for every iteration hook. Derive from it and
// override only the hooks you care about.
template <typename Derived> struct CallbackBase {
  template <typename... Args> constexpr void on_lm_outer(Args &&...) noexcept {}
  template <typename... Args> constexpr void on_lm_inner(Args &&...) noexcept {}
  template <typename... Args> constexpr void on_gn_iter(Args &&...) noexcept {}
  template <typename... Args> constexpr void on_qn_iter(Args &&...) noexcept {}
  template <typename... Args> constexpr void on_tr_iter(Args &&...) noexcept {}
  template <typename... Args>
  constexpr void on_amoeba_iter(Args &&...) noexcept {}
  template <typename... Args>
  constexpr void on_anneal_iter(Args &&...) noexcept {}

protected:
  constexpr Derived &self() noexcept { return static_cast<Derived &>(*this); }
  constexpr const Derived &self() const noexcept {
    return static_cast<const Derived &>(*this);
  }
};

// ── CCallback concept ──────────────────────────────────────────────────────
// Satisfied by any type that publicly inherits CallbackBase<T>.
template <typename T>
concept CCallback = std::is_base_of_v<CallbackBase<T>, T>;

// ── NoCallbacks
// ─────────────────────────────────────────────────────────────── Zero-overhead
// default: all hooks are inherited no-ops.
// [[no_unique_address]] on the member in each algorithm class makes this
// occupy zero bytes and produce zero instructions.
struct NoCallbacks : CallbackBase<NoCallbacks> {};

// ── CompositeCallbacks<Cbs...>
// ──────────────────────────────────────────────── Dispatches each on_* call to
// every contained callback in order. Satisfies CCallback so it can itself be
// nested inside another composite.
template <CCallback... Cbs>
struct CompositeCallbacks : CallbackBase<CompositeCallbacks<Cbs...>> {
  std::tuple<Cbs...> callbacks_;

  template <typename... Args> constexpr void on_lm_outer(Args &&...args) {
    std::apply([&]<typename... C>(C &...cb) { (cb.on_lm_outer(args...), ...); },
               callbacks_);
  }
  template <typename... Args> constexpr void on_lm_inner(Args &&...args) {
    std::apply([&]<typename... C>(C &...cb) { (cb.on_lm_inner(args...), ...); },
               callbacks_);
  }
  template <typename... Args> constexpr void on_gn_iter(Args &&...args) {
    std::apply([&]<typename... C>(C &...cb) { (cb.on_gn_iter(args...), ...); },
               callbacks_);
  }
  template <typename... Args> constexpr void on_qn_iter(Args &&...args) {
    std::apply([&]<typename... C>(C &...cb) { (cb.on_qn_iter(args...), ...); },
               callbacks_);
  }
  template <typename... Args> constexpr void on_tr_iter(Args &&...args) {
    std::apply([&]<typename... C>(C &...cb) { (cb.on_tr_iter(args...), ...); },
               callbacks_);
  }
  template <typename... Args> constexpr void on_amoeba_iter(Args &&...args) {
    std::apply(
        [&]<typename... C>(C &...cb) { (cb.on_amoeba_iter(args...), ...); },
        callbacks_);
  }
  template <typename... Args> constexpr void on_anneal_iter(Args &&...args) {
    std::apply(
        [&]<typename... C>(C &...cb) { (cb.on_anneal_iter(args...), ...); },
        callbacks_);
  }
};

// Factory: make_callbacks(cb1, cb2, ...) → CompositeCallbacks<Cb1, Cb2, ...>
// Nested composition is allowed: make_callbacks(make_callbacks(a, b), c).
template <CCallback... Cbs> constexpr auto make_callbacks(Cbs... cbs) {
  return CompositeCallbacks<Cbs...>{std::tuple<Cbs...>{std::move(cbs)...}};
}

} // namespace exprmin::callback

// Expose into exprmin:: for convenience — algorithms use
// exprmin::callback::NoCallbacks.
namespace exprmin {
using callback::CallbackBase;
using callback::CCallback;
using callback::CompositeCallbacks;
using callback::make_callbacks;
using callback::NoCallbacks;
} // namespace exprmin
