#pragma once

// logging_callback.hpp — concrete logging callbacks (signals2 + spdlog).
// Requires ENABLE_LOGGING=ON (default) which defines EXPRMIN_LOGGING.
// For the abstract callback infrastructure only, include callback/callback.hpp.

#ifdef EXPRMIN_LOGGING

#include "../callback/callback.hpp"
#include <boost/signals2/signal.hpp>
#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <string>

namespace exprmin::logging {

// ── Event structs ─────────────────────────────────────────────────────────────
struct LMOuterEvent    { int iter; double lambda, chi2; };
struct LMInnerEvent    { int iter; double lambda, chi2_new; bool accepted; };
struct GNIterEvent     { int iter; double step_norm, residual_norm; };
struct QNIterEvent     { int iter; double f, scaled_grad_inf, dx_norm; };
struct TRIterEvent     { int iter; double phi, gnorm, delta, rho; bool accepted; };
struct AmoebIterEvent  { int iter; double best_val, worst_val; };
struct AnnealIterEvent { int iter; double temperature, ybest; };

// ── Signal bundles ────────────────────────────────────────────────────────────
struct LMSignals {
    boost::signals2::signal<void(const LMOuterEvent &)> outer;
    boost::signals2::signal<void(const LMInnerEvent &)> inner;
};
struct GNSignals     { boost::signals2::signal<void(const GNIterEvent &)>     iter; };
struct QNSignals     { boost::signals2::signal<void(const QNIterEvent &)>     iter; };
struct TRSignals     { boost::signals2::signal<void(const TRIterEvent &)>     iter; };
struct AmoebSignals  { boost::signals2::signal<void(const AmoebIterEvent &)>  iter; };
struct AnnealSignals { boost::signals2::signal<void(const AnnealIterEvent &)> iter; };

// ── Logging callbacks — each inherits CallbackBase for no-op defaults ─────────
// Only override the algorithm hooks the signals bundle actually covers.

struct LMSignalCallbacks : callback::CallbackBase<LMSignalCallbacks> {
    LMSignals *signals_{nullptr};
    LMSignalCallbacks() = default;
    explicit LMSignalCallbacks(LMSignals *s) noexcept : signals_(s) {}
    template <typename T>
    void on_lm_outer(int iter, T lambda, T chi2) noexcept {
        if (signals_) signals_->outer({iter, static_cast<double>(lambda), static_cast<double>(chi2)});
    }
    template <typename T>
    void on_lm_inner(int iter, T lambda, T chi2_new, bool accepted) noexcept {
        if (signals_) signals_->inner({iter, static_cast<double>(lambda), static_cast<double>(chi2_new), accepted});
    }
};

struct GNSignalCallbacks : callback::CallbackBase<GNSignalCallbacks> {
    GNSignals *signals_{nullptr};
    GNSignalCallbacks() = default;
    explicit GNSignalCallbacks(GNSignals *s) noexcept : signals_(s) {}
    template <typename T>
    void on_gn_iter(int iter, T step_norm, T residual_norm) noexcept {
        if (signals_) signals_->iter({iter, static_cast<double>(step_norm), static_cast<double>(residual_norm)});
    }
};

struct QNSignalCallbacks : callback::CallbackBase<QNSignalCallbacks> {
    QNSignals *signals_{nullptr};
    QNSignalCallbacks() = default;
    explicit QNSignalCallbacks(QNSignals *s) noexcept : signals_(s) {}
    template <typename T>
    void on_qn_iter(int iter, T f, T scaled_grad_inf, T dx_norm) noexcept {
        if (signals_) signals_->iter({iter, static_cast<double>(f), static_cast<double>(scaled_grad_inf), static_cast<double>(dx_norm)});
    }
};

struct TRSignalCallbacks : callback::CallbackBase<TRSignalCallbacks> {
    TRSignals *signals_{nullptr};
    TRSignalCallbacks() = default;
    explicit TRSignalCallbacks(TRSignals *s) noexcept : signals_(s) {}
    template <typename T>
    void on_tr_iter(int iter, T phi, T gnorm, T delta, T rho, bool accepted) noexcept {
        if (signals_)
            signals_->iter({iter, static_cast<double>(phi), static_cast<double>(gnorm),
                            static_cast<double>(delta), static_cast<double>(rho), accepted});
    }
};

struct AmoebSignalCallbacks : callback::CallbackBase<AmoebSignalCallbacks> {
    AmoebSignals *signals_{nullptr};
    AmoebSignalCallbacks() = default;
    explicit AmoebSignalCallbacks(AmoebSignals *s) noexcept : signals_(s) {}
    template <typename T>
    void on_amoeba_iter(int iter, T best_val, T worst_val) noexcept {
        if (signals_) signals_->iter({iter, static_cast<double>(best_val), static_cast<double>(worst_val)});
    }
};

struct AnnealSignalCallbacks : callback::CallbackBase<AnnealSignalCallbacks> {
    AnnealSignals *signals_{nullptr};
    AnnealSignalCallbacks() = default;
    explicit AnnealSignalCallbacks(AnnealSignals *s) noexcept : signals_(s) {}
    template <typename T>
    void on_anneal_iter(int iter, T temperature, T ybest) noexcept {
        if (signals_) signals_->iter({iter, static_cast<double>(temperature), static_cast<double>(ybest)});
    }
};

// ── SpdlogIterSlot ────────────────────────────────────────────────────────────
// Callable that can be connected as a slot to any *Signals signal.
// All event types write to the same single log file in structured sections.
//
// Usage:
//   LMSignals sig;
//   SpdlogIterSlot slot{"run.log"};
//   sig.outer.connect(slot);
//   sig.inner.connect(slot);
struct SpdlogIterSlot {
    std::shared_ptr<spdlog::logger> log;

    // Creates (or truncates) the log file at `path`.
    // The logger is NOT registered in the global spdlog registry — multiple
    // SpdlogIterSlot objects can target different files without name collisions.
    explicit SpdlogIterSlot(const std::string &path) {
        auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, /*truncate=*/true);
        log = std::make_shared<spdlog::logger>("", std::move(sink));
        log->set_pattern("[%T.%e] %v");
        log->set_level(spdlog::level::trace);
    }

    void operator()(const LMOuterEvent &e) const {
        log->info("── LM  iter={:3d}  lambda={:.3e}  chi2={:.8g}", e.iter, e.lambda, e.chi2);
    }
    void operator()(const LMInnerEvent &e) const {
        log->info("         {:8s}  lambda={:.3e}  chi2_new={:.8g}",
                  e.accepted ? "ACCEPTED" : "REJECTED", e.lambda, e.chi2_new);
    }
    void operator()(const GNIterEvent &e) const {
        log->info("── GN  iter={:3d}  |step|={:.3e}  |r|={:.8g}", e.iter, e.step_norm, e.residual_norm);
    }
    void operator()(const QNIterEvent &e) const {
        log->info("── QN  iter={:3d}  f={:.8g}  grad_inf={:.3e}  |dx|={:.3e}",
                  e.iter, e.f, e.scaled_grad_inf, e.dx_norm);
    }
    void operator()(const TRIterEvent &e) const {
        log->info("── TR  iter={:3d}  phi={:.8g}  gnorm={:.3e}  delta={:.3e}  rho={:+.4f}  {}",
                  e.iter, e.phi, e.gnorm, e.delta, e.rho, e.accepted ? "ACC" : "REJ");
    }
    void operator()(const AmoebIterEvent &e) const {
        log->info("── AMO iter={:3d}  best={:.8g}  worst={:.8g}", e.iter, e.best_val, e.worst_val);
    }
    void operator()(const AnnealIterEvent &e) const {
        log->info("── SA  iter={:5d}  T={:.3e}  ybest={:.8g}", e.iter, e.temperature, e.ybest);
    }
};

} // namespace exprmin::logging

#endif // EXPRMIN_LOGGING
