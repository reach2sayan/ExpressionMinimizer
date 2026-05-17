Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Trust-region CRTP refactoring — eliminate repeated code

 Context

 Dogleg, NLSDogleg, and Subspace2D all follow the same trust-region loop but each
 duplicates: TR constants, convergence check, predicted-reduction formula, ρ ratio, step
 accept/reject, and (for NLSDogleg + Subspace2D) to_arr + eval_rJ.

 The goal is a two-level CRTP hierarchy that eliminates the duplication while keeping each
 class's unique logic (step geometry, Hessian source) in the derived class only.

 ---
 Duplication inventory

 ┌────────────────────────────────────────────┬────────┬───────────┬────────────┐
 │                    Code                    │ Dogleg │ NLSDogleg │ Subspace2D │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ TR constants (0.1 / 0.25 / 2.0 / 0.75)     │ ✓      │ ✓         │ ✓          │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ Convergence check: gnorm/den < tol         │ ✓      │ ✓         │ ✓          │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ pred = -(g·step + ½step·B·step)            │ ✓      │ ✓         │ ✓          │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ rho = (phi-phi_new)/pred                   │ ✓      │ ✓         │ ✓          │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ at_boundary = step.norm() ≥ 0.999·delta    │ ✓      │ ✓         │ ✓          │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ step-size break step.maxCoeff() < tol      │ ✓      │ ✓         │ ✓          │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ to_arr(p)                                  │ —      │ ✓         │ ✓          │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ eval_rJ(p)                                 │ —      │ ✓         │ ✓          │
 ├────────────────────────────────────────────┼────────┼───────────┼────────────┤
 │ member vars (tol, itmax, tr0, trmin, fret) │ ✓      │ ✓         │ ✓          │
 └────────────────────────────────────────────┴────────┴───────────┴────────────┘

 Key differences that stay in derived:
 - eval_state(p): (f,g) via diff::gradient (Dogleg) vs (r,J) via system.jacobian (NLS)
 - compute_step(g, B, delta): dogleg geometry / quartic subspace / Powell dogleg
 - adjust_tr(...): Dogleg pre-shrinks delta to nn (GN step norm) before the ×0.1; NLS uses the simpler 2-case update
 - commit_state(): BFGS rank-2 update (Dogleg) vs recompute J^T J (NLS)

 ---
 Design — two-level CRTP

 TrustRegionBase<Derived, T, N>               ← shared TR loop, constants, members
        │
        ├── NLSTrustRegionBase<Derived, RExprs...>   ← adds system, to_arr, eval_rJ
        │          │
        │          ├── NLSDogleg<Eq, DV>    ← compute_step only
        │          └── Subspace2D<Eq>       ← compute_step only
        │
        └── Dogleg<Expr, HM>                ← eval_grad, BFGS/ExactAD; compute_step

 TrustRegionBase<Derived, T, N> — new file trustregionbase.hpp

 Members: tol, itmax, trustregion0, trustregion_min, fret{}, iter{} (public).

 Static constexpr TR constants (all four, all three classes currently repeat these).

 Protected constructor: (tol, itmax, tr0, trmin).

 Public: get_optimal_value().

 CRTP hooks (called via self().hookname(...)):

 ┌──────────────┬────────────────────────────────┬─────────────────────────────────────────────────────┐
 │     Hook     │           Signature            │                     Called when                     │
 ├──────────────┼────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ eval_state   │ (p) → tuple<T, ParamVec, NMat> │ before loop & re-init                               │
 ├──────────────┼────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ compute_step │ (g, B, delta) → ParamVec       │ each iteration                                      │
 ├──────────────┼────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ eval_trial   │ (p_new) → T                    │ trial point eval; caches new state inside derived   │
 ├──────────────┼────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ adjust_tr    │ (delta&, rho, at_boundary)     │ after rho is known                                  │
 ├──────────────┼────────────────────────────────┼─────────────────────────────────────────────────────┤
 │ commit_state │ () → pair<ParamVec, NMat>      │ on accepted step; returns (g_new, B_new) from cache │
 └──────────────┴────────────────────────────────┴─────────────────────────────────────────────────────┘

 minimize(ParamVec p) loop (lives entirely in base):
 auto [phi, g, B] = self().eval_state(p);
 value_type delta = trustregion0;
 for (iter = 0; iter < itmax; ++iter) {
     // convergence check
     const ParamVec step = self().compute_step(g, B, delta);
     const bool at_boundary = step.norm() >= T{0.999} * delta;
     const T pred = -(g.dot(step) + T{0.5} * step.dot(B * step));
     const T phi_new = self().eval_trial(p + step);
     const T rho = (pred > T{0}) ? (phi - phi_new) / pred : T{0};
     self().adjust_tr(delta, rho, at_boundary);
     if (delta < trustregion_min) break;
     if (rho > T{0}) {
         if (step.cwiseAbs().maxCoeff() < tol) break;
         p += step; phi = phi_new;
         std::tie(g, B) = self().commit_state();
     }
 }
 fret = phi; return p;

 NLSTrustRegionBase<Derived, RExprs...> — in same file or nlsbase.hpp

 Inherits TrustRegionBase<Derived, T, N>.

 Adds: Sys system, NLS type aliases (M, RVec, JMat), to_arr, eval_rJ.

 Private cache members: RVec r_, JMat J_ (current); RVec r_new_, JMat J_new_ (trial).

 Implements the 4 non-step hooks:
 - eval_state(p): calls eval_rJ(p) → stores in r_, J_; returns {0.5*r_^2, J_^T r_, J_^T J_}
 - eval_trial(p_new): calls eval_rJ(p_new) → stores in r_new_, J_new_; returns 0.5*r_new_^2
 - adjust_tr(delta, rho, at_boundary): simple 2-case (no nn)
 - commit_state(): swaps r_←r_new_, J_←J_new_; returns {J_^T r_, J_^T J_}

 Constructor: (diff::Equation<RExprs...> sys, tol, itmax, tr0, trmin).

 Deduction guides live on the final derived classes (as now).

 Dogleg<Expr, HM> — modified dogleg.hpp

 Inherits TrustRegionBase<Dogleg<Expr,HM>, value_type, N>.

 Drops: member variables, TR constants, convergence check, pred/rho/loop (all move to base).

 Keeps: expr, eval_at, eval_grad, BFGS/ExactAD Hessian logic, compute_step.

 Private cache: Point g_new_ (stored during eval_trial), T nn_ (stored during compute_step = dx_gn.norm()).

 Implements hooks:
 - eval_state(p): eval_grad(p) → {f, g, B_init} (B = Identity for BFGS, or ExactAD)
 - eval_trial(p_new): eval_grad(p_new) → stores g_new_; returns f_new
 - adjust_tr(delta, rho, at_boundary): same as current update_trust_region using stored nn_
 - commit_state(): BFGS update with dg = g_new_ - g_old; returns {g_new_, B_updated}
 - compute_step(g, B, delta): existing logic; stores nn_ = dx_gn.norm()

 ---
 Files to create / modify

 ┌───────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐
 │                 File                  │                                      Change                                      │
 ├───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ include/minimizer/trustregionbase.hpp │ New — TrustRegionBase<Derived,T,N> + NLSTrustRegionBase<Derived,RExprs...>       │
 ├───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ include/minimizer/dogleg.hpp          │ Inherit TrustRegionBase; strip shared loop, members, constants; add 5 hook impls │
 ├───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ include/minimizer/nlsdogleg.hpp       │ Inherit NLSTrustRegionBase; strip everything except compute_step                 │
 ├───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ include/minimizer/subspace2d.hpp      │ Inherit NLSTrustRegionBase; strip everything except compute_step                 │
 ├───────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
 │ include/minimizer/minimizer.hpp       │ Add #include "trustregionbase.hpp" before the three TR headers                   │
 └───────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘

 ---
 What each file looks like after

 nlsdogleg.hpp (~80 lines → ~50 lines):
 #pragma once
 #include "trustregionbase.hpp"

 namespace exprmin {
 enum class DoglegVariant { Standard, Double };

 template <typename System, DoglegVariant DV = DoglegVariant::Standard>
 struct NLSDogleg;

 template <diff::CExpression... RExprs, DoglegVariant DV>
 struct NLSDogleg<diff::Equation<RExprs...>, DV>
     : NLSTrustRegionBase<NLSDogleg<diff::Equation<RExprs...>, DV>, RExprs...> {
   using Base = NLSTrustRegionBase<NLSDogleg<diff::Equation<RExprs...>, DV>, RExprs...>;
   using Base::Base;          // inherit constructor
   using typename Base::ParamVec; using typename Base::NMat; using typename Base::JMat;

   constexpr ParamVec compute_step(const ParamVec &g, const NMat &B, value_type delta);
   // (existing compute_step body, uses this->current_J() from base)
 };
 // deduction guides
 }

 subspace2d.hpp (~230 lines → ~130 lines):
 // same pattern; only compute_step remains (the quartic solver)

 dogleg.hpp (~170 lines → ~110 lines):
 // inherits TrustRegionBase; keeps expr, eval_grad, BFGS/ExactAD, compute_step
 // adds eval_state, eval_trial, adjust_tr, commit_state

 ---
 compute_step and J access

 NLSTrustRegionBase exposes a const JMat& current_J() const accessor so derived
 compute_step implementations can read the current Jacobian without it being a parameter.
 The function signature stays compute_step(g, B, delta) everywhere — the base calls it
 uniformly regardless of whether the derived class uses J or not.

 ---
 Verification

 cmake --build /home/sayan/CLionProjects/ExpressionMinimizer/cmake-build-release --target minimizer_tests
 /home/sayan/CLionProjects/ExpressionMinimizer/cmake-build-release/minimizer_tests
 # All 70 tests must still pass — zero behaviour change
 // same pattern; only compute_step remains (the quartic solver)

 dogleg.hpp (~170 lines → ~110 lines):
 // inherits TrustRegionBase; keeps expr, eval_grad, BFGS/ExactAD, compute_step
 // adds eval_state, eval_trial, adjust_tr, commit_state

 ---
 compute_step and J access

 NLSTrustRegionBase exposes a const JMat& current_J() const accessor so derived
 compute_step implementations can read the current Jacobian without it being a parameter.
 The function signature stays compute_step(g, B, delta) everywhere — the base calls it
 uniformly regardless of whether the derived class uses J or not.

 ---
 Verification

 cmake --build /home/sayan/CLionProjects/ExpressionMinimizer/cmake-build-release --target minimizer_tests
 /home/sayan/CLionProjects/ExpressionMinimizer/cmake-build-release/minimizer_tests
 # All 70 tests must still pass — zero behaviour change