# Scientific contract

This is the common convention sheet. The formulas below are project derivations to verify and explain, not new empirical results. Use calculus, sign diagrams and elementary algebra in the main exposition; explain any optional advanced concepts separately.

## 1. Variables, units and equations

`N>=0` is population, `t` time, `r>0` inverse time, `K>A>0` population levels, `H>=0` population/time and `h>=0` inverse time. Write `g(N)=r N (1-N/K)(N/A-1)`.

Use `x=N/K`, `tau=r t`, `a=A/K in (0,1)`, `q=H/(rK)`, `e=h/r`:

- Logistic baseline: `dx/dtau = x(1-x)-q`.
- A: `dx/dtau = G(x)-q`, `G(x)=x(1-x)(x/a-1)`.
- B: `dx/dtau = G(x)-e x`.

Inverse inference with unknown `r` or `K` uses fixed external scales `t_ref,N_ref` recorded in the config, and the chain rule. Normalising by true hidden `K` or `r` leaks information; optimising them also changes the scaling unless formulated explicitly.

## 2. Thresholds and equilibria

`G(x)=-x^3/a+(1+a)x^2/a-x` and `G'(x)=-3x^2/a+2(1+a)x/a-1`.

For A:

`x_star=(1+a+sqrt(1-a+a^2))/3`, `q_c=G(x_star)`, `N_star=K*x_star`, `H_c=rK*q_c`.

At `0<q<q_c`, two positive equilibria are in `(a,1)`. The smaller `x_minus` is unstable and the larger `x_plus` stable. There is also a negative polynomial root, excluded from the population domain. At `q=q_c`, the positive double root is **semistable**: attraction from above, departure toward extinction from below. `f'=0` alone is inconclusive; inspect signs. At `q>q_c` there is no positive equilibrium. At `q=0`, the roots are `0,a,1`, with stable/unstable/stable signs.

For B:

`e_c=(1-a)^2/(4a)`, `h_c=r*(K-A)^2/(4AK)`.

`x_plus/minus=(1+a +/- sqrt((1-a)^2-4ae))/2` when `e<=e_c`. Zero is always stable for `e>=0` because `f'(0)=-1-e`. For `e<e_c`, the lower positive root is unstable and the upper positive root stable. The double root `(1+a)/2` is semistable at the threshold. For `e>e_c`, zero is the only nonnegative equilibrium. For `e=0`, recover `0,a,1`.

Logistic constant yield has `q_c=1/4` or `H_c=rK/4`, roots `(1 +/- sqrt(1-4q))/2` below threshold. For `q=0`, zero is unstable in the logistic model, unlike strong Allee growth.

Use a numerical root/sign residual tolerance stated in each output; do not silently merge near-critical distinct roots. No positive equilibrium count should include an absorbing boundary convention as a root of A.

## 3. Extinction and basins

For A with `q>0`, `f(0)=-q`: integrate until the first downward zero crossing. On a monotonically declining path with no intervening equilibrium,

`tau_ext = integral_0^x0 [1/(q-G(u))] du`, `t_ext=tau_ext/r`.

Use this only when the path actually reaches zero. Below `q_c`, positive starts below `x_minus` hit zero in finite time; starts at an equilibrium stay there; starts above `x_minus` tend to `x_plus`. At `q_c`, starts below `x_star` go extinct, starts at the root stay, and starts above approach it asymptotically. Above `q_c`, any finite positive start eventually hits zero, possibly after a long bottleneck. A short solver horizon does not prove survival.

At `q=0`, positive starts below `a` approach zero asymptotically; **no finite extinction claim**. At `x0=0,q>0`, the original vector field immediately leaves the physical domain; the solver records initial extinction and an explicit boundary policy.

For B, exact zero cannot be reached from a positive start in finite time under this smooth ODE. Report time to `x<=epsilon` separately, with epsilon and solver tolerance. Below `e_c`, the lower positive root divides extinction and persistence basins. Exact initial roots require explicit classification rather than numerical drift.

An absorbing zero continuation after A's event is a modified physical policy (harvest stops once extinct), not an equilibrium of the original constant-yield equation. Never train/evaluate the unmodified residual on that continuation.

## 4. Harvesting history and recovery

Use a piecewise smooth or piecewise constant ramp `0 -> above critical -> 0`, starting near the populated branch. Carry the previous terminal population into the next segment and record ramp speed/dwell times. Do not silently reset it at the turning point.

Under nonnegative effort B, zero stays stable even when effort returns to zero; a population below the Allee threshold does not recover on its own. A uses the declared extinction/harvest-shutdown convention. Thus the intended result is path dependence, collapse and a recovery barrier. A separate **restocking** run explicitly jumps to a stated population above the current unstable root. A branch plot with independent equilibrated initialisations is not itself a dynamical recovery experiment. Call a closed hysteresis loop only if an explicitly changed model/control actually produces one.

Optional slowing down: for a stable positive equilibrium use `T_relax=-1/f'(N_eq)` in dimensional units, then check small perturbations numerically. Specify the perturbation/tolerance definition for a measured recovery time, exclude the zero-eigenvalue fold, and separate bottleneck passage after collapse from local relaxation before it.

## 5. Fair comparisons, yield and sensitivity

`H` and `h` have different units. Compare either (i) equal initial realised harvest `H=h*N_ref`, with the same reference state and its limitations, or (ii) equal fractions `H/H_c` and `h/h_c`, clearly answering a different question. Same `r,K` alone does not make logistic and Allee production curves equivalent: the extra factor can raise high-density growth. No universal statement that A/B/Allee is always riskier follows without a specified metric, initial state and calibration.

Optional MSY: the constant-yield supremum `H_c` lies at a semistable fold, with no robust safety margin. For B, equilibrium realised yield is `Y(h)=h*N_plus(h)`. Its interior maximum corresponds to the same production-maximising `N_star`, at `h_MSY=g(N_star)/N_star < h_c`. Do not confuse maximum effort `h_c` with maximum yield. Evaluate a policy below the cliff and report both harvest and basin margins (for example `1-H/H_c`, `N0-N_minus`).

Sensitivity: dimensional partial derivatives hold the other dimensional parameters fixed; an `a` sweep at fixed `K,r` is a different experiment. State which, compare to central finite differences away from degenerate limits, and optionally use elasticities `(p/threshold)*d(threshold)/dp`.

## 6. Inverse identifiability before optimisation

Expanding B gives `dN/dt = c1*N+c2*N^2+c3*N^3`, where

`c1=-(r+h)`, `c2=r(1/A+1/K)`, `c3=-r/(AK)`.

Population trajectories at one fixed effort reveal at most these three combinations, not four independent values. An explicit counterexample is `(r,A,K,h)=(1,0.2,1,0.5)` and `(1.35,0.3,0.9,0.15)`: both give `c1=-1.5,c2=6,c3=-5`. Additional initial conditions with the same coefficients do not remove this ambiguity.

**Main B experiment:** fix an independently declared `K`, estimate `r,A,h`, or start with a smaller identifiable subset. **Main A experiment:** start with known `K`, estimate `r,A,H`, then test all four `r,A,K,H` as a clearly labelled extension with practical conditioning checks. For A the polynomial has constant coefficient `-H`, linear coefficient `-r`, and the quadratic/cubic coefficients determine `A+K` and `AK`; the ordering `A<K` removes the interchange ambiguity in ideal data. Structural uniqueness does not guarantee useful recovery from short noisy trajectories near equilibrium.

An optional different B design may use multiple experiments with **known distinct effort multipliers** and common parameters, with a fresh identifiability argument. Merely adding trajectories at the same unknown h is insufficient.

## 7. PINN evaluation contract

- Forward loss uses initial condition and physics collocation points; disclose any supervised observations (that changes the experiment). Autodiff residual: `dx_dtau - f(x)`, differentiating with respect to physical time (no detaching of trainable parameters).
- Compare with independently generated, tighter-tolerance RK45/DOP853 reference points not used as supervised training labels. Evaluate A only up to its event; report event-time estimation separately if implemented.
- Report RMSE, MAE, global relative L2 error and absolute error near zero. Pointwise percentage error is unstable near extinction and is not a primary metric.
- Inverse runs fit only training observations. Validation selects hyperparameters/stopping; final test observations/time ranges remain untouched until freezing. Keep evaluation seeds independent of fitting noise seeds.
- At minimum use noise standard deviations `sigma=0,0.01,0.05` times the declared fixed population reference scale, and seeds `11,22,33`. Use the same saved observations/configs for PINN and a SciPy nonlinear least-squares ODE baseline.
- Hold out a contiguous late-time window per trajectory (default 60/20/20 split) and label interpolation vs temporal extrapolation. If physics collocation includes the held-out horizon, call this physics-supported prediction, not unseen-domain extrapolation. A new initial-condition test of shared estimated parameters is useful separate evidence.
- Include regular, near-fold and subthreshold/extinction regimes. Show failures, divergence, boundary violations, parameter error and train/validation losses. Do not promise a target PINN accuracy; choose budgets/tolerances before comparing.
