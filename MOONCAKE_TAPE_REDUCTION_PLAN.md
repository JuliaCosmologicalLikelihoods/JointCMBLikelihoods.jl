# Mooncake tape-size reduction plan

## 1. Current status

The complete 79-parameter joint likelihood is differentiable with both
ForwardDiff and Mooncake. A direct prepared Mooncake evaluation gives a finite
gradient, but the reverse-mode tape is unacceptably large:

| Stage | Resident memory | Elapsed time |
|---|---:|---:|
| Data loaded, before preparation | 1.11 GB | — |
| After `prepare_gradient` | 18.35 GB | 194.7 s |
| After prepared `gradient!` | 21.82 GB | 4.09 s |

These numbers were measured for

```julia
f(x) = joint_chi2(data, JointParameters(ntuple(i -> x[i], 79)...))
backend = AutoMooncake()
prep = prepare_gradient(f, backend, x)
gradient!(f, grad, prep, backend, x)
```

on the full Planck PR4 + ACT DR6 + SPT-3G D1 model. The gradient is finite.
The test process is nevertheless killed by the operating system because the
prepared tape consumes most of the available 31 GB before the rest of the test
environment is taken into account. Preparing both the likelihood and posterior
tapes in the same process makes the problem worse.

This memory use is not intrinsic to a likelihood with 79 active scalar
parameters. It comes from tracing large intermediate arrays and repeated
elementwise operations, especially in the ACT chromatic-bandpass model.

## 2. Diagnosis

### 2.1 Dominant source: ACT chromatic SED integration

The current `_act_sed` implementation constructs

```julia
rl[i, j] = bandpass[j] * dBdT(nu[j] + shift) * beam[i, j]
```

for every multipole and bandpass sample, then evaluates two trapezoidal
integrals independently for every multipole. For one ACT map this creates an
`n_ell × n_nu` active matrix with `n_ell = 8502`, followed by thousands of row
views and reductions.

`_act_sed` is called repeatedly:

- five maps for TT dust;
- five maps for TE dust;
- five maps for ET dust, although ET reuses the TE foreground model;
- five maps for EE dust;
- five maps for each of tSZ, CIB, radio, and dusty Poisson in TT.

Thus the tape records roughly forty chromatic SED evaluations, each containing
a large active matrix and row-by-row integration graph.

### 2.2 Repeated full-array foreground broadcasts

ACT TT foreground assembly applies separate broadcasts for dust, tSZ, kSZ,
clustered CIB, tSZ×CIB, radio Poisson, and dusty Poisson. Each operation touches
a `15 × 8502` array. Chaining these operations forces reverse mode to retain
multiple full-array intermediates.

The same pattern exists at smaller scale in the Planck and SPT foreground
builders.

### 2.3 Redundant ACT TE/ET evaluation

The frozen model gives ET the TE foreground parameters. The current joint
boundary nevertheless calls `act_foregrounds(..., :TE)` and
`act_foregrounds(..., :ET)` separately. Their results are identical, so one
complete chromatic foreground computation and its tape are redundant.

### 2.4 Fixed arrays still enter lower-level traces

`PlanckData`, `ACTData`, `SPTData`, and `JointData` are declared as
`Mooncake.NoTangent`. Fixed `SharedTemplates` and `JointCMBTheory` values are
still passed as lower-level arguments, however, and Mooncake can construct
forward-data structures for their arrays. These quantities must never receive
cotangents.

## 3. Exact algebraic rewrite of `_act_sed`

The present chromatic SED is

$$
R_\ell =
\frac{
    \int d\nu\,
    b(\nu)\,g(\nu+\Delta\nu)\,B_\ell(\nu)\,
    r(\nu+\Delta\nu)
}{
    \int d\nu\,
    b(\nu)\,g(\nu+\Delta\nu)\,B_\ell(\nu)
},
$$

where

- $b(\nu)$ is the measured bandpass;
- $g$ is the thermodynamic conversion `dBdT`;
- $B_\ell(\nu)$ is the chromatic beam;
- $r$ is the foreground frequency ratio;
- $\Delta\nu$ is the sampled band-centre shift.

Let $w_j$ be the trapezoidal integration weights of the native bandpass grid.
A uniform shift leaves all grid differences unchanged:

$$
(\nu_{j+1}+\Delta\nu)-(\nu_j+\Delta\nu)
= \nu_{j+1}-\nu_j.
$$

The trapezoidal weights therefore do not depend on the sampled shift. Define

$$
u_j = w_j b_j g(\nu_j+\Delta\nu),
\qquad
v_j = u_j r(\nu_j+\Delta\nu).
$$

Then the complete SED vector is exactly

$$
\boldsymbol z = B\boldsymbol u,
\qquad
\boldsymbol q = B\boldsymbol v,
\qquad
\boldsymbol R = \boldsymbol q \oslash \boldsymbol z.
$$

This replaces the active `8502 × n_nu` matrix `rl`, all row views, and all
row-wise trapezoidal reductions by:

1. two short active vectors of length `n_nu`;
2. two matrix-vector products with the fixed beam matrix;
3. one elementwise division of length 8502.

The rewrite is mathematically identical for nonuniform bandpass grids. It must
be validated against the existing implementation before the old path is
removed.

### Proposed implementation

At ACT-data load time, precompute the trapezoidal weights for each map:

```julia
function trapezoid_weights(nu)
    n = length(nu)
    w = zeros(eltype(nu), n)
    w[1] = (nu[2] - nu[1]) / 2
    for j in 2:n-1
        w[j] = (nu[j+1] - nu[j-1]) / 2
    end
    w[n] = (nu[n] - nu[n-1]) / 2
    return w
end
```

Store either `w` or the fixed product `w .* bandpass` in `ACTData`. The active
kernel then becomes schematically

```julia
nu_shifted = nu .+ shift
u = weighted_bandpass .* _fg_dbdt.(nu_shifted)
z = beam * u
q = beam * (u .* fg_ratio.(nu_shifted))
return q ./ z
```

No approximation or change of convention is introduced.

## 4. Foreground-assembly fusion

After rewriting `_act_sed`, assemble each foreground cross-spectrum in one
fused expression rather than a chain of `.+=` broadcasts.

For ACT TT, each row should have the structure

```julia
@. out[i, :] =
    A_dust * dl_dust * dust_1 * dust_2 +
    A_tsz  * dl_tsz  * tsz_1  * tsz_2  +
    A_ksz  * dl_ksz +
    A_cib  * dl_cib  * cib_1  * cib_2  -
    xi * sqrt(A_cib * A_tsz) * dl_szxcib *
        (tsz_1 * cib_2 + tsz_2 * cib_1) +
    A_radio * dl_ps * radio_1 * radio_2 +
    A_dusty * dl_ps * dusty_1 * dusty_2
```

The exact variables and signs must remain those of the frozen Hillik model.
Only the evaluation graph changes. Planck TT and SPT TT should receive the same
treatment after ACT is validated.

If a fused broadcast still produces an unnecessarily large reverse tape, the
next step is one narrow primitive with an analytical ChainRules pullback. That
primitive should produce one complete foreground row or matrix, not one
component at a time.

## 5. Remove redundant work

At the joint boundary:

```julia
dlfg_te = act_foregrounds(data.act, tpl, fg.shared, fg.act, :TE)
dlfg = Dict(
    :TT => act_foregrounds(data.act, tpl, fg.shared, fg.act, :TT),
    :TE => dlfg_te,
    :ET => dlfg_te,
    :EE => act_foregrounds(data.act, tpl, fg.shared, fg.act, :EE),
)
```

TE and ET may share the same immutable result because downstream code only
reads the foreground arrays. This removes five `_act_sed` calls and one full
foreground matrix from both the forward computation and tape.

Other fixed quantities should be cached where this does not complicate the
model:

- normalized kSZ, CIB, and tSZ×CIB templates for each required multipole grid;
- the fixed $\ell(\ell+1)$ Poisson template;
- ACT trapezoidal weights or weighted bandpasses;
- fixed theory slices when the CMB spectra are constant with respect to the
  nuisance-vector gradient.

The active tSZ tilt and active dust power laws must remain functions of their
sampled parameters.

## 6. Fixed-data tangent declarations

Add the following narrow declarations in the Mooncake extension:

```julia
Mooncake.tangent_type(::Type{SharedTemplates}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{JointCMBTheory}) = Mooncake.NoTangent
```

The same rule applies to any new cache struct containing only release data or
precomputed fixed templates. This is a correctness statement as well as a
performance optimization: nuisance-parameter differentiation must not produce
template, beam, window, covariance, or theory-spectrum tangents.

## 7. Narrow custom pullbacks, if still required

If steps 3–6 do not reduce the tape sufficiently, add analytical pullbacks at
the following boundaries, in this order:

1. `_act_sed` or explicit component-specific ACT SED kernels;
2. `act_residual_vector`;
3. `spt_residual_vector`;
4. `planck_residual_vector`;
5. fused foreground assembly kernels.

The residual-vector pullbacks are comparatively simple because release data,
windows, and covariance information are fixed. For example, for one ACT
spectrum,

$$
\boldsymbol\delta = \boldsymbol d -
\frac{W^T\boldsymbol s}{c},
$$

and an incoming cotangent $\bar{\boldsymbol\delta}$ gives

$$
\bar{\boldsymbol s}
= -\frac{W\bar{\boldsymbol\delta}}{c},
\qquad
\bar c
= \frac{\bar{\boldsymbol\delta}^{T}W^T\boldsymbol s}{c^2}.
$$

The sky cotangent is scattered into the relevant theory and foreground rows;
the fixed window receives `NoTangent()`.

For `_act_sed`, an analytical pullback should contract the incoming 8502-vector
directly into the few active scalars (`shift`, spectral index, and possibly
temperature). It must not construct a dense `8502 × n_parameter` Jacobian.

ChainRules rules belong in `JointCMBLikelihoodsChainRulesCoreExt`; Mooncake
imports them with `@from_chainrules` in `JointCMBLikelihoodsMooncakeExt`.

## 8. Rejected shortcut: hiding ForwardDiff inside a whole-likelihood rule

A custom rule for the complete joint likelihood could evaluate its pullback
with ForwardDiff. This would make the Mooncake tape very small, but it is not
the preferred solution:

- it does not remove the inefficient model graph; it merely hides it from
  Mooncake;
- the resulting “Mooncake” gradient is internally a ForwardDiff gradient;
- ForwardDiff–Mooncake agreement would no longer be an independent validation;
- performance would be bounded by the full 79-dimensional forward-mode pass.

Such a mixed-mode rule may be useful as a temporary diagnostic, but it should
not be the production implementation or the reported Mooncake benchmark.

## 9. Implementation sequence

Proceed incrementally:

1. Add trapezoidal-weight precomputation to `ACTData`.
2. Implement the matrix-vector `_act_sed` formulation alongside the old code.
3. Compare both formulations for every ACT map and all foreground SED types at
   the baseline and several perturbed parameter points.
4. Compare ForwardDiff derivatives of old and new `_act_sed` outputs.
5. Replace the old implementation after exact numerical agreement is shown.
6. Reuse TE for ET.
7. Remeasure prepared Mooncake tape size, preparation time, gradient time, and
   allocations.
8. Fuse ACT foreground assembly and remeasure.
9. Fuse Planck/SPT foreground assembly only if it provides a measurable gain.
10. Add narrow analytical rules only for the remaining dominant tape regions.
11. Run the complete package test suite.
12. Record the final 79-parameter prepared Mooncake benchmark.

Each step should be independently validated before proceeding to the next.

## 10. Validation requirements

### Numerical equivalence

For every changed primal kernel:

- compare against the current frozen-Hillik implementation at the baseline;
- compare at multiple perturbed foreground and instrument points;
- retain all existing frozen fixture tests;
- use the tightest tolerance supported by floating-point reassociation;
- explicitly verify finite outputs.

The matrix-vector SED rewrite should normally agree to near machine precision;
small differences from changed summation order must be quantified rather than
silently tolerated.

### Derivative validation

- All public gradient calls use DifferentiationInterface.
- Use prepared `AutoForwardDiff()` and prepared `AutoMooncake()`.
- Require finite gradients.
- Compare the complete 79-dimensional gradients with `rtol = 1e-3`.
- Retain central finite-difference checks on selected parameters.
- Any custom rrule receives an isolated pullback test before being used in the
  full likelihood.

### Performance measurements

Record at each major step:

- resident memory before and after `prepare_gradient`;
- preparation time;
- steady-state prepared `gradient!` time;
- allocations per prepared gradient;
- forward likelihood time.

The measurement function is the complete nuisance-vector boundary, not an
isolated toy model.

## 11. Acceptance criteria

The work is complete when:

1. all frozen primal reference tests pass;
2. ForwardDiff and Mooncake gradients are finite and agree at `rtol = 1e-3`;
3. selected finite-difference checks pass;
4. `Pkg.test()` completes without an out-of-memory kill;
5. one prepared 79-parameter Mooncake tape fits comfortably in ordinary CI or
   development memory, with substantial headroom rather than barely fitting;
6. the final benchmark reports steady-state prepared gradient time and
   allocations.

A reasonable first target is **below 2 GB peak RSS attributable to the prepared
tape**. If the algebraic rewrite and fusion do not reach that target, proceed
to the narrow analytical pullbacks rather than accepting a multi-gigabyte tape.

---

## Implementation log (2026-09-16)

| Change | Tape (prepare) | Peak | gradient! |
|---|---|---|---|
| baseline (before) | 18.35 GB | 21.82 GB | 4.09 s |
| matvec `_act_sed` + `weighted_bandpass` cache | 5.08 GB | 7.38 GB | 1.67 s |
| `theory_slice` (replace 45k `theory_at` calls) | 5.03 GB | 7.78 GB | 1.69 s |
| `_act_sed_rr` + analytical ChainRules pullback (`@from_chainrules`) | — | — | — |
| fused ACT/SPT assembly broadcasts | 4.85 GB | 6.50 GB | 2.35 s |
| NoTangent for PlanckData/ACTData/SPTData | 4.85 GB | 6.40 GB | — |

**Final 79-param logposterior benchmark (prepared Mooncake, steady state):**
prepare 157 s (one-off), gradient! ~1.2–1.8 s, forward ~45–50 ms, prep
footprint 4.7 GB. Full `Pkg.test()` green: 1367 tests (AD 22, foregrounds
745 incl. 5 new SED-rrule regression tests, quadratic 67, models 120,
instrument 242, priors 171). No OOM kill on the 31 GB machine.

Remaining known tape hotspots if further reduction is ever needed: ACT
assembly ~1.1 GB (fg_TT) + residual 0.77 GB (active-dlfg window
convolution), Planck 0.84 GB, SPT 0.73 GB. A fused whole-likelihood rrule
(CamSpec pattern) is the next lever, not needed for current memory headroom.

Per-experiment (final): Planck 843 MB, ACT 3782 MB, SPT 730 MB.
Component bisect: fg_TT 1102 MB, fg_TE 77 MB, residual(active dlfg) 770 MB, spt_sky 263 MB.

Notes:
- The dominant fix was the matvec `_act_sed` rewrite (trapz weights are
  shift-invariant; `z = B u`, `q = B (u .* r)`); the per-row views/trapz
  reductions were the bulk of the 18 GB.
- `_act_sed_rr` pullback verified against ForwardDiff to ~1e-15 on all 5 SED
  kinds (beta, T, shift active). Primal matches frozen `_act_sed` closures at
  rtol 1e-12 across all 25 map/kind combos incl. nonzero shifts.
- Assembly fusion (one broadcast per pair instead of chained `.+=`) cut fg_TT
  1551→1102 MB and fg_TE 246→77 MB.
- TE/ET reuse (`dlfg[:ET] = dlfg[:TE]`) in `joint_residuals`.
- FD spot check in `test_ad.jl` now uses `_planck_chi2_smooth` (the float64
  path AD differentiates): the frozen primal quantizes chi2 at ~1e-4, which
  corrupted central differences for small-gradient Planck directions
  (PLK_Adust143TT was the failing case).

### Complete SPT residual pullback (2026-09-17)

The public `SPTInstrumentParameters` API now packs into a private flat
22-element `_spt_residual_flat` boundary. Its ChainRules pullback reverses the
complete released SPT instrument model analytically: fixed windows,
calibrations, TE/EE leakage, T2P kernels, polarized beams, and temperature-beam
eigenmodes. Mooncake imports this one rule rather than recording the complete
SPT array graph.

Validation:

- arbitrary-cotangent VJP against ForwardDiff, exercising all 21 sky blocks
  and all 22 flat instrument entries;
- all 20 frozen SPT parameter points;
- full 79-parameter ForwardDiff/Mooncake likelihood and posterior gradients;
- full `Pkg.test()`: 1370 tests passing.

Full 79-parameter logposterior benchmark (BenchmarkTools, five prepared
gradient samples):

| Measurement | Result |
|---|---:|
| Forward minimum / median | 44.2 / 48.4 ms |
| Cold Mooncake preparation | 188.2 s |
| RSS increase across preparation | 2.31 GiB |
| Prepared `gradient!` minimum / median | 269 / 430 ms |
| Prepared-gradient allocation | 1.13 GB |

The steady-state minimum fell from roughly 0.95--1.4 s to 0.269 s. The
remaining cost and allocation are outside the SPT instrument boundary; a
further push toward 0.1 s requires attacking the remaining ACT/Planck generic
graphs rather than adding more local SPT rules.

### Complete ACT and Planck residual pullbacks (2026-09-17)

The public dictionary APIs now delegate to private concrete-array boundaries:

- `_act_residual_flat` takes the three theory vectors, three foreground
  matrices, and released calibration vector. Its pullback reverses all 41
  fixed windows, scatters into the appropriate theory and foreground rows,
  reuses TE for ET, and differentiates each calibration division.
- `_planck_residual_flat` takes the four 15x2501 model matrices and four
  calibration vectors. Its pullback reverses lite-bin selection, fixed
  cross-frequency weighted averaging, the combined TE+ET ratio, and the
  per-cross-spectrum calibration divisions.

Both rules return `NoTangent()` for release data and are imported by Mooncake
with concrete `@from_chainrules` signatures. An attempted removal of the
active dictionaries from `joint_residuals` produced no measurable benchmark
improvement and was reverted rather than retaining gratuitous churn.

Validation:

- arbitrary-cotangent ACT VJP against ForwardDiff, independently exercising
  all three theory modes, all 45 foreground rows, and all 41 calibrations;
- arbitrary-cotangent Planck VJP against ForwardDiff, independently exercising
  all 60 model rows and all 60 calibration entries;
- all frozen ACT and Planck boundary fixtures;
- full 79-parameter ForwardDiff/Mooncake likelihood and posterior gradients;
- full `Pkg.test()`: 1374 tests passing.

Full logposterior benchmark after the ACT/Planck rules (BenchmarkTools, seven
prepared gradient samples):

| Measurement | Result |
|---|---:|
| Forward minimum / median | 65.6 / 74.9 ms |
| Forward allocation | 78.7 MB |
| Cold Mooncake preparation | 179.2 s |
| RSS increase across preparation | 1.92 GiB |
| Prepared `gradient!` minimum / median | 225 / 729 ms |
| Prepared-gradient allocation | 980 MB |

Compared with the SPT-only state, the best prepared gradient improved from
269 ms to 225 ms and the preparation RSS delta fell from 2.31 GiB to 1.92 GiB.
The noisy median is driven by garbage collection of roughly 1 GB allocated per
call. Remaining performance work is in the foreground construction graphs,
especially ACT TT, not the three residual boundaries.

### Complete foreground pullbacks (2026-09-17)

The remaining foreground graphs now cross analytical flat-vector boundaries:

- `_planck_foregrounds_flat` packs the nine shared and 24 Planck foreground
  scalars. Its mode-specific pullback contracts each 15x2501 cotangent through
  dust, tSZ, kSZ, clustered CIB, tSZxCIB, radio, and dusty Poisson terms.
- `_act_foregrounds_flat` packs the nine shared and 17 ACT foreground scalars.
  Its pullback works on only the 15 released cross pairs, reverses all seven TT
  components and polarized dust, and reuses the exact `_act_sed_rr` pullback
  for the five beam-chromatic bandpass SEDs. The primal returns its SED/template
  cache to the rule, avoiding an otherwise expensive recomputation in reverse.
- `_spt_sky_flat` packs the nine shared and 14 SPT foreground scalars. Its
  pullback contracts all 21 pre-instrument sky blocks through SSL, aberration,
  and every foreground component.
- `_fg_powerlaw`, `_fg_tsz_template`, and `_act_sed_rr` have scalar analytical
  pullbacks. The ACT SED rule contracts the 8502-element cotangent through the
  fixed beam product and differentiates only the short bandpass vectors and
  scalar beta, temperature, and band shift.

The public struct-based APIs and all frozen numerical conventions are
unchanged. The analytical rules are private implementation boundaries imported
by Mooncake through concrete `@from_chainrules` signatures.

Validation:

- arbitrary-cotangent VJPs against ForwardDiff for all Planck modes, all ACT
  modes and SED kinds, and the complete 23-parameter SPT sky;
- direct angular-shape pullbacks against ForwardDiff;
- all frozen foreground, residual, and quadratic-form fixtures;
- full 79-parameter ForwardDiff/Mooncake likelihood and posterior gradients;
- final `Pkg.test()`: 1397 tests passing, including 768 foreground tests.

Prepared component-gradient minima before and after the whole foreground
boundaries were:

| Component | Before | After | Allocation after | Preparation RSS after |
|---|---:|---:|---:|---:|
| Planck foregrounds | 23.3 ms | 6.94 ms | 9.98 MB | 0.313 GiB |
| ACT foregrounds | 40.9 ms | 28.6 ms | 57.4 MB | 0.280 GiB |
| SPT sky | 26.5 ms | 2.33 ms | 14.0 MB | 0.305 GiB |

An isolated `_act_cross_pair_slice` rule reduced ACT preparation RSS but made
the prepared component gradient worse (40.9 to 58.0 ms) and was reverted. The
complete ACT rule initially had the same recomputation problem (47.5 ms); using
the primal SED/template cache reduced it to 28.6 ms.

Final full 79-parameter logposterior benchmark (BenchmarkTools, seven prepared
gradient samples):

| Measurement | Result |
|---|---:|
| Forward minimum / median | 51.9 / 70.8 ms |
| Forward allocation | 78.7 MB |
| Cold Mooncake preparation | 142.1 s |
| RSS increase across preparation | 0.324 GiB |
| Prepared `gradient!` minimum / median | 113 / 153 ms |
| Prepared-gradient allocation | 163 MB |

Relative to the ACT/Planck-residual state, the prepared minimum fell from
225 ms to 113 ms, gradient allocation from 980 MB to 163 MB, and preparation
RSS from 1.92 GiB to 0.324 GiB. Relative to the original generic Mooncake path,
the prepared gradient is roughly an order of magnitude faster. The practical
0.1-second target is now within about 13 ms; further work is no longer dominated
by one large foreground tape.
