# Quadratic forms reproducing the frozen Hillik conventions (d0e455cf).
#
# Target of the first implementation slice: given externally supplied residual
# vectors, reproduce the three frozen χ² values exactly. Foreground modelling
# comes later; the boundaries here take pre-assembled model vectors.

export JointData, joint_residuals, joint_chi2, joint_loglikelihood, joint_logposterior

"""
    planck_chi2(data::PlanckData, delta::AbstractVector{<:Real})

Planck quadratic form with the frozen Hillik conventions:
`delta` is the 4872-element residual (TT 1646 + EE 1466 + TE/ET 1760),
cast to Float32 before the product, χ² rounded to 8 significant digits.

The Float32 cast and the 8-digit rounding are frozen numerical conventions,
not differentiable structure. Under AD the primal keeps the frozen
conventions and the pullback differentiates the float64 quadratic form.
"""
function planck_chi2(data::PlanckData, delta::AbstractVector{<:Real})
    length(delta) == 4872 || throw(DimensionMismatch("Planck delta must have 4872 elements"))
    if eltype(delta) <: Union{Float32,Float64}
        # frozen primal path: float32 cast + 8-digit rounding (scipy dsymv
        # semantics: float32 matrix entries, float64 accumulation)
        d32 = Vector{Float32}(delta)
        chi2 = dot(d32, data.inv_cov64, d32)
        alpha = 8 - ceil(log10(chi2))
        return round(chi2 * 10.0^alpha) * 10.0^(-alpha)
    else
        return _planck_chi2_smooth(data, delta)
    end
end

"""
    _planck_chi2_smooth(data, delta)

The differentiable Planck quadratic form: the plain float64 product on the
float64 copy of the float32 inv_cov. This is what AD differentiates — the
float32 cast and 8-digit rounding of the frozen primal are numerical
conventions, not differentiable structure. Finite-difference checks of the
gradient must use this path: the frozen primal is quantized at the ~1e-4
level, which corrupts central differences for small-gradient directions.
"""
_planck_chi2_smooth(data::PlanckData, delta::AbstractVector{<:Real}) =
    dot(delta, data.inv_cov64, delta)

"""
    act_chi2(data::ACTData, delta::AbstractVector{<:Real})

ACT quadratic form: 1139-element residual, full float64 product.
"""
function act_chi2(data::ACTData, delta::AbstractVector{<:Real})
    length(delta) == 1139 || throw(DimensionMismatch("ACT delta must have 1139 elements"))
    return dot(delta, data.inv_cov, delta)
end

"""
    spt_chi2(data::SPTData, delta::AbstractVector{<:Real})

SPT quadratic form: 1392-element residual, full float64 product
(frozen code uses `delta @ inv_bpcov @ delta` in float64).

Uses the inverse covariance precomputed at load. Evaluating `data.cov \\ delta`
here refactorized the covariance on every call; see
[`_spt_inverse_covariance`](@ref).
"""
function spt_chi2(data::SPTData, delta::AbstractVector{<:Real})
    length(delta) == 1392 || throw(DimensionMismatch("SPT delta must have 1392 elements"))
    return dot(delta, data.inv_cov, delta)
end

"""
    chi2_components(planck, act, spt, delta_planck, delta_act, delta_spt)

Named tuple of the three experiment χ² values.
"""
function chi2_components(pd::PlanckData, ad::ACTData, sd::SPTData,
                         delta_planck, delta_act, delta_spt)
    return (
        planck = planck_chi2(pd, delta_planck),
        act = act_chi2(ad, delta_act),
        spt = spt_chi2(sd, delta_spt),
    )
end

function chi2(pd::PlanckData, ad::ACTData, sd::SPTData,
             delta_planck, delta_act, delta_spt)
    c = chi2_components(pd, ad, sd, delta_planck, delta_act, delta_spt)
    return c.planck + c.act + c.spt
end

"""
    loglikelihood(...)

Frozen Hillik convention: loglike = -χ²/2 (no normalization constants).
"""
function loglikelihood(pd::PlanckData, ad::ACTData, sd::SPTData,
                      delta_planck, delta_act, delta_spt)
    return -0.5 * chi2(pd, ad, sd, delta_planck, delta_act, delta_spt)
end

# ---------------------------------------------------------------------------
# Joint boundary: 79 sampled parameters -> residuals -> χ².
# Single entry point for AD (ForwardDiff/Mooncake) and inference.
# ---------------------------------------------------------------------------

"""
    JointData(pd, ad, sd, tpl, theory)

Bundle of the fixed inputs of the joint likelihood: the three experiment
data structs, the shared foreground templates, and the CMB theory spectra.
"""
struct JointData
    planck::PlanckData
    act::ACTData
    spt::SPTData
    templates::SharedTemplates
    theory::JointCMBTheory
end

"""
    joint_residuals(data::JointData, p::JointParameters) -> NamedTuple

Full forward model: expand the 79 parameters into foreground and instrument
structs, evaluate the three experiment model vectors, and return the three
residual vectors `(planck, act, spt)`.
"""
function joint_residuals(data::JointData, p::JointParameters)
    fg = foreground_parameters(p)
    ins = instrument_parameters(p)
    tpl = data.templates
    theory = data.theory

    # --- Planck: fg + theory per cross-spectrum, then calibration ---
    dlmodel = Dict{Symbol,Matrix{eltype(p.A_planck)}}()
    cal = Dict{Symbol,Vector{eltype(p.A_planck)}}()
    for mode in (:TT, :EE, :TE, :ET)
        dlmodel[mode] = planck_foregrounds(tpl, fg.shared, fg.planck, mode)
        cal[mode] = planck_calibration(ins.planck, mode)
    end
    for mode in (:TT, :EE, :TE, :ET)
        m = mode in (:TT, :EE, :TE) ? mode : :TE
        dlmodel[mode] .+= theory_slice(theory, m, 0, 2500)'
    end
    delta_planck = planck_residual_vector(data.planck, dlmodel, cal)

    # --- ACT: fg + theory per pol, then windows + calibration ---
    dlfg_te = act_foregrounds(data.act, tpl, fg.shared, fg.act, :TE)
    dlfg = Dict(:TT => act_foregrounds(data.act, tpl, fg.shared, fg.act, :TT),
                :TE => dlfg_te,
                :ET => dlfg_te,   # ET reuses the TE foreground model (frozen)
                :EE => act_foregrounds(data.act, tpl, fg.shared, fg.act, :EE))
    dlth = Dict(:TT => theory_slice(theory, :TT, 0, 8501),
                :TE => theory_slice(theory, :TE, 0, 8501),
                :EE => theory_slice(theory, :EE, 0, 8501))
    cal_act = act_calibration(ins.act, data.act)
    delta_act = act_residual_vector(data.act, dlth, dlfg, cal_act)

    # --- SPT: pre-instrument sky, then instrument + windows ---
    sky = spt_sky_model(data.spt, tpl, theory, fg.shared, fg.spt)
    delta_spt = spt_residual_vector(data.spt, sky, ins.spt)

    return (planck = delta_planck, act = delta_act, spt = delta_spt)
end

"""
    joint_chi2(data::JointData, p::JointParameters) -> Real

Joint data χ² (sum of the three experiment quadratic forms). Priors are NOT
included — see `joint_logposterior`.
"""
function joint_chi2(data::JointData, p::JointParameters)
    d = joint_residuals(data, p)
    return chi2(data.planck, data.act, data.spt, d.planck, d.act, d.spt)
end

"""
    joint_loglikelihood(data::JointData, p::JointParameters)

Frozen Hillik convention: -χ²/2, no normalization constants.
"""
joint_loglikelihood(data::JointData, p::JointParameters) = -0.5 * joint_chi2(data, p)

"""
    joint_logposterior(data::JointData, p::JointParameters)

`loglikelihood + logprior`. Bounded-prior violations return -Inf.
"""
function joint_logposterior(data::JointData, p::JointParameters)
    lp = logprior(p)
    isfinite(lp) || return lp
    return lp + joint_loglikelihood(data, p)
end
