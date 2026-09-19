module JointCMBLikelihoodsMooncakeExt

using JointCMBLikelihoods: PlanckData, ACTData, SPTData, JointData,
    SharedTemplates, JointCMBTheory,
    planck_chi2, act_chi2, spt_chi2,
    _act_window_project, _spt_window_project,
    _act_residual_flat, _planck_residual_flat,
    _act_sed_rr,
    _fg_powerlaw, _fg_tsz_template,
    _spt_sky_flat,
    _planck_foregrounds_flat,
    _act_foregrounds_flat,
    _np_gradient, _spt_temperature_beam, _spt_polarization_beam, _spt_t2p_kernels,
    _spt_residual_flat,
    joint_residuals, joint_chi2, joint_loglikelihood, joint_logposterior
using Mooncake: @from_chainrules, MinimalCtx
import Mooncake

# Release data is never a differentiable quantity: without these, Mooncake
# builds (and re-zeroes every gradient call) full-matrix tangents for the
# covariances and tangent structs for every fixed-data field.
Mooncake.tangent_type(::Type{JointData}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{PlanckData}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{ACTData}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{SPTData}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{SharedTemplates}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{JointCMBTheory}) = Mooncake.NoTangent

@from_chainrules MinimalCtx Tuple{
    typeof(planck_chi2), PlanckData, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(act_chi2), ACTData, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(spt_chi2), SPTData, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(_act_window_project), ACTData, Int, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(_spt_window_project), SPTData, Int, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(_act_sed_rr), ACTData, String, Symbol,
    Float64, Float64, Float64,
}
@from_chainrules MinimalCtx Tuple{typeof(_fg_powerlaw), Float64, Int, Int}
@from_chainrules MinimalCtx Tuple{
    typeof(_fg_tsz_template), SharedTemplates, Int, Float64,
}
@from_chainrules MinimalCtx Tuple{
    typeof(_spt_sky_flat), SPTData, SharedTemplates,
    JointCMBTheory{Float64}, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(_planck_foregrounds_flat), SharedTemplates, Symbol, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(_act_foregrounds_flat), ACTData, SharedTemplates, Symbol, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(_act_residual_flat), ACTData,
    Vector{Float64}, Vector{Float64}, Vector{Float64},
    Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(_planck_residual_flat), PlanckData,
    Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64},
    Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{
    typeof(_spt_residual_flat), SPTData, Vector{Float64}, Vector{Float64},
}
@from_chainrules MinimalCtx Tuple{typeof(_np_gradient), Vector{Float64}}
@from_chainrules MinimalCtx Tuple{typeof(_spt_temperature_beam), SPTData, Vector{Float64}}
@from_chainrules MinimalCtx Tuple{typeof(_spt_polarization_beam), SPTData, Matrix{Float64}, Vector{Float64}}
@from_chainrules MinimalCtx Tuple{typeof(_spt_t2p_kernels), SPTData, Vector{Float64}}

end
