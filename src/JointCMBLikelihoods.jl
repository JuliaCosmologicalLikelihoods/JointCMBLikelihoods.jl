module JointCMBLikelihoods

using LinearAlgebra
using NPZ
using JSON
using Artifacts
using CMBForegrounds: window_convolution, _fixed_beam_product, _fixed_chromatic_ratio,
    assemble_TT, assemble_TE, assemble_EE

export JointCMBTheory,
    theory_at,
    theory_slice,
    PlanckData,
    ACTData,
    SPTData,
    SharedTemplates,
    load_planck_data,
    load_act_data,
    load_spt_data,
    load_shared_templates,
    planck_chi2,
    planck_residual_vector,
    act_residual_vector,
    spt_residual_vector,
    SPTInstrumentParameters,
    PlanckInstrumentParameters,
    ACTInstrumentParameters,
    n_sampled,
    planck_calibration,
    act_calibration,
    JointParameters,
    logprior,
    foreground_parameters,
    instrument_parameters,
    act_chi2,
    spt_chi2,
    chi2_components,
    chi2,
    loglikelihood

include("theory.jl")
include("data.jl")
include("foregrounds.jl")
include("planck_model.jl")
include("act_model.jl")
include("spt_model.jl")
include("instrument.jl")
include("priors.jl")
include("likelihood.jl")

end # module JointCMBLikelihoods
