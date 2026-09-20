"""
    JointCMBLikelihoodsTuringExt

Turing.jl support for the joint Planck PR4 + ACT DR6 + SPT-3G D1 likelihood.

Loaded automatically when `JointCMBLikelihoods`, `Turing` and `Distributions`
are all present. Each experiment contributes its own `~` statement:

```julia
data.planck.data_vector ~ PlanckBandpowers(data.planck, planck_model)
data.act.data_vector    ~ ACTBandpowers(data.act,       act_model)
data.spt.data_vector    ~ SPTBandpowers(data.spt,       spt_model)
```

rather than one `@addlogprob!` for the sum. Three statements keep the
per-experiment contributions visible to DynamicPPL, so a decomposition of the
log density tells you which experiment moved.

Each `logpdf` calls this package's own quadratic form, so reverse mode stays on
the registered `planck_chi2` / `act_chi2` / `spt_chi2` rules and the prepared
tape stays the size the reduction campaign made it.

## These densities are unnormalized, on purpose

`logpdf` returns `-χ²/2`, the frozen Hillik convention this package implements
throughout — see `joint_loglikelihood`. A normalized Gaussian density would add
`-N/2·log(2π) - ½·logdet Σ` per experiment, and this package stores precision
matrices with no factors, so obtaining those constants would mean factorizing
three operators at load. The omitted term is **fixed**: it cancels in every
posterior ratio, every gradient and every MCMC result. It does mean
`logpdf` here is not comparable to a density from another package.
"""
module JointCMBLikelihoodsTuringExt

using JointCMBLikelihoods
using Turing
using Distributions
using LinearAlgebra
using Random

import JointCMBLikelihoods: planck_chi2, act_chi2, spt_chi2, loglikelihood

export PlanckBandpowers, ACTBandpowers, SPTBandpowers

for (T, D, F, N) in ((:PlanckBandpowers, :PlanckData, :planck_chi2, 4872),
                     (:ACTBandpowers, :ACTData, :act_chi2, 1139),
                     (:SPTBandpowers, :SPTData, :spt_chi2, 1392))
    @eval begin
        """
            $($T)(data, model)

        The $($(string(D))) bandpower likelihood as a multivariate
        distribution with mean `model`, evaluated through this package's own
        quadratic form. Unnormalized: see the module docstring.
        """
        struct $T{D<:$D,M<:AbstractVector} <: ContinuousMultivariateDistribution
            data::D
            model::M

            function $T(data::$D, model::AbstractVector)
                length(model) == $N || throw(DimensionMismatch(
                    "$($(string(T))) model must have $($N) entries, got $(length(model))",
                ))
                return new{typeof(data), typeof(model)}(data, model)
            end
        end

        Base.length(::$T) = $N
        Base.eltype(::Type{<:$T{<:Any,M}}) where {M} = eltype(M)

        function Distributions._logpdf(d::$T, x::AbstractVector{<:Real})
            return -$F(d.data, x .- d.model) / 2
        end

        # The fixed data is a precision matrix with no stored factor, so there
        # is no cheap square root to draw with. Refuse rather than hide an
        # n^3 factorization inside `rand`.
        function Distributions._rand!(::Random.AbstractRNG, ::$T,
                                      ::AbstractVector{<:Real})
            throw(ArgumentError(
                "$($(string(T))) cannot be sampled from: this package stores " *
                "the inverse covariance and no factor of the covariance.",
            ))
        end
    end
end

"""
    JOINT_PARAMETER_NAMES

The 79 parameter names, taken from `fieldnames(JointParameters)` so the model
cannot drift from the struct.
"""
const JOINT_PARAMETER_NAMES = fieldnames(JointParameters)

@eval @model function joint_cmb_model(data::JointData, priors::NamedTuple)
    $(Expr(:block, [:($(name) ~ priors.$(name)) for name in JOINT_PARAMETER_NAMES]...))
    parameters = JointParameters(promote($(JOINT_PARAMETER_NAMES...))...)

    residuals = joint_residuals(data, parameters)
    # The mean is defined from the residual, so evaluating at the observed
    # vector reproduces exactly the residual this package computed, whichever
    # sign convention `joint_residuals` uses.
    data.planck.data_vector ~ PlanckBandpowers(data.planck,
                                               data.planck.data_vector .- residuals.planck)
    data.act.data_vector ~ ACTBandpowers(data.act,
                                         data.act.data_vector .- residuals.act)
    data.spt.data_vector ~ SPTBandpowers(data.spt,
                                         data.spt.data_vector .- residuals.spt)

    return parameters
end

@doc """
    joint_cmb_model(data, priors)

Turing model for the joint likelihood: the 79 nuisance parameters as named `~`
statements, then one `~` per experiment.

`priors` is a `NamedTuple` with one `Distribution` for every field of
`JointParameters`; it is checked for completeness at construction.

**No default priors are supplied, deliberately.** This package's authoritative
prior is `logprior(p)`, a hand-written function with inline bounds and Gaussian
terms rather than a table, so any set of `Distribution`s is a *restatement* of
it and a second source of truth that can silently disagree. If you build one,
validate it: `sum(logpdf(priors[n], getfield(p, n)) for n in JOINT_PARAMETER_NAMES)`
must differ from `logprior(p)` by the same constant at every point inside the
support.

```julia
priors = (; Acib = Uniform(0, 50), Atsz = Uniform(0, 50), ...)
chain  = sample(joint_cmb_model(data, priors), NUTS(), 1000;
                initial_params = start)
```
"""
joint_cmb_model

"""
    check_joint_priors(priors)

Throw unless `priors` supplies a distribution for all 79 parameters.
"""
function check_joint_priors(priors::NamedTuple)
    missing_names = [n for n in JOINT_PARAMETER_NAMES if !haskey(priors, n)]
    isempty(missing_names) || throw(ArgumentError(
        "priors missing for $(length(missing_names)) parameters: " *
        join(first(missing_names, 8), ", ") * (length(missing_names) > 8 ? ", ..." : ""),
    ))
    return nothing
end

end # module
