# The Turing extension. Each experiment contributes its own `~`, so these tests
# check what that buys over a single `@addlogprob!`: each distribution's logpdf
# is this package's own quadratic form, the three contributions are separately
# visible to DynamicPPL, and the log joint is exactly priors plus likelihood.
#
# The densities here are unnormalized (-chi2/2), the frozen Hillik convention.
# That is deliberate and differs from the CamSpec extension, which normalizes:
# this package stores precision matrices with no factors, so the per-experiment
# constants would mean factorizing three operators at load. These tests pin the
# unnormalized identity exactly, so the choice cannot drift silently.

using NPZ
using JSON
using Turing
using Distributions
using Random
using ADTypes: AutoForwardDiff

const TURING_EXT = Base.get_extension(JointCMBLikelihoods, :JointCMBLikelihoodsTuringExt)
const FIXTURE_DIR_TURING = get(ENV, "JOINT_CMB_FIXTURES", artifact"joint_cmb_fixtures")

@testset "JointCMB Turing extension" begin
    @testset "loaded" begin
        @test TURING_EXT !== nothing
    end

    pd = load_planck_data(PLANCK_DATA_DIR)
    ad = load_act_data(ACT_DATA_DIR)
    sd = load_spt_data(SPT_DATA_DIR)
    tpl = load_shared_templates(get(ENV, "JOINT_FG_TEMPLATES", artifact"joint_fg_templates"))
    th = npzread(joinpath(FIXTURE_DIR_TURING, "theory.npz"))
    theory = JointCMBTheory(collect(0:9050), vec(th["dlTT"]), vec(th["dlTE"]), vec(th["dlEE"]))
    data = JointData(pd, ad, sd, tpl, theory)

    P = JSON.parsefile(joinpath(FIXTURE_DIR_TURING, "params_baseline.json"))
    p0 = JointParameters(
        P["Acib"], P["Atsz"], P["Aksz"], P["xi"], P["beta_cib"], P["beta_radio"],
        P["PLK_beta_dustTT"], P["PLK_beta_dustTE"], P["PLK_beta_dustEE"],
        P["PLK_alpha_dustTT"],
        P["PLK_Adust100TT"], P["PLK_Adust143TT"], P["PLK_Adust217TT"],
        P["PLK_Adust100TE"], P["PLK_Adust143TE"], P["PLK_Adust217TE"],
        P["PLK_Adust100EE"], P["PLK_Adust143EE"], P["PLK_Adust217EE"],
        P["PLK_radio_TT"], P["PLK_cib_ps"],
        P["ACT_AdustTT"], P["ACT_AdustTE"], P["ACT_AdustEE"],
        P["ACT_radio_TT"], P["ACT_cib_ps"],
        P["SPT3G_AdustTT"], P["SPT3G_AdustTE"], P["SPT3G_AdustEE"],
        P["SPT3G_radio_TT"], P["SPT3G_cib_ps"],
        P["A_planck"], P["PLK_cal_100A"], P["PLK_cal_100B"], P["PLK_cal_143B"],
        P["PLK_cal_217A"], P["PLK_cal_217B"], P["PLK_pe_100A"], P["PLK_pe_100B"],
        P["PLK_pe_143A"], P["PLK_pe_143B"], P["PLK_pe_217A"], P["PLK_pe_217B"],
        P["ACT_cal"], P["ACT_cal_dr6_pa4_f220"], P["ACT_cal_dr6_pa5_f090"],
        P["ACT_cal_dr6_pa6_f090"], P["ACT_cal_dr6_pa6_f150"],
        P["ACT_pe_dr6_pa5_f090"], P["ACT_pe_dr6_pa5_f150"],
        P["ACT_pe_dr6_pa6_f090"], P["ACT_pe_dr6_pa6_f150"],
        P["ACT_band_shift_dr6_pa4_f220"], P["ACT_band_shift_dr6_pa5_f090"],
        P["ACT_band_shift_dr6_pa5_f150"], P["ACT_band_shift_dr6_pa6_f090"],
        P["ACT_band_shift_dr6_pa6_f150"],
        P["SPT3G_cal"], P["SPT3G_cal_90"], P["SPT3G_cal_220"],
        P["SPT3G_pe_90"], P["SPT3G_pe_150"], P["SPT3G_pe_220"],
        (P["SPT3G_beta_$i"] for i in 1:9)...,
        P["SPT3G_beta_pol_90"], P["SPT3G_beta_pol_150"], P["SPT3G_beta_pol_220"],
        P["SPT3G_T2P2_90"], P["SPT3G_T2P2_150"], P["SPT3G_T2P2_220"],
        P["SPT3G_kappa"])

    names = TURING_EXT.JOINT_PARAMETER_NAMES
    baseline = NamedTuple{names}(getfield.(Ref(p0), names))
    residuals = joint_residuals(data, p0)

    @testset "parameter names come from the struct" begin
        # Taken from fieldnames(JointParameters) rather than restated, so the
        # model cannot drift from the struct.
        @test names == fieldnames(JointParameters)
        @test length(names) == 79
        @test n_sampled(p0) == 79
    end

    @testset "logpdf is the package quadratic form, unnormalized" begin
        distributions = (
            (TURING_EXT.PlanckBandpowers(pd, pd.data_vector .- residuals.planck),
             pd.data_vector, planck_chi2(pd, residuals.planck), 4872),
            (TURING_EXT.ACTBandpowers(ad, ad.data_vector .- residuals.act),
             ad.data_vector, act_chi2(ad, residuals.act), 1139),
            (TURING_EXT.SPTBandpowers(sd, sd.data_vector .- residuals.spt),
             sd.data_vector, spt_chi2(sd, residuals.spt), 1392),
        )
        for (d, observed, chi2_value, n) in distributions
            @test length(d) == n
            # Exactly -chi2/2, with no normalization: the frozen Hillik
            # convention. An added constant here would be a silent change of
            # convention that every posterior ratio would hide.
            @test logpdf(d, observed) == -chi2_value / 2

            # The distribution must not ignore its argument and echo the mean.
            shifted = observed .+ 1.0
            @test isfinite(logpdf(d, shifted))
            @test logpdf(d, shifted) != logpdf(d, observed)

            # Sampling is refused: this package stores the inverse covariance
            # and no factor of the covariance, so a draw would mean an n^3
            # factorization hidden inside `rand`.
            @test_throws ArgumentError rand(Random.MersenneTwister(20260920), d)
        end

        @test_throws DimensionMismatch TURING_EXT.PlanckBandpowers(pd, pd.data_vector[1:10])
        @test_throws DimensionMismatch TURING_EXT.ACTBandpowers(ad, ad.data_vector[1:10])
        @test_throws DimensionMismatch TURING_EXT.SPTBandpowers(sd, sd.data_vector[1:10])
    end

    # A scaffold, not a scientific prior. This package's authoritative prior is
    # `logprior`, a hand-written function; the extension deliberately ships no
    # default so that no second source of truth exists. Tight Gaussians centred
    # on the baseline keep the forward model in its valid region.
    priors = NamedTuple{names}(ntuple(i -> Normal(baseline[i], 0.01), length(names)))

    @testset "check_joint_priors demands all 79" begin
        @test TURING_EXT.check_joint_priors(priors) === nothing
        incomplete = Base.structdiff(priors, NamedTuple{(names[1], names[2])})
        @test_throws ArgumentError TURING_EXT.check_joint_priors(incomplete)
    end

    @testset "log joint is priors plus likelihood" begin
        model = TURING_EXT.joint_cmb_model(data, priors)

        likelihood_term = -0.5 * (planck_chi2(pd, residuals.planck) +
                                  act_chi2(ad, residuals.act) +
                                  spt_chi2(sd, residuals.spt))
        @test likelihood_term ≈ joint_loglikelihood(data, p0) rtol=1e-12

        prior_term = sum(logpdf(priors[n], baseline[n]) for n in names)

        @test Turing.DynamicPPL.logjoint(model, baseline) ≈
            prior_term + likelihood_term rtol=1e-12
        @test Turing.DynamicPPL.loglikelihood(model, baseline) ≈
            likelihood_term rtol=1e-12
        @test isfinite(Turing.DynamicPPL.logjoint(model, baseline))
    end

    @testset "the three experiments are separately attributable" begin
        # The reason for three `~` statements rather than one `@addlogprob!`:
        # DynamicPPL can report which experiment contributed what. With a single
        # injected number this decomposition does not exist.
        model = TURING_EXT.joint_cmb_model(data, priors)
        pointwise = Turing.DynamicPPL.pointwise_loglikelihoods(
            model, Turing.DynamicPPL.InitFromParams(baseline))
        contributions = sort(collect(Iterators.flatten(values(pointwise))))
        expected = sort([-planck_chi2(pd, residuals.planck) / 2,
                         -act_chi2(ad, residuals.act) / 2,
                         -spt_chi2(sd, residuals.spt) / 2])

        # Three observations, one per experiment, each carrying exactly that
        # experiment's -chi2/2 and nothing of the other two.
        @test length(contributions) == 3
        @test contributions ≈ expected rtol=1e-12
    end

    @testset "the model is differentiable" begin
        # Establishes that the `~` formulation stays on the registered chi2
        # rules and that a gradient-based sampler can run. The gradient itself
        # is checked against Mooncake in test_ad.jl; NUTS is not run here
        # because 79 ForwardDiff duals through the full joint forward model
        # would dominate the suite for nothing this does not already show.
        model = TURING_EXT.joint_cmb_model(data, priors)
        density = Turing.DynamicPPL.LogDensityFunction(
            model, Turing.DynamicPPL.getlogjoint, Turing.DynamicPPL.VarInfo(model);
            adtype=AutoForwardDiff())
        x0 = collect(Float64, baseline)
        value, gradient = Turing.LogDensityProblems.logdensity_and_gradient(density, x0)
        @test isfinite(value)
        @test length(gradient) == 79
        @test all(isfinite, gradient)
        @test !all(iszero, gradient)
    end
end
