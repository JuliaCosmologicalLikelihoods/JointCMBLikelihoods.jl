# Phase 5 — automatic differentiation through the full joint model.
# ForwardDiff vs Mooncake (prepared, via DifferentiationInterface), finite
# primal and gradient, and an independent finite-difference spot check.

const FIXTURE_DIR_AD = get(ENV, "JOINT_CMB_FIXTURES", artifact"joint_cmb_fixtures")

@testset "AD" begin
    using NPZ
    using JSON
    using DifferentiationInterface
    using ADTypes: AutoForwardDiff, AutoMooncake
    using ForwardDiff
    using Mooncake
    using ChainRulesCore
    using JSON
    import JointCMBLikelihoods: _planck_chi2_smooth

    pd = load_planck_data(PLANCK_DATA_DIR)
    ad = load_act_data(ACT_DATA_DIR)
    sd = load_spt_data(SPT_DATA_DIR)
    tpl = load_shared_templates(get(ENV, "JOINT_FG_TEMPLATES", artifact"joint_fg_templates"))
    th = npzread(joinpath(FIXTURE_DIR_AD, "theory.npz"))
    theory = JointCMBTheory(collect(0:9050), vec(th["dlTT"]), vec(th["dlTE"]), vec(th["dlEE"]))
    data = JointData(pd, ad, sd, tpl, theory)

    # baseline params from the fixture JSON (paper joint model: shared betas)
    P = JSON.parsefile(joinpath(FIXTURE_DIR_AD, "params_baseline.json"))
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

    # vector boundary: x <-> JointParameters round trip
    fields = fieldnames(JointParameters)
    to_vec(p) = [getfield(p, f) for f in fields]
    from_vec(x) = JointParameters(ntuple(i -> x[i], length(fields))...)

    x0 = to_vec(p0)
    @test length(x0) == 79
    @test from_vec(x0) isa JointParameters{Float64}

    # ------------------------------------------------------------------
    # primal: finite, and the vector boundary matches the struct boundary
    # ------------------------------------------------------------------
    f_chi2(x) = joint_chi2(data, from_vec(x))
    f_lp(x) = joint_logposterior(data, from_vec(x))

    c0 = f_chi2(x0)
    @test isfinite(c0)
    @test f_chi2(x0) ≈ joint_chi2(data, p0) rtol = 1e-12
    lp0 = f_lp(x0)
    @test isfinite(lp0)
    @test f_lp(x0) ≈ logprior(p0) + joint_loglikelihood(data, p0) rtol = 1e-12

    # ------------------------------------------------------------------
    # prepared ForwardDiff vs prepared Mooncake, full 79-dim gradient
    # ------------------------------------------------------------------
    backend_fd = AutoForwardDiff()
    backend_mc = AutoMooncake()

    prep_fd = prepare_gradient(f_chi2, backend_fd, x0)
    g_fd = similar(x0)
    gradient!(f_chi2, g_fd, prep_fd, backend_fd, x0)
    @test all(isfinite, g_fd)

    # Mooncake tapes are ~18 GB each — scope them in let blocks so each is
    # freed before the next one is built (31 GB machine)
    g_mc = let
        prep_mc = prepare_gradient(f_chi2, backend_mc, x0)
        g = similar(x0)
        gradient!(f_chi2, g, prep_mc, backend_mc, x0)
        g
    end
    GC.gc()
    @test all(isfinite, g_mc)

    @test isapprox(g_fd, g_mc, rtol = 1e-3)

    # logposterior gradient (prior adds exact quadratic terms)
    prep_lp_fd = prepare_gradient(f_lp, backend_fd, x0)
    glp_fd = similar(x0)
    gradient!(f_lp, glp_fd, prep_lp_fd, backend_fd, x0)
    @test all(isfinite, glp_fd)
    glp_mc = let
        prep_lp_mc = prepare_gradient(f_lp, backend_mc, x0)
        g = similar(x0)
        gradient!(f_lp, g, prep_lp_mc, backend_mc, x0)
        g
    end
    GC.gc()
    @test all(isfinite, glp_mc)
    @test isapprox(glp_fd, glp_mc, rtol = 1e-3)

    # ------------------------------------------------------------------
    # independent finite-difference check on selected directions
    # ------------------------------------------------------------------
    # The AD gradient differentiates the smooth float64 Planck quadratic form
    # (_planck_chi2_smooth): the frozen primal's float32 cast + 8-digit
    # rounding quantize chi2 at the ~1e-4 level, which corrupts central
    # differences for small-gradient Planck directions. The FD check therefore
    # uses the same smooth path that AD differentiates.
    f_chi2_smooth(x) = begin
        d = joint_residuals(data, from_vec(x))
        return _planck_chi2_smooth(data.planck, d.planck) +
               act_chi2(data.act, d.act) + spt_chi2(data.spt, d.spt)
    end
    # Step size: 1e-4, not 1e-5. All three quadratic forms now contract a
    # precomputed inverse with a threaded BLAS symmetric product, which leaves a
    # couple of ulp of reduction noise in chi2 (~1e-12 relative on chi2 ~ 9163).
    # That is irrelevant scientifically and fatal to a central difference whose
    # numerator is ~1e-5: SPT_beta_7 (index 70) has |df/dx| ~ 0.64, so at
    # h = 1e-5 the FD estimate is noise-dominated and lands ~4e-4 off the
    # analytic value. Measured convergence for that direction against the
    # ForwardDiff gradient -0.6401286:
    #
    #     h = 1e-5 -> -0.6405229   (6e-4 off, noise-dominated)
    #     h = 1e-4 -> -0.6400466   (1.3e-4 off)
    #     h = 1e-3 -> -0.6401374   (1.4e-5 off, but truncation breaks SPT_cal_220)
    #
    # 1e-4 is the step at which all ten directions pass with margin.
    for i in (1, 5, 12, 25, 33, 40, 50, 60, 70, 79)   # Atsz, beta_radio, dust, radio_TT, A_planck, cal, band, SPT cal, T2P, kappa
        h = 1e-4 * max(abs(x0[i]), 1.0)
        xp = copy(x0); xp[i] += h
        xm = copy(x0); xm[i] -= h
        fd = (f_chi2_smooth(xp) - f_chi2_smooth(xm)) / (2h)
        @test isapprox(g_fd[i], fd, rtol = 1e-3, atol = 1e-3)
    end
end
