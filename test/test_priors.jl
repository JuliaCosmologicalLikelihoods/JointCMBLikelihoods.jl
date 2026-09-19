# Prior layer + full parameter pipeline: JointParameters (79 sampled) ->
# foreground/instrument structs -> residual vectors -> chi2, validated against
# the frozen Hillik fixtures at every exported parameter point.

using NPZ
using JSON

@testset "Prior layer" begin
    tags = ["baseline", "common_fg", "plk_cal", "plk_dust", "plk_ps", "act_cal",
        "act_band", "act_fg", "spt_cal", "spt_beam", "spt_leak", "spt_fg",
        "spt_kappa", "multi1", "multi2", "multi3", "multi4", "multi5",
        "boundary_low", "boundary_high"]

    function joint_params(P)
        return JointParameters(
            # shared FG (10)
            P["Acib"], P["Atsz"], P["Aksz"], P["xi"], P["beta_cib"], P["beta_radio"],
            P["PLK_beta_dustTT"], P["PLK_beta_dustTE"], P["PLK_beta_dustEE"],
            P["PLK_alpha_dustTT"],
            # Planck FG (11)
            P["PLK_Adust100TT"], P["PLK_Adust143TT"], P["PLK_Adust217TT"],
            P["PLK_Adust100TE"], P["PLK_Adust143TE"], P["PLK_Adust217TE"],
            P["PLK_Adust100EE"], P["PLK_Adust143EE"], P["PLK_Adust217EE"],
            P["PLK_radio_TT"], P["PLK_cib_ps"],
            # ACT FG (5)
            P["ACT_AdustTT"], P["ACT_AdustTE"], P["ACT_AdustEE"],
            P["ACT_radio_TT"], P["ACT_cib_ps"],
            # SPT FG (5)
            P["SPT3G_AdustTT"], P["SPT3G_AdustTE"], P["SPT3G_AdustEE"],
            P["SPT3G_radio_TT"], P["SPT3G_cib_ps"],
            # Planck instrument (12)
            P["A_planck"], P["PLK_cal_100A"], P["PLK_cal_100B"], P["PLK_cal_143B"],
            P["PLK_cal_217A"], P["PLK_cal_217B"], P["PLK_pe_100A"], P["PLK_pe_100B"],
            P["PLK_pe_143A"], P["PLK_pe_143B"], P["PLK_pe_217A"], P["PLK_pe_217B"],
            # ACT instrument (14)
            P["ACT_cal"], P["ACT_cal_dr6_pa4_f220"], P["ACT_cal_dr6_pa5_f090"],
            P["ACT_cal_dr6_pa6_f090"], P["ACT_cal_dr6_pa6_f150"],
            P["ACT_pe_dr6_pa5_f090"], P["ACT_pe_dr6_pa5_f150"],
            P["ACT_pe_dr6_pa6_f090"], P["ACT_pe_dr6_pa6_f150"],
            P["ACT_band_shift_dr6_pa4_f220"], P["ACT_band_shift_dr6_pa5_f090"],
            P["ACT_band_shift_dr6_pa5_f150"], P["ACT_band_shift_dr6_pa6_f090"],
            P["ACT_band_shift_dr6_pa6_f150"],
            # SPT instrument (22)
            P["SPT3G_cal"], P["SPT3G_cal_90"], P["SPT3G_cal_220"],
            P["SPT3G_pe_90"], P["SPT3G_pe_150"], P["SPT3G_pe_220"],
            (P["SPT3G_beta_$i"] for i in 1:9)...,
            P["SPT3G_beta_pol_90"], P["SPT3G_beta_pol_150"], P["SPT3G_beta_pol_220"],
            P["SPT3G_T2P2_90"], P["SPT3G_T2P2_150"], P["SPT3G_T2P2_220"],
            P["SPT3G_kappa"])
    end

    # ------------------------------------------------------------------
    # logprior: finite inside support, -Inf outside
    # ------------------------------------------------------------------
    @testset "logprior support" begin
        P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_baseline.json"))
        p = joint_params(P)
        @test n_sampled(p) == 79
        @test isfinite(logprior(p))

        # rebuild a JointParameters with one field replaced (struct is immutable)
        function with_field(p::JointParameters, field::Symbol, value)
            nt = NamedTuple{fieldnames(JointParameters)}(getfield.(Ref(p), fieldnames(JointParameters)))
            return JointParameters(merge(nt, (field => value,))...)
        end

        # each bounded parameter pushed outside its support -> -Inf
        for (field, value) in (
            (:Acib, -1.0), (:Acib, 51.0),
            (:Atsz, -1.0), (:Atsz, 51.0),
            (:Aksz, -1.0), (:Aksz, 51.0),
            (:xi, -1.5), (:xi, 1.5),
            (:beta_cib, 0.5), (:beta_cib, 3.5),
            (:beta_radio, -2.0), (:beta_radio, 0.5),
            (:alpha_dust_tt, -3.5), (:alpha_dust_tt, -1.5),
            (:PLK_Adust100TT, -1.0),
            (:PLK_Adust100TE, -1.0),
            (:PLK_Adust100EE, 11.0),
            (:PLK_radio_TT, 151.0), (:PLK_cib_ps, -1.0),
            (:ACT_radio_TT, 21.0), (:ACT_cib_ps, -1.0),
            (:SPT_radio_TT, 11.0), (:SPT_cib_ps, -1.0),
            (:ACT_cal, 0.5), (:ACT_cal, 1.4),
            (:SPT_cal, 0.5), (:SPT_cal, 1.4),
            (:PLK_pe_100A, 0.7),
            (:ACT_pe_pa5_f090, 1.3),
            (:SPT_pe_90, 0.7),
            (:SPT_beta_pol_90, -0.1),
            (:SPT_beta_pol_220, 1.1),
        )
            q = with_field(p, field, value)
            @test logprior(q) == -Inf
        end

        # Gaussian priors: shift from the mean changes logprior by the exact quadratic
        # (baseline A_planck = 1.001 already contributes its own quadratic term)
        q = with_field(p, :A_planck, 1.01)
        lp0 = logprior(p)
        expected = lp0 - 0.5 * ((1.01 - 1.0) / 0.0025)^2 +
                   0.5 * ((1.001 - 1.0) / 0.0025)^2
        @test logprior(q) ≈ expected atol = 1e-10
    end

    # ------------------------------------------------------------------
    # Shared-dust propagation (paper joint model): beta_dust_tt/te/ee and
    # alpha_dust_tt propagate to all three experiments; ET reuses TE.
    # ------------------------------------------------------------------
    @testset "shared dust propagation" begin
        P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_baseline.json"))
        p = joint_params(P)
        fg = foreground_parameters(p)
        @test fg.planck.beta_dust == (p.beta_dust_tt, p.beta_dust_te, p.beta_dust_te, p.beta_dust_ee)
        @test fg.act.beta_dustTT == p.beta_dust_tt
        @test fg.act.beta_dustTE == p.beta_dust_te
        @test fg.act.beta_dustEE == p.beta_dust_ee
        @test fg.act.alpha_dustTT == p.alpha_dust_tt
        @test fg.spt.beta_dustTT == p.beta_dust_tt
        @test fg.spt.alpha_dustTT == p.alpha_dust_tt
        @test fg.shared.beta_dusty == p.beta_cib
        # fixed quantities
        @test fg.planck.alpha_dustTE == -2.4
        @test fg.planck.alpha_dustEE == -2.4
        @test fg.shared.T_cib == 25.0
        @test fg.shared.alpha_tsz == 0.0
        ins = instrument_parameters(p)
        @test ins.planck.cal[3] == 1.0        # PLK 143A reference
        @test ins.act.cal_map[3] == 1.0      # ACT pa5_f150 reference
        @test ins.act.pe_map[1] == 1.0       # ACT pa4_f220 no polarisation
        @test ins.spt.cal_freq[2] == 1.0     # SPT 150 reference
    end

    # ------------------------------------------------------------------
    # Full pipeline: 79 params -> residuals -> chi2, vs fixtures
    # ------------------------------------------------------------------
    @testset "full parameter pipeline" begin
        pd = load_planck_data(PLANCK_DATA_DIR)
        ad = load_act_data(ACT_DATA_DIR)
        sd = load_spt_data(SPT_DATA_DIR)
        tpl = load_shared_templates(get(ENV, "JOINT_FG_TEMPLATES", artifact"joint_fg_templates"))
        th = npzread(joinpath(FIXTURE_DIR, "theory.npz"))
        theory = JointCMBTheory(collect(0:9050), vec(th["dlTT"]), vec(th["dlTE"]), vec(th["dlEE"]))

        # NOTE: the frozen fixtures were exported with PER-EXPERIMENT dust
        # spectral indices (the frozen yamls fix them per survey), while the
        # paper's joint model samples them shared. To reproduce the fixtures
        # numerically we build the per-experiment structs directly from the
        # fixture JSON; the shared propagation is tested separately above.
        for tag in tags
            P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_$(tag).json"))
            p = joint_params(P)
            shared = SharedForegroundParameters(
                P["Atsz"], P["Acib"], P["Aksz"], P["xi"], P["beta_cib"],
                P["beta_radio"], P["beta_dusty"], P["T_cib"], P["alpha_tsz"])
            plk_fg = PlanckForegroundParameters(
                ntuple(i -> P["PLK_Adust$(["100","143","217"][(i-1)%3+1])$(["TT","TE","ET","EE"][(i-1)÷3+1])"], 12),
                (P["PLK_beta_dustTT"], P["PLK_beta_dustTE"], P["PLK_beta_dustET"], P["PLK_beta_dustEE"]),
                (P["PLK_alpha_dust100TT"], P["PLK_alpha_dust143TT"], P["PLK_alpha_dust217TT"]),
                P["PLK_alpha_dustTE"], P["PLK_alpha_dustET"], P["PLK_alpha_dustEE"],
                P["PLK_radio_TT"], P["PLK_cib_ps"])
            act_fg = ACTForegroundParameters(
                P["ACT_AdustTT"], P["ACT_AdustTE"], P["ACT_AdustEE"],
                P["ACT_beta_dustTT"], P["ACT_beta_dustTE"], P["ACT_beta_dustEE"],
                P["ACT_alpha_dustTT"], P["ACT_alpha_dustTE"], P["ACT_alpha_dustEE"],
                P["ACT_radio_TT"], P["ACT_radio_EE"], P["ACT_cib_ps"],
                Dict(m => P["ACT_band_shift_$(m)"] for m in
                    ("dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150",
                     "dr6_pa6_f090", "dr6_pa6_f150")))
            spt_fg = SPTForegroundParameters(
                P["SPT3G_AdustTT"], P["SPT3G_AdustTE"], P["SPT3G_AdustEE"],
                P["SPT3G_beta_dustTT"], P["SPT3G_beta_dustTE"], P["SPT3G_beta_dustEE"],
                P["SPT3G_alpha_dustTT"], P["SPT3G_alpha_dustTE"], P["SPT3G_alpha_dustEE"],
                P["SPT3G_radio_TT"], P["SPT3G_radio_TE"], P["SPT3G_radio_EE"],
                P["SPT3G_cib_ps"], P["SPT3G_kappa"])
            ins = instrument_parameters(p)

            # Planck
            dplk = npzread(joinpath(FIXTURE_DIR, "planck_$(tag).npz"))
            dlmodel = Dict{Symbol,Matrix{Float64}}()
            cal = Dict{Symbol,Vector{Float64}}()
            for mode in (:TT, :EE, :TE, :ET)
                dlmodel[mode] = planck_foregrounds(tpl, shared, plk_fg, mode)
                cal[mode] = planck_calibration(ins.planck, mode)
            end
            # add theory inside the model boundary: dlmodel is fg-only in the
            # frozen convention for TE/ET; rebuild with theory added
            for mode in (:TT, :EE, :TE, :ET)
                m = mode in (:TT, :EE, :TE) ? mode : :TE
                dlmodel[mode] .+= [theory_at(theory, m, l) for l in 0:2500]'
            end
            delta_plk = planck_residual_vector(pd, dlmodel, cal)
            @test delta_plk ≈ Float64.(dplk["delta_dl"]) rtol = 1e-6

            # ACT
            dact = npzread(joinpath(FIXTURE_DIR, "act_$(tag).npz"))
            dlfg = Dict(:TT => act_foregrounds(ad, tpl, shared, act_fg, :TT),
                        :TE => act_foregrounds(ad, tpl, shared, act_fg, :TE),
                        :ET => act_foregrounds(ad, tpl, shared, act_fg, :ET),
                        :EE => act_foregrounds(ad, tpl, shared, act_fg, :EE))
            dlth = Dict(:TT => [theory_at(theory, :TT, l) for l in 0:8501],
                        :TE => [theory_at(theory, :TE, l) for l in 0:8501],
                        :EE => [theory_at(theory, :EE, l) for l in 0:8501])
            cal_act = act_calibration(ins.act, ad)
            delta_act = act_residual_vector(ad, dlth, dlfg, cal_act)
            @test delta_act ≈ Float64.(dact["delta_dl"]) rtol = 1e-6

            # SPT
            dspt = npzread(joinpath(FIXTURE_DIR, "spt_$(tag).npz"))
            sky = spt_sky_model(sd, tpl, theory, shared, spt_fg)
            delta_spt = spt_residual_vector(sd, sky, ins.spt)
            @test delta_spt ≈ Float64.(dspt["delta_dl"]) rtol = 1e-6

            # joint chi2 matches the frozen export
            c = chi2_components(pd, ad, sd, delta_plk, delta_act, delta_spt)
            @test c.planck ≈ Float64(dplk["chi2"]) rtol = 1e-6
            @test c.act ≈ Float64(dact["chi2"]) rtol = 1e-6
            @test c.spt ≈ Float64(dspt["chi2"]) rtol = 1e-6
        end
    end
end
