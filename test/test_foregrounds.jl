# Shared foreground model: per-component / per-block validation against the
# frozen Hillik fixtures at every exported parameter point.

using NPZ
using JSON
using LinearAlgebra
using DifferentiationInterface
using ForwardDiff
using ADTypes: AutoMooncake, AutoForwardDiff
using ChainRulesCore: rrule
import JointCMBLikelihoods: _act_sed_rr, _fg_powerlaw, _fg_tsz_template,
    _spt_sky_flat, _spt_sky_parameters
import JointCMBLikelihoods: _planck_foregrounds_flat, _planck_foreground_parameters
import JointCMBLikelihoods: _act_foregrounds_flat, _act_foreground_parameters

const FG_TEMPLATE_DIR = get(ENV, "JOINT_FG_TEMPLATES", artifact"joint_fg_templates")

@testset "Foreground model" begin
    tpl = load_shared_templates(FG_TEMPLATE_DIR)

    tags = ["baseline", "common_fg", "plk_cal", "plk_dust", "plk_ps", "act_cal",
        "act_band", "act_fg", "spt_cal", "spt_beam", "spt_leak", "spt_fg",
        "spt_kappa", "multi1", "multi2", "multi3", "multi4", "multi5",
        "boundary_low", "boundary_high"]

    @testset "angular-shape pullbacks" begin
        output_bar_power = [sin(0.021i) for i in 1:8502]
        alpha = -0.6
        power_objective(a) = dot(output_bar_power, _fg_powerlaw(a, 8501, 500))
        power_fd = ForwardDiff.derivative(power_objective, alpha)
        _, power_pullback = rrule(_fg_powerlaw, alpha, 8501, 500)
        @test power_pullback(output_bar_power)[2] ≈ power_fd rtol = 1e-11

        output_bar_tsz = [cos(0.017i) for i in 1:2501]
        tilt = 0.13
        tsz_objective(a) = dot(output_bar_tsz, _fg_tsz_template(tpl, 2500, a))
        tsz_fd = ForwardDiff.derivative(tsz_objective, tilt)
        _, tsz_pullback = rrule(_fg_tsz_template, tpl, 2500, tilt)
        @test tsz_pullback(output_bar_tsz)[4] ≈ tsz_fd rtol = 1e-11
    end

    # ------------------------------------------------------------------
    # Planck: per-component foreground spectra
    # ------------------------------------------------------------------
    @testset "Planck components" begin
        pd = load_planck_data(PLANCK_DATA_DIR)
        for tag in tags
            d = npzread(joinpath(FIXTURE_DIR, "planck_$(tag).npz"))
            P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_$(tag).json"))
            shared = SharedForegroundParameters(
                P["Atsz"], P["Acib"], P["Aksz"], P["xi"], P["beta_cib"],
                P["beta_radio"], P["beta_dusty"], P["T_cib"], P["alpha_tsz"])
            plk = PlanckForegroundParameters(
                ntuple(i -> P["PLK_Adust$(["100","143","217"][(i-1)%3+1])$(["TT","TE","ET","EE"][(i-1)÷3+1])"], 12),
                (P["PLK_beta_dustTT"], P["PLK_beta_dustTE"], P["PLK_beta_dustET"], P["PLK_beta_dustEE"]),
                (P["PLK_alpha_dust100TT"], P["PLK_alpha_dust143TT"], P["PLK_alpha_dust217TT"]),
                P["PLK_alpha_dustTE"], P["PLK_alpha_dustET"], P["PLK_alpha_dustEE"],
                P["PLK_radio_TT"], P["PLK_cib_ps"])
            for mode in (:TE, :ET, :EE)
                fg = planck_foregrounds(tpl, shared, plk, mode)
                @test size(fg) == (15, 2501)
                @test fg ≈ Float64.(d["fg0_Dust_$(mode)"]) rtol = 1e-8
            end
            # TT: planck_foregrounds returns all 7 components summed
            fgTT = planck_foregrounds(tpl, shared, plk, :TT)
            total = Float64.(d["fg0_Dust_TT"]) .+ Float64.(d["fg1_tSZ_TT"]) .+
                Float64.(d["fg2_kSZ_TT"]) .+ Float64.(d["fg3_clustered CIB_TT"]) .+
                Float64.(d["fg4_SZxCIB_TT"]) .+ Float64.(d["fg5_PS radio_TT"]) .+
                Float64.(d["fg6_PS dusty_TT"])
            @test fgTT ≈ total rtol = 1e-8
            if tag == "baseline"
                x_planck = _planck_foreground_parameters(shared, plk)
                for (mode_index, mode) in enumerate((:TT, :TE, :ET, :EE))
                    output = planck_foregrounds(tpl, shared, plk, mode)
                    output_bar = [sin(0.017r + 0.0031l + 0.23mode_index)
                                  for r in 1:15, l in 1:2501]
                    objective(x) = dot(output_bar, _planck_foregrounds_flat(tpl, mode, x))
                    gradient_fd = ForwardDiff.gradient(objective, x_planck)
                    _, pullback = rrule(_planck_foregrounds_flat, tpl, mode, x_planck)
                    gradient_rule = pullback(output_bar)[4]
                    @test all(isfinite, gradient_rule)
                    @test gradient_rule ≈ gradient_fd rtol = 1e-9 atol = 1e-8
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # ACT: per-mode dlfg (all components summed)
    # ------------------------------------------------------------------
    @testset "ACT dlfg" begin
        ad = load_act_data(ACT_DATA_DIR)
        for tag in tags
            d = npzread(joinpath(FIXTURE_DIR, "act_$(tag).npz"))
            P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_$(tag).json"))
            shared = SharedForegroundParameters(
                P["Atsz"], P["Acib"], P["Aksz"], P["xi"], P["beta_cib"],
                P["beta_radio"], P["beta_dusty"], P["T_cib"], P["alpha_tsz"])
            act = ACTForegroundParameters(
                P["ACT_AdustTT"], P["ACT_AdustTE"], P["ACT_AdustEE"],
                P["ACT_beta_dustTT"], P["ACT_beta_dustTE"], P["ACT_beta_dustEE"],
                P["ACT_alpha_dustTT"], P["ACT_alpha_dustTE"], P["ACT_alpha_dustEE"],
                P["ACT_radio_TT"], P["ACT_radio_EE"], P["ACT_cib_ps"],
                Dict(m => P["ACT_band_shift_$(m)"] for m in
                    ("dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150",
                     "dr6_pa6_f090", "dr6_pa6_f150")))
            for mode in (:TT, :TE, :ET, :EE)
                fg = act_foregrounds(ad, tpl, shared, act, mode)
                @test size(fg) == (15, 8502)
                @test fg ≈ Float64.(d["dlfg_$(mode)"]) rtol = 1e-8
            end
            if tag == "baseline"
                x_act = _act_foreground_parameters(shared, act)
                for (mode_index, mode) in enumerate((:TT, :TE, :EE))
                    output = act_foregrounds(ad, tpl, shared, act, mode)
                    output_bar = [cos(0.019r + 0.0017l + 0.31mode_index)
                                  for r in 1:15, l in 1:8502]
                    objective(x) = dot(output_bar, _act_foregrounds_flat(ad, tpl, mode, x))
                    gradient_fd = ForwardDiff.gradient(objective, x_act)
                    _, pullback = rrule(_act_foregrounds_flat, ad, tpl, mode, x_act)
                    gradient_rule = pullback(output_bar)[5]
                    @test all(isfinite, gradient_rule)
                    @test gradient_rule ≈ gradient_fd rtol = 1e-8 atol = 1e-7
                end
            end
        end

        # chromatic SED path: Mooncake through CMBForegrounds' fixed-beam
        # primitives must match ForwardDiff on all five SED kinds
        @testset "ACT chromatic SED AD" begin
            ad = load_act_data(ACT_DATA_DIR)
            w = randn(8502)
            for (kind, beta, T, shift) in ((:dust, 1.51, 0.0, 0.3), (:tsz, 0.0, 0.0, -0.2),
                                          (:cib, 1.7, 25.0, 0.1), (:radio, -2.5, 0.0, 0.05),
                                          (:dusty, 1.7, 25.0, 0.0))
                f(x) = dot(_act_sed_rr(ad, "dr6_pa5_f090", kind, x[1], x[2], x[3]), w)
                x = [beta, T, shift]
                g_mc = DifferentiationInterface.gradient(f, AutoMooncake(), x)
                g_fd = DifferentiationInterface.gradient(f, AutoForwardDiff(), x)
                @test g_mc ≈ g_fd rtol = 1e-8
                _, pullback = rrule(
                    _act_sed_rr, ad, "dr6_pa5_f090", kind, beta, T, shift)
                tangents = pullback(w)
                @test [tangents[5], tangents[6], tangents[7]] ≈ g_fd rtol = 1e-10 atol = 1e-10
            end
        end

    end

    # ------------------------------------------------------------------
    # SPT: pre-instrument sky model (21 blocks)
    # ------------------------------------------------------------------
    @testset "SPT sky model" begin
        sd = load_spt_data(SPT_DATA_DIR)
        th = npzread(joinpath(FIXTURE_DIR, "theory.npz"))
        ell = collect(0:9050)
        theory = JointCMBTheory(ell, vec(th["dlTT"]), vec(th["dlTE"]), vec(th["dlEE"]))
        specs = ("TT_90x90", "TE_90x90", "EE_90x90", "TT_90x150", "TE_90x150", "TE_150x90",
            "EE_90x150", "TT_90x220", "TE_90x220", "TE_220x90", "EE_90x220",
            "TT_150x150", "TE_150x150", "EE_150x150", "TT_150x220", "TE_150x220",
            "TE_220x150", "EE_150x220", "TT_220x220", "TE_220x220", "EE_220x220")
        for tag in tags
            d = npzread(joinpath(FIXTURE_DIR, "spt_$(tag).npz"))
            P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_$(tag).json"))
            shared = SharedForegroundParameters(
                P["Atsz"], P["Acib"], P["Aksz"], P["xi"], P["beta_cib"],
                P["beta_radio"], P["beta_dusty"], P["T_cib"], P["alpha_tsz"])
            spt = SPTForegroundParameters(
                P["SPT3G_AdustTT"], P["SPT3G_AdustTE"], P["SPT3G_AdustEE"],
                P["SPT3G_beta_dustTT"], P["SPT3G_beta_dustTE"], P["SPT3G_beta_dustEE"],
                P["SPT3G_alpha_dustTT"], P["SPT3G_alpha_dustTE"], P["SPT3G_alpha_dustEE"],
                P["SPT3G_radio_TT"], P["SPT3G_radio_TE"], P["SPT3G_radio_EE"],
                P["SPT3G_cib_ps"], P["SPT3G_kappa"])
            sky = spt_sky_model(sd, tpl, theory, shared, spt)
            @test length(sky) == 21 * 4094
            n = 4094
            for (b, s) in enumerate(specs)
                block = sky[(b-1)*n+1:b*n]
                @test block ≈ Float64.(d["sky_$(s)"]) rtol = 1e-8
            end
            if tag == "baseline"
                x_spt = _spt_sky_parameters(shared, spt)
                output_bar = [sin(0.0031i) for i in eachindex(sky)]
                objective(x) = dot(output_bar, _spt_sky_flat(sd, tpl, theory, x))
                gradient_fd = ForwardDiff.gradient(objective, x_spt)
                _, pullback = rrule(_spt_sky_flat, sd, tpl, theory, x_spt)
                gradient_rule = pullback(output_bar)[5]
                @test all(isfinite, gradient_rule)
                @test gradient_rule ≈ gradient_fd rtol = 1e-9 atol = 1e-8
            end
        end
    end
end
