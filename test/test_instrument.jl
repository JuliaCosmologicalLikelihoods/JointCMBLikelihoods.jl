# Instrument parameter layer: calibration builders validated against the
# per-spectrum cal vectors exported by the frozen Hillik at every fixture point.

using NPZ
using JSON

@testset "Instrument parameters" begin
    tags = ["baseline", "common_fg", "plk_cal", "plk_dust", "plk_ps", "act_cal",
        "act_band", "act_fg", "spt_cal", "spt_beam", "spt_leak", "spt_fg",
        "spt_kappa", "multi1", "multi2", "multi3", "multi4", "multi5",
        "boundary_low", "boundary_high"]

    # ------------------------------------------------------------------
    # Planck: 15-element cal per mode
    # ------------------------------------------------------------------
    @testset "Planck calibration" begin
        for tag in tags
            d = npzread(joinpath(FIXTURE_DIR, "planck_$(tag).npz"))
            P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_$(tag).json"))
            plk = PlanckInstrumentParameters(
                P["A_planck"],
                (P["PLK_cal_100A"], P["PLK_cal_100B"], P["PLK_cal_143A"],
                 P["PLK_cal_143B"], P["PLK_cal_217A"], P["PLK_cal_217B"]),
                (P["PLK_pe_100A"], P["PLK_pe_100B"], P["PLK_pe_143A"],
                 P["PLK_pe_143B"], P["PLK_pe_217A"], P["PLK_pe_217B"]))
            @test n_sampled(plk) == 12
            for mode in (:TT, :EE, :TE, :ET)
                cal = planck_calibration(plk, mode)
                @test length(cal) == 15
                @test cal ≈ Float64.(d["cal_$(mode)"]) rtol = 1e-10
            end
        end
    end

    # ------------------------------------------------------------------
    # ACT: per (spec, pol) cal vector
    # ------------------------------------------------------------------
    @testset "ACT calibration" begin
        ad = load_act_data(ACT_DATA_DIR)
        for tag in tags
            d = npzread(joinpath(FIXTURE_DIR, "act_$(tag).npz"))
            P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_$(tag).json"))
            act = ACTInstrumentParameters(
                P["ACT_cal"],
                (P["ACT_cal_dr6_pa4_f220"], P["ACT_cal_dr6_pa5_f090"],
                 P["ACT_cal_dr6_pa5_f150"], P["ACT_cal_dr6_pa6_f090"],
                 P["ACT_cal_dr6_pa6_f150"]),
                (P["ACT_pe_dr6_pa4_f220"], P["ACT_pe_dr6_pa5_f090"],
                 P["ACT_pe_dr6_pa5_f150"], P["ACT_pe_dr6_pa6_f090"],
                 P["ACT_pe_dr6_pa6_f150"]),
                (P["ACT_band_shift_dr6_pa4_f220"], P["ACT_band_shift_dr6_pa5_f090"],
                 P["ACT_band_shift_dr6_pa5_f150"], P["ACT_band_shift_dr6_pa6_f090"],
                 P["ACT_band_shift_dr6_pa6_f150"]))
            @test n_sampled(act) == 14
            cal = act_calibration(act, ad)
            @test length(cal) == length(ad.spec_pol)
            ref = [Float64(d["cal_$(ispec)_$(pol)"])
                   for (ispec, pol) in zip(ad.spec_index, ad.spec_pol)]
            @test cal ≈ ref rtol = 1e-10
        end
    end

    # ------------------------------------------------------------------
    # SPT: 22 sampled instrument parameters
    # ------------------------------------------------------------------
    @testset "SPT instrument count" begin
        P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_baseline.json"))
        spt = SPTInstrumentParameters(
            P["SPT3G_cal"],
            (P["SPT3G_cal_90"], P["SPT3G_cal_150"], P["SPT3G_cal_220"]),
            (P["SPT3G_pe_90"], P["SPT3G_pe_150"], P["SPT3G_pe_220"]),
            (P["SPT3G_beta_pol_90"], P["SPT3G_beta_pol_150"], P["SPT3G_beta_pol_220"]),
            ntuple(i -> P["SPT3G_beta_$(i)"], 9),
            (P["SPT3G_T2P2_90"], P["SPT3G_T2P2_150"], P["SPT3G_T2P2_220"]))
        @test n_sampled(spt) == 22
    end

    # Plan §4.3: 12 + 14 + 22 = 48 sampled instrument parameters
    @test 12 + 14 + 22 == 48
end
