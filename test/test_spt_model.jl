# SPT instrument boundary: sky -> instrument-corrected binned residual,
# validated against the frozen Hillik fixtures at every exported parameter point.

using NPZ
using JSON
using ForwardDiff
using ChainRulesCore: rrule
using LinearAlgebra: dot
import JointCMBLikelihoods: _spt_flat_parameters, _spt_residual_flat

@testset "SPT model boundary" begin
    sd = load_spt_data(SPT_DATA_DIR)

    tags = ["baseline", "common_fg", "plk_cal", "plk_dust", "plk_ps", "act_cal",
        "act_band", "act_fg", "spt_cal", "spt_beam", "spt_leak", "spt_fg",
        "spt_kappa", "multi1", "multi2", "multi3", "multi4", "multi5",
        "boundary_low", "boundary_high"]

    for tag in tags
        d = npzread(joinpath(FIXTURE_DIR, "spt_$(tag).npz"))
        P = JSON.parsefile(joinpath(FIXTURE_DIR, "params_$(tag).json"))
        sky = vcat((vec(d["sky_$(s)"]) for s in
            ("TT_90x90", "TE_90x90", "EE_90x90", "TT_90x150", "TE_90x150", "TE_150x90",
             "EE_90x150", "TT_90x220", "TE_90x220", "TE_220x90", "EE_90x220",
             "TT_150x150", "TE_150x150", "EE_150x150", "TT_150x220", "TE_150x220",
             "TE_220x150", "EE_150x220", "TT_220x220", "TE_220x220", "EE_220x220"))...)
        params = SPTInstrumentParameters(
            P["SPT3G_cal"],
            (P["SPT3G_cal_90"], P["SPT3G_cal_150"], P["SPT3G_cal_220"]),
            (P["SPT3G_pe_90"], P["SPT3G_pe_150"], P["SPT3G_pe_220"]),
            (P["SPT3G_beta_pol_90"], P["SPT3G_beta_pol_150"], P["SPT3G_beta_pol_220"]),
            ntuple(i -> P["SPT3G_beta_$(i)"], 9),
            (P["SPT3G_T2P2_90"], P["SPT3G_T2P2_150"], P["SPT3G_T2P2_220"]),
        )
        delta = spt_residual_vector(sd, sky, params)
        @test length(delta) == 1392
        @test delta ≈ Float64.(d["delta_dl"]) rtol = 1e-6

        if tag == "baseline"
            @testset "whole-boundary analytical VJP" begin
                # Exercise every sky block and every instrument parameter with
                # one reduced coordinate each. ForwardDiff differentiates the
                # arbitrary-cotangent contraction; the custom VJP is then
                # contracted with the same 21 independent sky directions.
                n_ell = length(sd.ells)
                sky_directions = [sin(0.013i + 0.37b) for i in 1:n_ell, b in 1:21]
                delta_bar = [cos(0.17i) for i in eachindex(delta)]
                x = _spt_flat_parameters(params)
                z0 = zeros(43)

                function reduced_objective(z)
                    sky_z = vec(
                        reshape(sky, n_ell, 21) .+
                        sky_directions .* reshape(@view(z[1:21]), 1, :),
                    )
                    x_z = x .+ @view(z[22:43])
                    return dot(delta_bar, _spt_residual_flat(sd, sky_z, x_z))
                end

                gradient_fd = ForwardDiff.gradient(reduced_objective, z0)
                _, pullback = rrule(_spt_residual_flat, sd, sky, x)
                _, _, skybar, xbar = pullback(delta_bar)
                gradient_rule = vcat(
                    [dot(
                        @view(skybar[(b - 1) * n_ell + 1:b * n_ell]),
                        @view(sky_directions[:, b]),
                    ) for b in 1:21],
                    xbar,
                )

                @test all(isfinite, skybar)
                @test all(isfinite, xbar)
                @test gradient_rule ≈ gradient_fd rtol = 1e-9 atol = 1e-8
            end
        end
    end
end
