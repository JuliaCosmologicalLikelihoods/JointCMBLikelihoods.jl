# ACT model boundary: window-projected model + calibration -> residual vector,
# validated against the frozen Hillik fixtures at every exported parameter point.

using NPZ
using ForwardDiff
using ChainRulesCore: rrule
using LinearAlgebra: dot
import JointCMBLikelihoods: _act_residual_flat

@testset "ACT model boundary" begin
    ad = load_act_data(ACT_DATA_DIR)
    th = npzread(joinpath(FIXTURE_DIR, "theory.npz"))
    dlth = Dict(:TT => Float64.(vec(th["dlTT"])), :TE => Float64.(vec(th["dlTE"])),
                :EE => Float64.(vec(th["dlEE"])))

    tags = ["baseline", "common_fg", "plk_cal", "plk_dust", "plk_ps", "act_cal",
        "act_band", "act_fg", "spt_cal", "spt_beam", "spt_leak", "spt_fg",
        "spt_kappa", "multi1", "multi2", "multi3", "multi4", "multi5",
        "boundary_low", "boundary_high"]

    for tag in tags
        d = npzread(joinpath(FIXTURE_DIR, "act_$(tag).npz"))
        dlfg = Dict(:TT => Float64.(d["dlfg_TT"]), :TE => Float64.(d["dlfg_TE"]),
                    :EE => Float64.(d["dlfg_EE"]))
        cal = Float64[]
        for (i, (ispec, pol)) in enumerate(zip(ad.spec_index, ad.spec_pol))
            push!(cal, Float64(d["cal_$(ispec)_$(pol)"]))
        end
        delta = act_residual_vector(ad, dlth, dlfg, cal)
        @test length(delta) == 1139
        @test delta ≈ Float64.(d["delta_dl"]) rtol = 1e-6

        if tag == "baseline"
            @testset "whole-boundary analytical VJP" begin
                # One independent coordinate for every theory mode, every
                # foreground spectrum row, and every released calibration.
                n_theory = length(dlth[:TT])
                n_foreground = size(dlfg[:TT], 2)
                theory_directions = [
                    [sin(0.011i + 0.41m) for i in 1:n_theory] for m in 1:3]
                foreground_directions = [
                    [cos(0.007i + 0.23r + 0.31m) for r in 1:15, i in 1:n_foreground]
                    for m in 1:3
                ]
                delta_bar = [sin(0.19i) for i in eachindex(delta)]
                z0 = zeros(3 + 3 * 15 + length(cal))

                function reduced_objective(z)
                    th_tt = dlth[:TT] .+ z[1] .* theory_directions[1]
                    th_te = dlth[:TE] .+ z[2] .* theory_directions[2]
                    th_ee = dlth[:EE] .+ z[3] .* theory_directions[3]
                    fg_tt = dlfg[:TT] .+ reshape(@view(z[4:18]), 15, 1) .* foreground_directions[1]
                    fg_te = dlfg[:TE] .+ reshape(@view(z[19:33]), 15, 1) .* foreground_directions[2]
                    fg_ee = dlfg[:EE] .+ reshape(@view(z[34:48]), 15, 1) .* foreground_directions[3]
                    cal_z = cal .+ @view(z[49:end])
                    residual = _act_residual_flat(
                        ad, th_tt, th_te, th_ee, fg_tt, fg_te, fg_ee, cal_z)
                    return dot(delta_bar, residual)
                end

                gradient_fd = ForwardDiff.gradient(reduced_objective, z0)
                _, pullback = rrule(
                    _act_residual_flat, ad,
                    dlth[:TT], dlth[:TE], dlth[:EE],
                    dlfg[:TT], dlfg[:TE], dlfg[:EE], cal,
                )
                tangents = pullback(delta_bar)
                thbars = tangents[3:5]
                fgbars = tangents[6:8]
                calbar = tangents[9]
                gradient_rule = vcat(
                    [dot(thbars[m], theory_directions[m]) for m in 1:3],
                    [dot(@view(fgbars[m][r, :]), @view(foreground_directions[m][r, :]))
                     for m in 1:3 for r in 1:15],
                    calbar,
                )

                @test all(isfinite, gradient_rule)
                @test gradient_rule ≈ gradient_fd rtol = 1e-9 atol = 1e-8
            end
        end
    end
end
