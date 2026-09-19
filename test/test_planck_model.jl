# Planck model boundary: dlmodel/cal -> residual vector, validated against
# the frozen Hillik fixtures at every exported parameter point.

using NPZ
using ForwardDiff
using ChainRulesCore: rrule
using LinearAlgebra: dot
import JointCMBLikelihoods: _planck_residual_flat

@testset "Planck model boundary" begin
    pd = load_planck_data(PLANCK_DATA_DIR)
    th = npzread(joinpath(FIXTURE_DIR, "theory.npz"))
    dlth = Dict(:TT => vec(th["dlTT"])[1:2501], :EE => vec(th["dlEE"])[1:2501],
                :TE => vec(th["dlTE"])[1:2501])

    tags = ["baseline", "common_fg", "plk_cal", "plk_dust", "plk_ps", "act_cal",
        "act_band", "act_fg", "spt_cal", "spt_beam", "spt_leak", "spt_fg",
        "spt_kappa", "multi1", "multi2", "multi3", "multi4", "multi5",
        "boundary_low", "boundary_high"]

    for tag in tags
        d = npzread(joinpath(FIXTURE_DIR, "planck_$(tag).npz"))
        dlmodel = Dict{Symbol,Matrix{Float64}}()
        cal = Dict{Symbol,Vector{Float64}}()
        for mode in (:TT, :EE, :TE, :ET)
            m = dlth[mode in (:TT, :EE, :TE) ? mode : :TE]
            dlmodel[mode] = Float64.(d["dlmodel_$(mode)"]) .+ m'
            cal[mode] = Float64.(d["cal_$(mode)"])
        end
        Xl = planck_residual_vector(pd, dlmodel, cal)
        @test length(Xl) == 4872
        @test Xl ≈ Float64.(d["delta_dl"]) rtol = 1e-6

        if tag == "baseline"
            @testset "whole-boundary analytical VJP" begin
                modes = (:TT, :EE, :TE, :ET)
                model_directions = [
                    [sin(0.009l + 0.29r + 0.17m) for r in 1:15, l in 1:2501]
                    for m in 1:4
                ]
                delta_bar = [cos(0.13i) for i in eachindex(Xl)]
                z0 = zeros(4 * 15 + 4 * 15)

                function reduced_objective(z)
                    models = ntuple(4) do m
                        offset = (m - 1) * 15
                        dlmodel[modes[m]] .+
                            reshape(@view(z[offset+1:offset+15]), 15, 1) .* model_directions[m]
                    end
                    cals = ntuple(4) do m
                        offset = 60 + (m - 1) * 15
                        cal[modes[m]] .+ @view(z[offset+1:offset+15])
                    end
                    residual = _planck_residual_flat(
                        pd, models[1], models[2], models[3], models[4],
                        cals[1], cals[2], cals[3], cals[4],
                    )
                    return dot(delta_bar, residual)
                end

                gradient_fd = ForwardDiff.gradient(reduced_objective, z0)
                _, pullback = rrule(
                    _planck_residual_flat, pd,
                    dlmodel[:TT], dlmodel[:EE], dlmodel[:TE], dlmodel[:ET],
                    cal[:TT], cal[:EE], cal[:TE], cal[:ET],
                )
                tangents = pullback(delta_bar)
                modelbars = tangents[3:6]
                calbars = tangents[7:10]
                gradient_rule = vcat(
                    [dot(@view(modelbars[m][r, :]), @view(model_directions[m][r, :]))
                     for m in 1:4 for r in 1:15],
                    calbars...,
                )

                @test all(isfinite, gradient_rule)
                @test gradient_rule ≈ gradient_fd rtol = 1e-9 atol = 1e-8
            end
        end
    end
end
