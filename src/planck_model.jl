# Planck model boundary: per-xspec model spectra -> 4872-element residual vector.
# Reproduces the frozen Hillik sequence (d0e455cf):
#   Rspec = dldata - dlmodel/cal            (per cross-spectrum)
#   Rl    = weighted xfreq average          (zero-weight ells -> 0)
#   TE/ET = unnormalised sums, combined, divided by total weight
#   Xl    = per-xfreq cut binning, Dl=False (plain bin average)
# Vector order: TT (1646) + EE (1466) + TE/ET combined (1760).

"""
    planck_residual_vector(data::PlanckData, dlmodel, cal) -> Vector{Float64}

`dlmodel[mode]` is the full per-cross-spectrum model `(15, 2501)` on ℓ=0..2500
(theory + foregrounds, uncalibrated). `cal[mode]` is the 15-element calibration
product per cross-spectrum. Returns the 4872-element residual vector before the
float32 cast.
"""
function planck_residual_vector(
    data::PlanckData,
    dlmodel::Dict{Symbol,<:Matrix{<:Real}},
    cal::Dict{Symbol,<:Vector{<:Real}},
)
    return _planck_residual_flat(
        data,
        dlmodel[:TT], dlmodel[:EE], dlmodel[:TE], dlmodel[:ET],
        cal[:TT], cal[:EE], cal[:TE], cal[:ET],
    )
end

"""
    _planck_residual_flat(data, model_tt, model_ee, model_te, model_et,
                         cal_tt, cal_ee, cal_te, cal_et)

Concrete-array internal boundary for the complete released Planck residual
model. This avoids struct- and dictionary-typed differentiable arguments while
preserving the public API and frozen TT/EE/TE+ET averaging conventions.
"""
function _planck_residual_flat(
    data::PlanckData,
    model_tt::AbstractMatrix{<:Real},
    model_ee::AbstractMatrix{<:Real},
    model_te::AbstractMatrix{<:Real},
    model_et::AbstractMatrix{<:Real},
    cal_tt::AbstractVector{<:Real},
    cal_ee::AbstractVector{<:Real},
    cal_te::AbstractVector{<:Real},
    cal_et::AbstractVector{<:Real},
)
    Rspec_tt = data.dldata[:TT] .- model_tt ./ cal_tt
    Rspec_ee = data.dldata[:EE] .- model_ee ./ cal_ee
    Rl_tt = _xfreq_average(data, Rspec_tt, data.dlweight[:TT])
    Rl_ee = _xfreq_average(data, Rspec_ee, data.dlweight[:EE])

    # TE + ET: unnormalised weighted sums, then combined ratio
    Rspec_te = data.dldata[:TE] .- model_te ./ cal_te
    Rspec_et = data.dldata[:ET] .- model_et ./ cal_et
    num_te, den_te = _xfreq_sum(data, Rspec_te, data.dlweight[:TE])
    num_et, den_et = _xfreq_sum(data, Rspec_et, data.dlweight[:ET])
    Rl_te = _safe_ratio.(num_te .+ num_et, den_te .+ den_et)

    Xl = vcat(
        _select_spectra(data, Rl_tt, :TT),
        _select_spectra(data, Rl_ee, :EE),
        _select_spectra(data, Rl_te, :TE),
    )
    return Xl
end

function _xfreq_average(data::PlanckData, Rspec::Matrix{<:Real}, weight::Matrix{<:Real})
    n_ell = size(Rspec, 2)
    num = zeros(eltype(Rspec), 6, n_ell)
    den = zeros(eltype(Rspec), 6, n_ell)
    _accumulate_xfreq!(num, den, data, Rspec, weight)
    return _safe_ratio.(num, den)
end

function _xfreq_sum(data::PlanckData, Rspec::Matrix{<:Real}, weight::Matrix{<:Real})
    n_ell = size(Rspec, 2)
    num = zeros(eltype(Rspec), 6, n_ell)
    den = zeros(eltype(Rspec), 6, n_ell)
    _accumulate_xfreq!(num, den, data, Rspec, weight)
    return num, den
end

function _accumulate_xfreq!(num, den, data::PlanckData, Rspec, weight)
    for xs in 1:15
        xf = data.xspec2xfreq[xs] + 1
        @inbounds for l in 1:size(Rspec, 2)
            num[xf, l] += weight[xs, l] * Rspec[xs, l]
            den[xf, l] += weight[xs, l]
        end
    end
    return nothing
end

@inline _safe_ratio(n, d) = iszero(d) ? zero(n) : n / d

# Per-xfreq multipole cuts: lmin/lmax of the FIRST cross-spectrum in each xfreq
# (frozen convention: self._lmins[mode][self._xspec2xfreq.index(xf)]).
function _xfreq_cut(data::PlanckData, mode::Symbol, xf::Int)
    xs = findfirst(==(xf - 1), data.xspec2xfreq)
    return data.lmins[mode][xs], data.lmaxs[mode][xs]
end

"""
    _select_spectra(data, Rl, mode)

Per-xfreq cut binning with the lite binning (Dl=False: plain bin average),
following the frozen `cut_binning` + `_bin_operators(Dl=False)` sequence.
"""
function _select_spectra(data::PlanckData, Rl::Matrix{<:Real}, mode::Symbol)
    xl = eltype(Rl)[]
    for xf in 1:6
        lmin, lmax = _xfreq_cut(data, mode, xf)
        for (bmin, bmax) in zip(data.bin_lmins, data.bin_lmaxs)
            bmin >= lmin && bmax <= lmax || continue
            dl = bmax - bmin + 1
            # bins are 1-based over ℓ=0..2500: ℓ = index - 1
            push!(xl, sum(@view Rl[xf, bmin+1:bmax+1]) / dl)
        end
    end
    return xl
end
