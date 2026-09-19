# ACT model boundary: per-spectrum foreground spectra + calibration -> 1139-element residual.
# Frozen Hillik sequence (d0e455cf):
#   X_model = Wᵀ (dlth + dlfg)[values]     per (spectrum, pol)
#   delta   = dl - X_model / cal           concatenated in spec_index order

"""
    act_residual_vector(data::ACTData, dlth, dlfg, cal) -> Vector{Float64}

`dlth[pol]` is the theory Dℓ on the full bandpower window ℓ grid (0..lmax_bpw);
`dlfg[pol]` is the per-spectrum foreground array `(n_spec, n_ell)`; `cal[i]` is
the calibration product for selected spectrum `i`. Returns the 1139-element
residual `dl - X_model/cal` in the frozen spec order.
"""
function act_residual_vector(
    data::ACTData,
    dlth::Dict{Symbol,<:Vector{<:Real}},
    dlfg::Dict{Symbol,<:Matrix{<:Real}},
    cal::AbstractVector{<:Real},
)
    return _act_residual_flat(
        data,
        dlth[:TT], dlth[:TE], dlth[:EE],
        dlfg[:TT], dlfg[:TE], dlfg[:EE],
        cal,
    )
end

"""
    _act_residual_flat(data, dlth_tt, dlth_te, dlth_ee,
                      dlfg_tt, dlfg_te, dlfg_ee, cal)

Concrete-array internal boundary for the complete released ACT residual model.
ET entries intentionally reuse the TE theory and foreground arrays, matching
the frozen likelihood. The public dictionary API remains unchanged.
"""
function _act_residual_flat(
    data::ACTData,
    dlth_tt::AbstractVector{<:Real},
    dlth_te::AbstractVector{<:Real},
    dlth_ee::AbstractVector{<:Real},
    dlfg_tt::AbstractMatrix{<:Real},
    dlfg_te::AbstractMatrix{<:Real},
    dlfg_ee::AbstractMatrix{<:Real},
    cal::AbstractVector{<:Real},
)
    length(cal) == length(data.windows) ||
        throw(DimensionMismatch("ACT cal must have one entry per selected spectrum"))
    T = promote_type(
        eltype(dlth_tt), eltype(dlth_te), eltype(dlth_ee),
        eltype(dlfg_tt), eltype(dlfg_te), eltype(dlfg_ee), eltype(cal),
    )
    delta = Vector{T}(undef, length(data.data_vector))
    i0 = 1
    for (i, (W, values, pol, ispec)) in enumerate(zip(data.windows, data.window_values, data.spec_pol, data.spec_index))
        nb = size(W, 2)
        idx = values .+ 1          # window values are 0-based absolute multipoles
        if pol == "TT"
            m = dlth_tt
            f = dlfg_tt
        elseif pol == "EE"
            m = dlth_ee
            f = dlfg_ee
        else                       # TE and ET share the frozen TE model
            m = dlth_te
            f = dlfg_te
        end
        s = m[idx] .+ f[ispec + 1, idx]
        X = _act_window_project(data, i, s)
        delta[i0:i0+nb-1] .= data.data_vector[i0:i0+nb-1] .- X ./ cal[i]
        i0 += nb
    end
    return delta
end

"""Released ACT window projection; the window is fixed data, never an AD input."""
_act_window_project(data::ACTData, i::Int, spectrum::AbstractVector) =
    window_convolution(data.windows[i], spectrum)
