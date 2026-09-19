# SPT instrument boundary: pre-instrument 21-spectrum sky vector -> 1392-element residual.
# Frozen Hillik sequence (d0e455cf, spt3g_2025.py):
#   per spectrum:  model = (sky + T2P leakage) .* beam ./ (cal^2 * internal calibration)
#   binned = W' * model;  delta = data - binned, concatenated in release order.
# Artifact ℓ grid is 2:4095; absolute-ℓ 800 is Julia index 799.

struct SPTInstrumentParameters{T}
    cal::T                    # SPT3G_cal
    cal_freq::NTuple{3,T}     # SPT3G_cal_{90,150,220}
    pe_freq::NTuple{3,T}      # SPT3G_pe_{90,150,220}
    beta_pol::NTuple{3,T}     # SPT3G_beta_pol_{90,150,220}
    beta_modes::NTuple{9,T}   # SPT3G_beta_1..9
    t2p::NTuple{3,T}          # SPT3G_T2P2_{90,150,220}
end

const _SPT_CAL = 1
const _SPT_CAL_FREQ = 2:4
const _SPT_PE_FREQ = 5:7
const _SPT_BETA_POL = 8:10
const _SPT_BETA_MODES = 11:19
const _SPT_T2P = 20:22

const _SPT_T2P_SIGMAS = (0.000274, 0.000192, 0.000169)

# (mode, freq1, freq2) per release-ordered block 1..21; freq index 1=90, 2=150, 3=220.
const _SPT_SPECS = (
    ("TT", 1, 1), ("TE", 1, 1), ("EE", 1, 1),
    ("TT", 1, 2), ("TE", 1, 2), ("TE", 2, 1), ("EE", 1, 2),
    ("TT", 1, 3), ("TE", 1, 3), ("TE", 3, 1), ("EE", 1, 3),
    ("TT", 2, 2), ("TE", 2, 2), ("EE", 2, 2),
    ("TT", 2, 3), ("TE", 2, 3), ("TE", 3, 2), ("EE", 2, 3),
    ("TT", 3, 3), ("TE", 3, 3), ("EE", 3, 3),
)
const _SPT_TT_BLOCKS = (1, 4, 8, 12, 15, 19)
const _SPT_TT_PAIRS = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
const _SPT_TE_BLOCKS = (2, 5, 6, 9, 10, 13, 16, 17, 20)
const _SPT_TE_PAIRS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3))
const _SPT_EE_BLOCKS = (3, 7, 11, 14, 18, 21)

@inline function _spt_tt_block(i, j)
    a, b = minmax(i, j)
    return _SPT_TT_BLOCKS[findfirst(==((a, b)), _SPT_TT_PAIRS)]
end

@inline function _spt_te_block(i, j)
    return _SPT_TE_BLOCKS[findfirst(==((i, j)), _SPT_TE_PAIRS)]
end

"""
    spt_residual_vector(data::SPTData, sky, params) -> Vector

`sky` is the 21-spectrum pre-instrument vector in release order (each block on
the artifact ℓ grid 2:4095): CMB + foregrounds with SSL and aberration already
applied. Applies the frozen D1 instrument sequence (T2P leakage, beam
eigenmodes + polarized beam, calibration) and window binning, returning the
1392-element residual.
"""
function spt_residual_vector(
    data::SPTData,
    sky::AbstractVector{<:Real},
    params::SPTInstrumentParameters,
)
    return _spt_residual_flat(data, sky, _spt_flat_parameters(params))
end

"""Flatten the SPT instrument parameters at the internal AD boundary."""
function _spt_flat_parameters(params::SPTInstrumentParameters)
    return [
        params.cal,
        params.cal_freq...,
        params.pe_freq...,
        params.beta_pol...,
        params.beta_modes...,
        params.t2p...,
    ]
end

"""
    _spt_residual_flat(data, sky, x) -> Vector

Flat internal boundary for the complete SPT instrument model. The parameter
layout is global calibration, three frequency calibrations, three polarization
efficiencies, three polarized-beam amplitudes, nine temperature-beam modes,
and three T-to-P amplitudes. Keeping this boundary vector-typed lets reverse
mode use one analytical pullback without changing the public struct API.
"""
function _spt_residual_flat(
    data::SPTData,
    sky::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
)
    n_ell = length(data.ells)
    length(sky) == 21 * n_ell ||
        throw(DimensionMismatch("SPT sky vector must have 21 blocks of $n_ell"))
    length(x) == 22 || throw(DimensionMismatch("SPT instrument vector must have 22 elements"))
    block(b) = @view sky[(b - 1) * n_ell + 1:b * n_ell]

    T = promote_type(eltype(sky), eltype(x))
    # T2P leakage kernels per frequency: eps * sigma^2 * ell^2
    t2p_kernel = _spt_t2p_kernels(data, @view(x[_SPT_T2P]))

    # beam eigenmode response and polarized beam per frequency
    temperature_beam = _spt_temperature_beam(data, @view(x[_SPT_BETA_MODES]))
    polarization_beam = _spt_polarization_beam(
        data, temperature_beam, @view(x[_SPT_BETA_POL]))

    calT = @view x[_SPT_CAL_FREQ]
    calE = @view x[_SPT_PE_FREQ]
    cal = x[_SPT_CAL]
    delta = Vector{T}(undef, length(data.data_vector))
    i0 = 1
    for (b, (mode, f1, f2)) in enumerate(_SPT_SPECS)
        s = collect(block(b))
        lo, hi = minmax(f1, f2)
        if mode == "TT"
            leak = zero(s)
            beam = temperature_beam[f1, :] .* temperature_beam[f2, :]
            calint = calT[f1] * calT[f2]
        elseif mode == "TE"
            leak = t2p_kernel[f2] .* block(_spt_tt_block(f1, f2))
            beam = temperature_beam[f1, :] .* polarization_beam[f2, :]
            calint = calT[f1] * (calT[f2] * calE[f2])
        else  # EE
            leak = t2p_kernel[f1] .* block(_spt_te_block(lo, hi)) .+
                   t2p_kernel[f2] .* block(_spt_te_block(hi, lo)) .+
                   (t2p_kernel[f1] .* t2p_kernel[f2]) .* block(_spt_tt_block(f1, f2))
            beam = polarization_beam[f1, :] .* polarization_beam[f2, :]
            calint = (calT[f1] * calE[f1]) * (calT[f2] * calE[f2])
        end
        model = (s .+ leak) .* beam ./ (cal^2 * calint)
        nb = size(data.windows[b], 2)
        binned = _spt_window_project(data, b, model)
        delta[i0:i0+nb-1] .= data.data_vector[i0:i0+nb-1] .- binned
        i0 += nb
    end
    return delta
end

"""Fixed SPT T2P leakage kernels for the three sampled leakage amplitudes."""
function _spt_t2p_kernels(data::SPTData, t2p::AbstractVector{<:Real})
    length(t2p) == 3 || throw(DimensionMismatch("expected three SPT T2P amplitudes"))
    ell2 = eltype(t2p).(data.ells) .^ 2
    return [t2p[f] * _SPT_T2P_SIGMAS[f]^2 .* ell2 for f in 1:3]
end

"""Fixed SPT beam-eigenmode response for the nine sampled beam amplitudes."""
function _spt_temperature_beam(data::SPTData, beta_modes::AbstractVector{<:Real})
    size(data.eigenmodes, 3) == length(beta_modes) || throw(DimensionMismatch("expected nine SPT beam modes"))
    T = promote_type(eltype(data.eigenmodes), eltype(beta_modes))
    out = ones(T, 3, size(data.eigenmodes, 2))
    for m in eachindex(beta_modes)
        @views out .+= beta_modes[m] .* data.eigenmodes[:, :, m]
    end
    return out
end

"""Fixed SPT polarized-beam response, normalized at absolute multipole 800."""
function _spt_polarization_beam(data::SPTData, temperature_beam::AbstractMatrix{<:Real}, beta_pol::AbstractVector{<:Real})
    T = promote_type(eltype(temperature_beam), eltype(beta_pol))
    out = Matrix{T}(undef, size(temperature_beam))
    i800 = 800 - data.ells[1] + 1
    for f in 1:3
        main = @view data.main_temperature[f, :]
        β = beta_pol[f]
        norm = main[i800] + β * (1 - main[i800])
        @views out[f, :] .= (main .+ β .* (temperature_beam[f, :] .- main)) ./ norm
    end
    return out
end

"""Released SPT window projection; the window is fixed data, never an AD input."""
_spt_window_project(data::SPTData, b::Int, spectrum::AbstractVector) =
    window_convolution(data.windows[b], spectrum)
