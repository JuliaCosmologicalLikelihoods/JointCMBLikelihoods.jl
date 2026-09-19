# Shared foreground model for the joint Planck + ACT + SPT likelihood
# (frozen Hillik d0e455cf, hillik_foregrounds/foregrounds.py).
#
# SEDs are normalized to nu0 = 150 GHz. Frequency handling per survey:
#   Planck / SPT: effective frequencies (feff tables)
#   ACT: beam chromaticity — bandpass * dBdT(nu + shift) * beam(ell, nu), normalized
#        over nu, then fgRatio integrated over the bandpass.
# Templates are raw D_l arrays normalized at lnorm BEFORE slicing to lmax
# (_read_dl_template semantics). Power laws are built on max(lmax, lnorm) and
# normalized at lnorm before slicing (_gen_dl_powerlaw semantics).

export SharedForegroundParameters,
    PlanckForegroundParameters,
    ACTForegroundParameters,
    SPTForegroundParameters,
    planck_foregrounds,
    act_foregrounds,
    spt_sky_model

# ---------------------------------------------------------------------------
# Physical constants (frozen SPT values, foregrounds.py)
# ---------------------------------------------------------------------------

const _FG_T_CMB = 2.72548
const _FG_K_B = 1.3806488e-23
const _FG_H_PL = 6.62606957e-34
const _FG_T_DUST = 19.6

"""Thermodynamic-to-antenna conversion, frequency in GHz (frozen dBdT)."""
function _fg_dbdt(f::Real)
    nu = f * 1e9
    x = _FG_H_PL * nu / (_FG_K_B * _FG_T_CMB)
    return nu^4 * exp(x) / expm1(x)^2
end

"""Planck spectrum, frequency in GHz, temperature in K (frozen f_Planck)."""
function _fg_planck(f::Real, T::Real)
    nu = f * 1e9
    x = _FG_H_PL * nu / (_FG_K_B * T)
    return nu^3 / expm1(x)
end

"""Non-relativistic tSZ SED, frequency in GHz (frozen f_tsz)."""
function _fg_tsz(f::Real)
    nu = f * 1e9
    x = _FG_H_PL * nu / (_FG_K_B * _FG_T_CMB)
    return x / tanh(x / 2) - 4
end

# SED ratios normalized to nu0 = 150 GHz (frozen frequency_scaling / fgRatio)
_fg_dust_ratio(nu, beta) =
    (nu^beta * _fg_planck(nu, _FG_T_DUST) / _fg_dbdt(nu)) /
    (150.0^beta * _fg_planck(150.0, _FG_T_DUST) / _fg_dbdt(150.0))
_fg_cib_ratio(nu, beta, T) =
    (nu^beta * _fg_planck(nu, T) / _fg_dbdt(nu)) /
    (150.0^beta * _fg_planck(150.0, T) / _fg_dbdt(150.0))
_fg_radio_ratio(nu, beta) =
    (nu^beta / _fg_dbdt(nu)) / (150.0^beta / _fg_dbdt(150.0))
_fg_tsz_ratio(nu) = _fg_tsz(nu) / _fg_tsz(150.0)

# ---------------------------------------------------------------------------
# Template / power-law helpers (frozen _read_dl_template / _gen_dl_powerlaw)
# ---------------------------------------------------------------------------

"""
    _fg_read_template(tpl::SharedTemplates, which::Symbol, lmax::Int, lnorm::Int)

Raw template normalized at `lnorm`, then sliced to ℓ=0..lmax. The template
grid starts at ℓ=2; ℓ=0,1 entries are zero.
"""
function _fg_read_template(tpl::SharedTemplates, which::Symbol, lmax::Int, lnorm::Int)
    dl = getfield(tpl, which)
    ell = tpl.ell
    nfull = max(lmax, ell[length(dl)])   # CIB grid stops at ℓ=13000, others at 13500
    t = zeros(nfull + 1)
    for i in eachindex(dl)
        t[ell[i]+1] = dl[i]
    end
    t ./= t[lnorm+1]
    return t[1:lmax+1]
end

"""Power-law D_l = l(l+1)/2π · l^alpha on ℓ=0..lmax, normalized at lnorm."""
function _fg_powerlaw(alpha::Real, lmax::Int, lnorm::Int)
    lmax_ = max(lmax, lnorm)
    t = zeros(typeof(zero(alpha) * 1.0), lmax_ + 1)
    for l in 2:lmax_
        t[l+1] = l * (l + 1) / 2π * l^alpha
    end
    t ./= t[lnorm+1]
    return t[1:lmax+1]
end

"""tSZ template with the (ℓ/3000)^alpha_tsz tilt applied on ℓ ≥ 2."""
function _fg_tsz_template(tpl::SharedTemplates, lmax::Int, alpha_tsz::Real)
    dl = _fg_read_template(tpl, :tsz, lmax, 3000)
    out = similar(dl, promote_type(eltype(dl), typeof(alpha_tsz)))
    for l in 1:lmax+1
        out[l] = dl[l]
    end
    for l in 2:lmax
        out[l+1] *= (l / 3000)^alpha_tsz
    end
    return out
end

# trapezoid rule matching numpy.trapz/np.trapezoid semantics
function _trapz(y::AbstractVector, x::AbstractVector)
    n = length(y)
    n == 1 && return zero(y[1])
    s = zero(y[1])
    for i in 1:n-1
        s += (x[i+1] - x[i]) * (y[i] + y[i+1])
    end
    return s / 2
end

# ---------------------------------------------------------------------------
# Parameter structs
# ---------------------------------------------------------------------------

"""
    SharedForegroundParameters

The 9 shared extragalactic parameters. `alpha_cib` is fixed and unused by the
frozen code (the clustered-CIB shape comes from the `cib_extra.dat` template).
"""
struct SharedForegroundParameters{T}
    Atsz::T
    Acib::T
    Aksz::T
    xi::T
    beta_cib::T
    beta_radio::T
    beta_dusty::T
    T_cib::T
    alpha_tsz::T
end

"""
    PlanckForegroundParameters

Planck dust (per-frequency amplitudes for each mode, per-frequency TT tilts,
per-mode TE/ET/EE tilts and spectral indices), radio and dusty Poisson
amplitudes. `Adust` ordering: (TT 100, TT 143, TT 217, TE 100, TE 143, TE 217,
ET 100, ET 143, ET 217, EE 100, EE 143, EE 217).
"""
struct PlanckForegroundParameters{T}
    Adust::NTuple{12,T}
    beta_dust::NTuple{4,T}      # TT, TE, ET, EE
    alpha_dustTT::NTuple{3,T}   # 100, 143, 217 (per max-frequency of the xspec)
    alpha_dustTE::T
    alpha_dustET::T
    alpha_dustEE::T
    radio_TT::T
    cib_ps::T
end

@inline _plk_dust_index(mode::Symbol, f::Int) =
    (mode == :TT ? 0 : mode == :TE ? 3 : mode == :ET ? 6 : 9) +
    (f == 100 ? 1 : f == 143 ? 2 : 3)

@inline _plk_freq_index(f::Int) = f == 100 ? 1 : f == 143 ? 2 : 3

"""
    ACTForegroundParameters

ACT dust (per-mode), radio / dusty Poisson amplitudes, and bandpass shifts.
The frozen EE foreground list contains dust only (`radio_EE` is a defined but
unused parameter, kept for the parameter vector); `radio_TE` is fixed to 0.
"""
struct ACTForegroundParameters{T}
    AdustTT::T
    AdustTE::T
    AdustEE::T
    beta_dustTT::T
    beta_dustTE::T
    beta_dustEE::T
    alpha_dustTT::T
    alpha_dustTE::T
    alpha_dustEE::T
    radio_TT::T
    radio_EE::T
    cib_ps::T
    band_shift::Dict{String,T}   # map name -> shift [GHz]
end

"""
    SPTForegroundParameters

SPT dust (per-mode, plain power-law class with lnorm=80), radio and dusty
Poisson amplitudes, plus the super-sample-lensing amplitude. The aberration
coefficient is the frozen default.
"""
struct SPTForegroundParameters{T}
    AdustTT::T
    AdustTE::T
    AdustEE::T
    beta_dustTT::T
    beta_dustTE::T
    beta_dustEE::T
    alpha_dustTT::T
    alpha_dustTE::T
    alpha_dustEE::T
    radio_TT::T
    radio_TE::T
    radio_EE::T
    cib_ps::T
    kappa::T
end

const _SPT_ABERRATION = -0.0004826   # frozen default (Jeong+13 boost amplitude)

# ---------------------------------------------------------------------------
# Planck foregrounds (feff SEDs, 15 cross-spectra of 6 maps)
# ---------------------------------------------------------------------------

const _PLK_FEFF_DUST = Dict(100 => 105.2, 143 => 148.5, 217 => 228.1)
const _PLK_FEFF_TSZ = Dict(100 => 100.2, 143 => 143.0, 217 => 222.0)
const _PLK_FEFF_RADIO = Dict(100 => 100.4, 143 => 140.5, 217 => 218.6)

# all 15 map pairs, combinations(range(6), 2) order
const _PLK_XSPECS = ((1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5),
    (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6))

"""
    planck_foregrounds(tpl, shared, plk, mode) -> Matrix (15, 2501)

Planck per-cross-spectrum foreground D_l for `mode ∈ (:TT, :TE, :ET, :EE)`.
TT: dust + tSZ + kSZ + clustered CIB + SZxCIB + radio PS + dusty PS.
TE/ET/EE: dust only. CIB and dusty SEDs use the dust feff table.
"""
function planck_foregrounds(
    tpl::SharedTemplates,
    shared::SharedForegroundParameters,
    plk::PlanckForegroundParameters,
    mode::Symbol,
)
    return _planck_foregrounds_flat(
        tpl, mode, _planck_foreground_parameters(shared, plk))
end

function _planck_foreground_parameters(
    shared::SharedForegroundParameters, plk::PlanckForegroundParameters,
)
    return [
        shared.Atsz, shared.Acib, shared.Aksz, shared.xi,
        shared.beta_cib, shared.beta_radio, shared.beta_dusty,
        shared.T_cib, shared.alpha_tsz,
        plk.Adust..., plk.beta_dust..., plk.alpha_dustTT...,
        plk.alpha_dustTE, plk.alpha_dustET, plk.alpha_dustEE,
        plk.radio_TT, plk.cib_ps,
    ]
end

function _planck_foregrounds_flat(
    tpl::SharedTemplates, mode::Symbol, x::AbstractVector{<:Real},
)
    length(x) == 33 || throw(DimensionMismatch("Planck foreground parameter vector must have 33 elements"))
    shared = SharedForegroundParameters(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
    plk = PlanckForegroundParameters(
        ntuple(i -> x[9 + i], 12), ntuple(i -> x[21 + i], 4),
        ntuple(i -> x[25 + i], 3), x[29], x[30], x[31], x[32], x[33],
    )
    return _planck_foregrounds_impl(tpl, shared, plk, mode)
end

function _planck_foregrounds_impl(
    tpl::SharedTemplates,
    shared::SharedForegroundParameters,
    plk::PlanckForegroundParameters,
    mode::Symbol,
)
    lmax = 2500
    exps = (100, 143, 217)

    beta = mode == :TT ? plk.beta_dust[1] : mode == :TE ? plk.beta_dust[2] :
           mode == :ET ? plk.beta_dust[3] : plk.beta_dust[4]
    dust_sed = Dict(f => _fg_dust_ratio(_PLK_FEFF_DUST[f], beta) for f in exps)

    T = typeof(dust_sed[100])
    out = zeros(T, 15, lmax + 1)
    for (i, (m1, m2)) in enumerate(_PLK_XSPECS)
        f1 = PLK_FREQS[m1]
        f2 = PLK_FREQS[m2]
        fm = max(f1, f2)
        ad = plk.Adust[_plk_dust_index(mode, fm)]
        alpha = mode == :TT ? plk.alpha_dustTT[_plk_freq_index(fm)] :
                mode == :TE ? plk.alpha_dustTE :
                mode == :ET ? plk.alpha_dustET : plk.alpha_dustEE
        dlg = _fg_powerlaw(alpha, lmax, 200)
        out[i, :] .+= ad .* dlg .* (dust_sed[f1] * dust_sed[f2])
    end

    if mode == :TT
        tsz_sed = Dict(f => _fg_tsz_ratio(_PLK_FEFF_TSZ[f]) for f in exps)
        cib_sed = Dict(f => _fg_cib_ratio(_PLK_FEFF_DUST[f], shared.beta_cib, shared.T_cib)
                       for f in exps)
        radio_sed = Dict(f => _fg_radio_ratio(_PLK_FEFF_RADIO[f], shared.beta_radio) for f in exps)
        dusty_sed = Dict(f => _fg_cib_ratio(_PLK_FEFF_DUST[f], shared.beta_dusty, shared.T_cib)
                         for f in exps)

        dl_tsz = _fg_tsz_template(tpl, lmax, shared.alpha_tsz)
        dl_ksz = _fg_read_template(tpl, :ksz, lmax, 3000)
        dl_cib = _fg_read_template(tpl, :cib, lmax, 3000)
        dl_szxcib = _fg_read_template(tpl, :szxcib, lmax, 3000)
        dl_ps = _fg_powerlaw(0.0, lmax, 3000)

        for (i, (m1, m2)) in enumerate(_PLK_XSPECS)
            f1 = PLK_FREQS[m1]
            f2 = PLK_FREQS[m2]
            out[i, :] .+= shared.Atsz .* dl_tsz .* (tsz_sed[f1] * tsz_sed[f2])
            out[i, :] .+= shared.Aksz .* dl_ksz
            out[i, :] .+= shared.Acib .* dl_cib .* (cib_sed[f1] * cib_sed[f2])
            out[i, :] .+= -shared.xi * sqrt(shared.Acib * shared.Atsz) .* dl_szxcib .*
                          (tsz_sed[f1] * cib_sed[f2] + tsz_sed[f2] * cib_sed[f1])
            out[i, :] .+= plk.radio_TT .* dl_ps .* (radio_sed[f1] * radio_sed[f2])
            out[i, :] .+= plk.cib_ps .* dl_ps .* (dusty_sed[f1] * dusty_sed[f2])
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# ACT foregrounds (beam chromaticity SEDs, 15 cross-frequency pairs)
# ---------------------------------------------------------------------------

"""
    _act_sed_rr(data, m, kind, beta, T, shift) -> Vector (n_ell,)

Chromatic SED for the frozen foreground SED shapes. The fixed-beam projection
and chromatic ratio are delegated to CMBForegrounds, whose ChainRules/Mooncake
rules suppress tangents for the released beam matrix. `kind` is `:dust`,
`:cib`, `:dusty`, `:radio`, or `:tsz`. `beta` is the spectral index (unused
for `:tsz`); `T` is the CIB temperature (used by `:cib`/`:dusty`, otherwise
ignored); `shift` is the band-centre shift in GHz. Numerically identical to
the frozen closure-based row-wise trapezoid (`weighted_bandpass` matvec form).
Flat scalar signature on purpose:
tuple-typed differentiable arguments break `@from_chainrules`.
"""
function _act_sed_rr(data::ACTData, m::String, kind::Symbol, beta::Real, T::Real, shift::Real)
    nus = data.bandpass_nu[m] .+ shift
    u = data.weighted_bandpass[m] .* _fg_dbdt.(nus)
    beam = data.beams[m]            # (n_ell, n_nu), normalized at ℓ=0
    r = [_act_sed_ratio_val(kind, beta, T, nu) for nu in nus]
    denominator = _fixed_beam_product(beam, u)
    return _fixed_chromatic_ratio(beam, u, r, denominator)
end

# SED ratio value at one frequency (normalized at 150 GHz)
function _act_sed_ratio_val(kind::Symbol, beta::Real, T::Real, nu::Real)
    if kind === :dust
        return _fg_dust_ratio(nu, beta)
    elseif kind === :radio
        return _fg_radio_ratio(nu, beta)
    elseif kind === :cib || kind === :dusty
        return _fg_cib_ratio(nu, beta, T)
    else  # :tsz
        return _fg_tsz_ratio(nu)
    end
end

# ---------------------------------------------------------------------------
# ACT foregrounds (beam chromaticity SEDs, 15 cross-frequency pairs)
# ---------------------------------------------------------------------------

"""Materialize map-ordered chromatic SED rows for CMBForegrounds kernels."""
_act_sed_matrix(sed::Dict{String,<:AbstractVector}, maps) =
    permutedims(reduce(hcat, (sed[m] for m in maps)))

"""Select the 15 released ACT cross pairs from a full map×map×ℓ cube."""
function _act_cross_pair_slice(data::ACTData, dl::AbstractArray{<:Real,3})
    T = eltype(dl)
    out = Matrix{T}(undef, length(data.cross_pairs), size(dl, 3))
    for i in eachindex(data.cross_pairs)
        @views out[i, :] .= dl[data.cross_pair_i1[i], data.cross_pair_i2[i], :]
    end
    return out
end

"""
    act_foregrounds(data, tpl, shared, act, mode) -> Matrix (15, 8502)

ACT per-cross-frequency foreground D_l for `mode ∈ (:TT, :TE, :ET, :EE)`.
TT: dust + tSZ + kSZ + clustered CIB + SZxCIB + radio PS + dusty PS.
TE/ET/EE: dust only (frozen foregrounds.yaml; ET reuses the TE instance).
"""
function act_foregrounds(
    data::ACTData,
    tpl::SharedTemplates,
    shared::SharedForegroundParameters,
    act::ACTForegroundParameters,
    mode::Symbol,
)
    return _act_foregrounds_flat(
        data, tpl, mode, _act_foreground_parameters(shared, act))
end

function _act_foreground_parameters(
    shared::SharedForegroundParameters, act::ACTForegroundParameters,
)
    return [
        shared.Atsz, shared.Acib, shared.Aksz, shared.xi,
        shared.beta_cib, shared.beta_radio, shared.beta_dusty,
        shared.T_cib, shared.alpha_tsz,
        act.AdustTT, act.AdustTE, act.AdustEE,
        act.beta_dustTT, act.beta_dustTE, act.beta_dustEE,
        act.alpha_dustTT, act.alpha_dustTE, act.alpha_dustEE,
        act.radio_TT, act.radio_EE, act.cib_ps,
        (act.band_shift[m] for m in _ACT_MAPS)...,
    ]
end

function _act_foregrounds_flat(
    data::ACTData, tpl::SharedTemplates, mode::Symbol, x::AbstractVector{<:Real},
)
    return first(_act_foregrounds_flat_cached(data, tpl, mode, x))
end

function _act_foregrounds_flat_cached(
    data::ACTData, tpl::SharedTemplates, mode::Symbol, x::AbstractVector{<:Real},
)
    length(x) == 26 || throw(DimensionMismatch("ACT foreground parameter vector must have 26 elements"))
    shared = SharedForegroundParameters(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
    act = ACTForegroundParameters(
        x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18],
        x[19], x[20], x[21], Dict(_ACT_MAPS[i] => x[21 + i] for i in 1:5),
    )
    return _act_foregrounds_impl_cached(data, tpl, shared, act, mode)
end

function _act_foregrounds_impl(
    data::ACTData,
    tpl::SharedTemplates,
    shared::SharedForegroundParameters,
    act::ACTForegroundParameters,
    mode::Symbol,
)
    return first(_act_foregrounds_impl_cached(data, tpl, shared, act, mode))
end

function _act_foregrounds_impl_cached(
    data::ACTData,
    tpl::SharedTemplates,
    shared::SharedForegroundParameters,
    act::ACTForegroundParameters,
    mode::Symbol,
)
    lmax = size(first(values(data.beams)), 1) - 1   # 8501 (beam ell grid 0..lmax_bpw)
    maps = data.map_names
    shift(m) = get(act.band_shift, m, 0.0)

    # dust (lnorm = 500); ET reuses the TE parameters
    key = mode == :ET ? :TE : mode
    beta = key == :TT ? act.beta_dustTT : key == :TE ? act.beta_dustTE : act.beta_dustEE
    alpha = key == :TT ? act.alpha_dustTT : key == :TE ? act.alpha_dustTE : act.alpha_dustEE
    ad = key == :TT ? act.AdustTT : key == :TE ? act.AdustTE : act.AdustEE
    dust_sed = Dict(m => _act_sed_rr(data, m, :dust, beta, 0.0, shift(m)) for m in maps)
    dlg = _fg_powerlaw(alpha, lmax, 500)

    dust = _act_sed_matrix(dust_sed, maps)

    if mode == :TT
        tsz_sed = Dict(m => _act_sed_rr(data, m, :tsz, 0.0, 0.0, shift(m)) for m in maps)
        cib_sed = Dict(m => _act_sed_rr(data, m, :cib, shared.beta_cib, shared.T_cib, shift(m)) for m in maps)
        radio_sed = Dict(m => _act_sed_rr(data, m, :radio, shared.beta_radio, 0.0, shift(m)) for m in maps)
        dusty_sed = Dict(m => _act_sed_rr(data, m, :dusty, shared.beta_dusty, shared.T_cib, shift(m)) for m in maps)

        dl_tsz = _fg_tsz_template(tpl, lmax, shared.alpha_tsz)
        dl_ksz = _fg_read_template(tpl, :ksz, lmax, 3000)
        dl_cib = _fg_read_template(tpl, :cib, lmax, 3000)
        dl_szxcib = _fg_read_template(tpl, :szxcib, lmax, 3000)
        dl_ps = _fg_powerlaw(0.0, lmax, 3000)

        xirt = -shared.xi * sqrt(shared.Acib * shared.Atsz)
        tsz = _act_sed_matrix(tsz_sed, maps)
        cib = _act_sed_matrix(cib_sed, maps)
        radio = _act_sed_matrix(radio_sed, maps)
        dusty = _act_sed_matrix(dusty_sed, maps)
        ones_sed = ones(eltype(dust), size(dust))
        full = assemble_TT(
            act.cib_ps, ad, act.radio_TT,
            ones_sed, dusty, dust, radio, tsz, cib,
            shared.Aksz .* dl_ksz, dl_ps, dlg, dl_ps,
            shared.Atsz .* dl_tsz, shared.Acib .* dl_cib, xirt .* dl_szxcib,
        )
        cache = (
            dust=dust_sed, tsz=tsz_sed, cib=cib_sed, radio=radio_sed,
            dusty=dusty_sed, dust_shape=dlg, dl_tsz=dl_tsz, dl_ksz=dl_ksz,
            dl_cib=dl_cib, dl_szxcib=dl_szxcib, dl_ps=dl_ps,
        )
        return _act_cross_pair_slice(data, full), cache
    end

    # ACT TE/EE carry Galactic dust only in the frozen model. The zero radio
    # amplitude keeps the common CMBForegrounds fused kernels while preserving
    # that release convention exactly.
    full = mode in (:TE, :ET) ?
        assemble_TE(zero(ad), ad, dust, dust, dust, dust, dlg, dlg) :
        assemble_EE(zero(ad), ad, dust, dust, dlg, dlg)
    return _act_cross_pair_slice(data, full), (dust=dust_sed, dust_shape=dlg)
end

# ---------------------------------------------------------------------------
# SPT sky model (feff SEDs, 6 TT + 9 TE + 6 EE blocks, SSL + aberration)
# ---------------------------------------------------------------------------

const _SPT_FEFF_DUST = Dict(90 => 95.9631, 150 => 150.012, 220 => 222.773)
const _SPT_FEFF_TSZ = Dict(90 => 95.6933, 150 => 148.849, 220 => 220.15)
const _SPT_FEFF_CIB = Dict(90 => 95.96, 150 => 150.00, 220 => 222.76)
const _SPT_FEFF_RADIO = Dict(90 => 94.40, 150 => 146.00, 220 => 212.7)
const _SPT_FREQS = (90, 150, 220)

"""Cross-frequency pairs per mode, in the frozen cross-list order."""
function _spt_pairs(mode::Symbol)
    if mode == :TT
        return ((90, 90), (90, 150), (90, 220), (150, 150), (150, 220), (220, 220))
    elseif mode == :TE
        return ((90, 90), (90, 150), (150, 90), (90, 220), (220, 90), (150, 150),
            (150, 220), (220, 150), (220, 220))
    else
        return ((90, 90), (90, 150), (90, 220), (150, 150), (150, 220), (220, 220))
    end
end

"""
    spt_sky_model(data, tpl, theory, shared, spt) -> Vector (21 * 4094)

SPT-3G D1 pre-instrument sky model, 21 blocks in release order on the artifact
ℓ grid 2..4095: CMB + super-sample lensing + aberration + foregrounds.
TT: dust (plain power-law class, lnorm=80) + tSZ + kSZ + clustered CIB + SZxCIB
(template) + radio PS + dusty PS. TE/EE: dust + radio PS.
"""
function spt_sky_model(
    data::SPTData,
    tpl::SharedTemplates,
    theory::JointCMBTheory,
    shared::SharedForegroundParameters,
    spt::SPTForegroundParameters,
)
    return _spt_sky_flat(data, tpl, theory, _spt_sky_parameters(shared, spt))
end

function _spt_sky_parameters(shared::SharedForegroundParameters, spt::SPTForegroundParameters)
    return [
        shared.Atsz, shared.Acib, shared.Aksz, shared.xi,
        shared.beta_cib, shared.beta_radio, shared.beta_dusty,
        shared.T_cib, shared.alpha_tsz,
        spt.AdustTT, spt.AdustTE, spt.AdustEE,
        spt.beta_dustTT, spt.beta_dustTE, spt.beta_dustEE,
        spt.alpha_dustTT, spt.alpha_dustTE, spt.alpha_dustEE,
        spt.radio_TT, spt.radio_TE, spt.radio_EE, spt.cib_ps, spt.kappa,
    ]
end

function _spt_sky_flat(
    data::SPTData,
    tpl::SharedTemplates,
    theory::JointCMBTheory,
    x::AbstractVector{<:Real},
)
    length(x) == 23 || throw(DimensionMismatch("SPT sky parameter vector must have 23 elements"))
    shared = SharedForegroundParameters(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
    spt = SPTForegroundParameters(
        x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18],
        x[19], x[20], x[21], x[22], x[23],
    )
    return _spt_sky_model_impl(data, tpl, theory, shared, spt)
end

function _spt_sky_model_impl(
    data::SPTData,
    tpl::SharedTemplates,
    theory::JointCMBTheory,
    shared::SharedForegroundParameters,
    spt::SPTForegroundParameters,
)
    ells = data.ells                    # 2..4095
    n_ell = length(ells)
    lmax = ells[end]                    # 4095
    freqs = _SPT_FREQS

    # theory D_l on the artifact grid
    dl_cmb = Dict(:TT => theory_slice(theory, :TT, first(ells), last(ells)),
                  :TE => theory_slice(theory, :TE, first(ells), last(ells)),
                  :EE => theory_slice(theory, :EE, first(ells), last(ells)))

    # SEDs
    dust_sed = Dict(mode => Dict(f => _fg_dust_ratio(_SPT_FEFF_DUST[f],
            mode == :TT ? spt.beta_dustTT : mode == :TE ? spt.beta_dustTE : spt.beta_dustEE)
        for f in freqs) for mode in (:TT, :TE, :EE))
    tsz_sed = Dict(f => _fg_tsz_ratio(_SPT_FEFF_TSZ[f]) for f in freqs)
    cib_sed = Dict(f => _fg_cib_ratio(_SPT_FEFF_CIB[f], shared.beta_cib, shared.T_cib) for f in freqs)
    radio_sed = Dict(f => _fg_radio_ratio(_SPT_FEFF_RADIO[f], shared.beta_radio) for f in freqs)
    dusty_sed = Dict(f => _fg_cib_ratio(_SPT_FEFF_CIB[f], shared.beta_dusty, shared.T_cib) for f in freqs)

    # templates / power laws on the full 0..lmax grid
    dl_tsz = _fg_tsz_template(tpl, lmax, shared.alpha_tsz)
    dl_ksz = _fg_read_template(tpl, :ksz, lmax, 3000)
    dl_cib = _fg_read_template(tpl, :cib, lmax, 3000)
    dl_szxcib = _fg_read_template(tpl, :szxcib, lmax, 3000)
    dl_ps = _fg_powerlaw(0.0, lmax, 3000)
    dl_dust = Dict(mode => _fg_powerlaw(
        mode == :TT ? spt.alpha_dustTT : mode == :TE ? spt.alpha_dustTE : spt.alpha_dustEE,
        lmax, 80) for mode in (:TT, :TE, :EE))

    # per-mode foreground blocks on the ℓ≥2 grid
    Tsky = typeof(dust_sed[:TT][90] * dl_dust[:TT][3])
    fg = Dict{Symbol,Dict{Tuple{Int,Int},Vector{Tsky}}}()
    for mode in (:TT, :TE, :EE)
        ad = mode == :TT ? spt.AdustTT : mode == :TE ? spt.AdustTE : spt.AdustEE
        radio_amp = mode == :TT ? spt.radio_TT : mode == :TE ? spt.radio_TE : spt.radio_EE
        fgm = Dict{Tuple{Int,Int},Vector{Tsky}}()
        xirt = -shared.xi * sqrt(shared.Acib * shared.Atsz)
        for (f1, f2) in _spt_pairs(mode)
            # one fused broadcast per pair (tape-entry count, see act_foregrounds)
            if mode == :TT
                v = @. ad * dl_dust[mode][3:end] * (dust_sed[mode][f1] * dust_sed[mode][f2]) +
                    radio_amp * dl_ps[3:end] * (radio_sed[f1] * radio_sed[f2]) +
                    shared.Atsz * dl_tsz[3:end] * (tsz_sed[f1] * tsz_sed[f2]) +
                    shared.Aksz * dl_ksz[3:end] +
                    shared.Acib * dl_cib[3:end] * (cib_sed[f1] * cib_sed[f2]) +
                    xirt * dl_szxcib[3:end] * (tsz_sed[f1] * cib_sed[f2] + tsz_sed[f2] * cib_sed[f1]) +
                    spt.cib_ps * dl_ps[3:end] * (dusty_sed[f1] * dusty_sed[f2])
            else
                v = @. ad * dl_dust[mode][3:end] * (dust_sed[mode][f1] * dust_sed[mode][f2]) +
                    radio_amp * dl_ps[3:end] * (radio_sed[f1] * radio_sed[f2])
            end
            fgm[(f1, f2)] = v
        end
        fg[mode] = fgm
    end

    # sky blocks: CMB + SSL + aberration (sequential) + foreground
    sky = Vector{Tsky}(undef, 21 * n_ell)
    for (b, (mode, i1, i2)) in enumerate(_SPT_SPECS)
        msym = Symbol(mode)
        f1 = freqs[i1]
        f2 = freqs[i2]
        block = dl_cmb[msym] .+ _spt_ssl(spt.kappa, dl_cmb[msym], ells)
        block .+= _spt_aberration(_SPT_ABERRATION, block, ells)
        # foreground: TE keeps the ordered pair, TT/EE use the sorted pair
        pair = msym == :TE ? (f1, f2) : (minmax(f1, f2)[1], minmax(f1, f2)[2])
        block .+= fg[msym][pair]
        copyto!(view(sky, (b-1)*n_ell+1:b*n_ell), block)
    end
    return sky
end

"""Super-sample lensing: -k (ℓ dC_ℓ/dℓ ℓ(ℓ+1)/2π + 2 D_ℓ), np.gradient derivative."""
function _spt_ssl(kappa, Dl::AbstractVector, ells)
    ll2pi = [l * (l + 1) / 2π for l in ells]
    Cl = Dl ./ ll2pi
    dCl = _np_gradient(Cl)
    return @. -kappa * ((ells * dCl) * ll2pi + 2 * Dl)
end

"""Aberration: -ab ℓ dC_ℓ/dℓ ℓ(ℓ+1)/2π (np.gradient derivative)."""
function _spt_aberration(ab, Dl::AbstractVector, ells)
    ll2pi = [l * (l + 1) / 2π for l in ells]
    Cl = Dl ./ ll2pi
    dCl = _np_gradient(Cl)
    return @. -ab * (ells * dCl) * ll2pi
end

"""np.gradient: central differences interior, one-sided at the endpoints."""
function _np_gradient(y::AbstractVector)
    n = length(y)
    g = similar(y)
    g[1] = y[2] - y[1]
    g[n] = y[n] - y[n-1]
    for i in 2:n-1
        g[i] = (y[i+1] - y[i-1]) / 2
    end
    return g
end
