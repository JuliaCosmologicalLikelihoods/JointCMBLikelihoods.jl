module JointCMBLikelihoodsChainRulesCoreExt

using JointCMBLikelihoods:
    PlanckData, ACTData, SPTData, SharedTemplates, JointCMBTheory,
    planck_chi2, act_chi2, spt_chi2,
    _act_window_project, _spt_window_project, _act_residual_flat,
    _planck_residual_flat, _xfreq_cut,
    _act_sed_rr, _act_sed_ratio_val, _fg_dbdt,
    _fg_powerlaw, _fg_tsz_template,
    _fg_dust_ratio, _fg_cib_ratio, _fg_radio_ratio, _fg_tsz_ratio,
    _fg_read_template, _spt_sky_flat, _spt_pairs, _spt_ssl, _spt_aberration,
    _planck_foregrounds_flat, _plk_dust_index, _plk_freq_index,
    _act_foregrounds_flat, _act_foregrounds_flat_cached, _ACT_MAPS,
    theory_slice,
    _SPT_FREQS, _SPT_FEFF_DUST, _SPT_FEFF_TSZ, _SPT_FEFF_CIB,
    _SPT_FEFF_RADIO, _SPT_ABERRATION, _SPT_SPECS,
    _PLK_XSPECS, PLK_FREQS, _PLK_FEFF_DUST, _PLK_FEFF_TSZ, _PLK_FEFF_RADIO,
    _FG_H_PL, _FG_K_B, _FG_T_CMB, _FG_T_DUST,
    _np_gradient, _spt_temperature_beam,
    _spt_polarization_beam, _spt_t2p_kernels, _spt_residual_flat,
    _SPT_T2P_SIGMAS, _SPT_SPECS, _spt_tt_block, _spt_te_block,
    _SPT_CAL, _SPT_CAL_FREQ, _SPT_PE_FREQ, _SPT_BETA_POL,
    _SPT_BETA_MODES, _SPT_T2P
using ChainRulesCore: rrule, NoTangent, @thunk, unthunk
import ChainRulesCore
using LinearAlgebra: dot, transpose

# The fixed release data (covariances) must never receive a tangent.
ChainRulesCore.ProjectTo(::PlanckData) = identity  # not used; data is not returned

# --- Planck: fixed float32 inv_cov (cached float64 copy); the float32 cast
# and 8-digit rounding are frozen numerical conventions, treated as identity
# for the pullback (they are ~1e-7-level primal effects, far below any
# gradient comparison tolerance).
function ChainRulesCore.rrule(::typeof(planck_chi2), data::PlanckData, delta::AbstractVector{<:Real})
    chi2 = planck_chi2(data, delta)
    function planck_chi2_pullback(Δ)
        d64 = Vector{Float64}(delta)
        cot = 2 .* (data.inv_cov64 * d64)
        return NoTangent(), NoTangent(), Δ .* cot
    end
    return chi2, planck_chi2_pullback
end

# --- ACT: fixed inv_cov, plain float64 quadratic form.
function ChainRulesCore.rrule(::typeof(act_chi2), data::ACTData, delta::AbstractVector{<:Real})
    chi2 = act_chi2(data, delta)
    function act_chi2_pullback(Δ)
        cot = 2 .* (data.inv_cov * delta)
        return NoTangent(), NoTangent(), Δ .* cot
    end
    return chi2, act_chi2_pullback
end

# --- SPT: fixed covariance, inverted once at load. This comment previously
# claimed the pullback used a precomputed inverse while the code refactorized
# `data.cov` on every call, so a gradient paid for two full factorizations.
function ChainRulesCore.rrule(::typeof(spt_chi2), data::SPTData, delta::AbstractVector{<:Real})
    chi2 = spt_chi2(data, delta)
    function spt_chi2_pullback(Δ)
        cot = 2 .* (data.inv_cov * delta)
        return NoTangent(), NoTangent(), Δ .* cot
    end
    return chi2, spt_chi2_pullback
end

# --- Released ACT/SPT windows are fixed data. Keeping the data struct as the
# argument (rather than passing `data.windows[i]`) prevents Mooncake from
# constructing a `Vector{Matrix}` tangent for the complete release window set.
function ChainRulesCore.rrule(::typeof(_act_window_project), data::ACTData,
                              i::Int, spectrum::AbstractVector{<:Real})
    projection = _act_window_project(data, i, spectrum)
    function act_window_pullback(Δ)
        return NoTangent(), NoTangent(), NoTangent(), data.windows[i] * unthunk(Δ)
    end
    return projection, act_window_pullback
end

function ChainRulesCore.rrule(::typeof(_spt_window_project), data::SPTData,
                              b::Int, spectrum::AbstractVector{<:Real})
    projection = _spt_window_project(data, b, spectrum)
    function spt_window_pullback(Δ)
        return NoTangent(), NoTangent(), NoTangent(), data.windows[b] * unthunk(Δ)
    end
    return projection, spt_window_pullback
end

@inline function _act_dbdt_logderivative(nu)
    x = _FG_H_PL * (nu * 1e9) / (_FG_K_B * _FG_T_CMB)
    return (4 + x * (1 - 2 * exp(x) / expm1(x))) / nu
end


@inline function _act_planck_logderivative_nu(nu, temperature)
    x = _FG_H_PL * (nu * 1e9) / (_FG_K_B * temperature)
    return (3 - x * exp(x) / expm1(x)) / nu
end

@inline function _act_planck_logderivative_temperature(nu, temperature)
    x = _FG_H_PL * (nu * 1e9) / (_FG_K_B * temperature)
    return x * exp(x) / (temperature * expm1(x))
end

function _act_sed_partials(kind::Symbol, beta, temperature, nu, ratio)
    beta_partial = zero(ratio)
    temperature_partial = zero(ratio)
    if kind === :dust
        beta_partial = ratio * log(nu / 150)
        nu_partial = ratio * (
            beta / nu + _act_planck_logderivative_nu(nu, _FG_T_DUST) -
            _act_dbdt_logderivative(nu))
    elseif kind === :cib || kind === :dusty
        beta_partial = ratio * log(nu / 150)
        temperature_partial = ratio * (
            _act_planck_logderivative_temperature(nu, temperature) -
            _act_planck_logderivative_temperature(150.0, temperature))
        nu_partial = ratio * (
            beta / nu + _act_planck_logderivative_nu(nu, temperature) -
            _act_dbdt_logderivative(nu))
    elseif kind === :radio
        beta_partial = ratio * log(nu / 150)
        nu_partial = ratio * (beta / nu - _act_dbdt_logderivative(nu))
    else
        x = _FG_H_PL * (nu * 1e9) / (_FG_K_B * _FG_T_CMB)
        x0 = _FG_H_PL * (150.0 * 1e9) / (_FG_K_B * _FG_T_CMB)
        g0 = x0 / tanh(x0 / 2) - 4
        dgdx = inv(tanh(x / 2)) - x / (2 * sinh(x / 2)^2)
        nu_partial = dgdx * (x / nu) / g0
    end
    return beta_partial, temperature_partial, nu_partial
end

# --- Complete ACT chromatic SED. Contract the 8502-element output cotangent
# through the two fixed-beam products, then analytically differentiate only the
# short native bandpass vectors and three scalar inputs.
function ChainRulesCore.rrule(
    ::typeof(_act_sed_rr), data::ACTData, map_name::String, kind::Symbol,
    beta::Real, temperature::Real, shift::Real,
)
    output = _act_sed_rr(data, map_name, kind, beta, temperature, shift)
    nus = data.bandpass_nu[map_name] .+ shift
    weighted_bandpass = data.weighted_bandpass[map_name]
    dbdt = _fg_dbdt.(nus)
    weights = weighted_bandpass .* dbdt
    ratios = [_act_sed_ratio_val(kind, beta, temperature, nu) for nu in nus]
    beam = data.beams[map_name]
    denominator = beam * weights

    function act_sed_pullback(Δ)
        output_bar = unthunk(Δ)
        numerator_bar = output_bar ./ denominator
        weighted_sed_bar = transpose(beam) * numerator_bar
        denominator_bar = -numerator_bar .* output
        weights_bar = ratios .* weighted_sed_bar + transpose(beam) * denominator_bar
        ratios_bar = weights .* weighted_sed_bar

        Tbar = promote_type(eltype(output_bar), typeof(beta), typeof(temperature), typeof(shift))
        beta_bar = zero(Tbar)
        temperature_bar = zero(Tbar)
        shift_bar = zero(Tbar)
        for i in eachindex(nus)
            beta_partial, temperature_partial, nu_partial =
                _act_sed_partials(kind, beta, temperature, nus[i], ratios[i])
            beta_bar += ratios_bar[i] * beta_partial
            temperature_bar += ratios_bar[i] * temperature_partial
            shift_bar += ratios_bar[i] * nu_partial
            shift_bar += weights_bar[i] * weighted_bandpass[i] * dbdt[i] *
                         _act_dbdt_logderivative(nus[i])
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(),
            beta_bar, temperature_bar, shift_bar
    end
    return output, act_sed_pullback
end

# --- Foreground angular shapes have one active scalar and thousands of output
# multipoles. Their exact scalar VJPs avoid recording a power and normalization
# graph for every ell.
function ChainRulesCore.rrule(
    ::typeof(_fg_powerlaw), alpha::Real, lmax::Int, lnorm::Int,
)
    output = _fg_powerlaw(alpha, lmax, lnorm)
    function fg_powerlaw_pullback(Δ)
        output_bar = unthunk(Δ)
        alpha_bar = zero(promote_type(eltype(output_bar), typeof(alpha)))
        for l in 2:lmax
            alpha_bar += output_bar[l + 1] * output[l + 1] * log(l / lnorm)
        end
        return NoTangent(), alpha_bar, NoTangent(), NoTangent()
    end
    return output, fg_powerlaw_pullback
end

function ChainRulesCore.rrule(
    ::typeof(_fg_tsz_template), templates, lmax::Int, alpha::Real,
)
    output = _fg_tsz_template(templates, lmax, alpha)
    function fg_tsz_template_pullback(Δ)
        output_bar = unthunk(Δ)
        alpha_bar = zero(promote_type(eltype(output_bar), typeof(alpha)))
        for l in 2:lmax
            alpha_bar += output_bar[l + 1] * output[l + 1] * log(l / 3000)
        end
        return NoTangent(), NoTangent(), NoTangent(), alpha_bar
    end
    return output, fg_tsz_template_pullback
end

@inline _spt_mode_indices(mode::Symbol) =
    mode === :TT ? (10, 13, 16, 19) :
    mode === :TE ? (11, 14, 17, 20) : (12, 15, 18, 21)

# --- Complete SPT pre-instrument sky. All release data and CMB theory are
# fixed; the pullback contracts the 21 sky-block cotangents directly into the
# 23 shared/SPT foreground scalars.
function ChainRulesCore.rrule(
    ::typeof(_spt_sky_flat), data::SPTData, templates::SharedTemplates,
    theory::JointCMBTheory, x::AbstractVector{<:Real},
)
    sky = _spt_sky_flat(data, templates, theory, x)
    function spt_sky_flat_pullback(Δ)
        skybar = unthunk(Δ)
        Tbar = promote_type(eltype(skybar), eltype(x))
        xbar = zeros(Tbar, length(x))
        ells = data.ells
        n_ell = length(ells)
        lmax = last(ells)

        cmb = Dict(
            :TT => theory_slice(theory, :TT, first(ells), last(ells)),
            :TE => theory_slice(theory, :TE, first(ells), last(ells)),
            :EE => theory_slice(theory, :EE, first(ells), last(ells)),
        )
        kappa_response = Dict{Symbol,Vector{Float64}}()
        for mode in (:TT, :TE, :EE)
            unit = _spt_ssl(one(x[23]), cmb[mode], ells)
            kappa_response[mode] = unit .+ _spt_aberration(_SPT_ABERRATION, unit, ells)
        end

        dl_tsz = @view(_fg_tsz_template(templates, lmax, x[9])[3:end])
        dl_tsz_alpha = dl_tsz .* log.(ells ./ 3000)
        dl_ksz = @view(_fg_read_template(templates, :ksz, lmax, 3000)[3:end])
        dl_cib = @view(_fg_read_template(templates, :cib, lmax, 3000)[3:end])
        dl_szxcib = @view(_fg_read_template(templates, :szxcib, lmax, 3000)[3:end])
        dl_ps = @view(_fg_powerlaw(zero(x[1]), lmax, 3000)[3:end])

        dust_shape = Dict{Symbol,Vector{eltype(x)}}()
        dust_alpha_shape = Dict{Symbol,Vector{eltype(x)}}()
        dust_sed = Dict{Symbol,Dict{Int,eltype(x)}}()
        dust_beta_sed = Dict{Symbol,Dict{Int,eltype(x)}}()
        for mode in (:TT, :TE, :EE)
            _, beta_index, alpha_index, _ = _spt_mode_indices(mode)
            shape = _fg_powerlaw(x[alpha_index], lmax, 80)[3:end]
            dust_shape[mode] = shape
            dust_alpha_shape[mode] = shape .* log.(ells ./ 80)
            dust_sed[mode] = Dict(f => _fg_dust_ratio(_SPT_FEFF_DUST[f], x[beta_index])
                                  for f in _SPT_FREQS)
            dust_beta_sed[mode] = Dict(f => dust_sed[mode][f] *
                log(_SPT_FEFF_DUST[f] / 150) for f in _SPT_FREQS)
        end

        tsz_sed = Dict(f => _fg_tsz_ratio(_SPT_FEFF_TSZ[f]) for f in _SPT_FREQS)
        cib_sed = Dict(f => _fg_cib_ratio(_SPT_FEFF_CIB[f], x[5], x[8])
                       for f in _SPT_FREQS)
        cib_beta_sed = Dict(f => cib_sed[f] * log(_SPT_FEFF_CIB[f] / 150)
                            for f in _SPT_FREQS)
        cib_temperature_sed = Dict(f => cib_sed[f] * (
            _act_planck_logderivative_temperature(_SPT_FEFF_CIB[f], x[8]) -
            _act_planck_logderivative_temperature(150.0, x[8])) for f in _SPT_FREQS)
        radio_sed = Dict(f => _fg_radio_ratio(_SPT_FEFF_RADIO[f], x[6])
                         for f in _SPT_FREQS)
        radio_beta_sed = Dict(f => radio_sed[f] * log(_SPT_FEFF_RADIO[f] / 150)
                              for f in _SPT_FREQS)
        dusty_sed = Dict(f => _fg_cib_ratio(_SPT_FEFF_CIB[f], x[7], x[8])
                         for f in _SPT_FREQS)
        dusty_beta_sed = Dict(f => dusty_sed[f] * log(_SPT_FEFF_CIB[f] / 150)
                              for f in _SPT_FREQS)
        dusty_temperature_sed = Dict(f => dusty_sed[f] * (
            _act_planck_logderivative_temperature(_SPT_FEFF_CIB[f], x[8]) -
            _act_planck_logderivative_temperature(150.0, x[8])) for f in _SPT_FREQS)

        root_ac = sqrt(x[1] * x[2])
        xirt = -x[4] * root_ac
        for (b, (mode_string, i1, i2)) in enumerate(_SPT_SPECS)
            mode = Symbol(mode_string)
            f1 = _SPT_FREQS[i1]
            f2 = _SPT_FREQS[i2]
            bar = @view skybar[(b - 1) * n_ell + 1:b * n_ell]
            xbar[23] += dot(bar, kappa_response[mode])

            ad_index, beta_index, alpha_index, radio_index = _spt_mode_indices(mode)
            d1, d2 = dust_sed[mode][f1], dust_sed[mode][f2]
            db1, db2 = dust_beta_sed[mode][f1], dust_beta_sed[mode][f2]
            dust_pair = d1 * d2
            xbar[ad_index] += dot(bar, dust_shape[mode]) * dust_pair
            xbar[alpha_index] += x[ad_index] * dot(bar, dust_alpha_shape[mode]) * dust_pair
            xbar[beta_index] += x[ad_index] * dot(bar, dust_shape[mode]) *
                                (db1 * d2 + d1 * db2)

            r1, r2 = radio_sed[f1], radio_sed[f2]
            rb1, rb2 = radio_beta_sed[f1], radio_beta_sed[f2]
            xbar[radio_index] += dot(bar, dl_ps) * r1 * r2
            xbar[6] += x[radio_index] * dot(bar, dl_ps) * (rb1 * r2 + r1 * rb2)

            mode === :TT || continue
            t1, t2 = tsz_sed[f1], tsz_sed[f2]
            c1, c2 = cib_sed[f1], cib_sed[f2]
            cb1, cb2 = cib_beta_sed[f1], cib_beta_sed[f2]
            ct1, ct2 = cib_temperature_sed[f1], cib_temperature_sed[f2]
            xbar[1] += dot(bar, dl_tsz) * t1 * t2
            xbar[9] += x[1] * dot(bar, dl_tsz_alpha) * t1 * t2
            xbar[3] += dot(bar, dl_ksz)
            xbar[2] += dot(bar, dl_cib) * c1 * c2
            xbar[5] += x[2] * dot(bar, dl_cib) * (cb1 * c2 + c1 * cb2)
            xbar[8] += x[2] * dot(bar, dl_cib) * (ct1 * c2 + c1 * ct2)

            cross_sed = t1 * c2 + t2 * c1
            cross_contract = dot(bar, dl_szxcib)
            xbar[4] -= root_ac * cross_sed * cross_contract
            xbar[1] += (-x[4] * x[2] / (2root_ac)) * cross_sed * cross_contract
            xbar[2] += (-x[4] * x[1] / (2root_ac)) * cross_sed * cross_contract
            xbar[5] += xirt * (t1 * cb2 + t2 * cb1) * cross_contract
            xbar[8] += xirt * (t1 * ct2 + t2 * ct1) * cross_contract

            p1, p2 = dusty_sed[f1], dusty_sed[f2]
            pb1, pb2 = dusty_beta_sed[f1], dusty_beta_sed[f2]
            pt1, pt2 = dusty_temperature_sed[f1], dusty_temperature_sed[f2]
            ps_contract = dot(bar, dl_ps)
            xbar[22] += ps_contract * p1 * p2
            xbar[7] += x[22] * ps_contract * (pb1 * p2 + p1 * pb2)
            xbar[8] += x[22] * ps_contract * (pt1 * p2 + p1 * pt2)
        end

        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), xbar
    end
    return sky, spt_sky_flat_pullback
end

@inline _planck_beta_index(mode::Symbol) =
    mode === :TT ? 22 : mode === :TE ? 23 : mode === :ET ? 24 : 25

@inline function _planck_alpha_index(mode::Symbol, frequency::Int)
    mode === :TT && return 25 + _plk_freq_index(frequency)
    return mode === :TE ? 29 : mode === :ET ? 30 : 31
end

# --- Complete Planck foreground mode. Contract one 15x2501 mode cotangent
# directly into the 33 shared/Planck foreground scalars.
function ChainRulesCore.rrule(
    ::typeof(_planck_foregrounds_flat), templates::SharedTemplates,
    mode::Symbol, x::AbstractVector{<:Real},
)
    foreground = _planck_foregrounds_flat(templates, mode, x)
    function planck_foregrounds_flat_pullback(Δ)
        foreground_bar = unthunk(Δ)
        Tbar = promote_type(eltype(foreground_bar), eltype(x))
        xbar = zeros(Tbar, length(x))
        lmax = 2500
        ells = collect(0:lmax)
        beta_index = _planck_beta_index(mode)
        dust_sed = Dict(f => _fg_dust_ratio(_PLK_FEFF_DUST[f], x[beta_index])
                        for f in (100, 143, 217))
        dust_beta_sed = Dict(f => dust_sed[f] * log(_PLK_FEFF_DUST[f] / 150)
                             for f in (100, 143, 217))
        dust_shapes = Dict{Int,Vector{eltype(x)}}()
        dust_alpha_shapes = Dict{Int,Vector{eltype(x)}}()
        for frequency in (100, 143, 217)
            alpha_index = _planck_alpha_index(mode, frequency)
            shape = _fg_powerlaw(x[alpha_index], lmax, 200)
            dust_shapes[frequency] = shape
            derivative = zeros(eltype(x), length(shape))
            for l in 2:lmax
                derivative[l + 1] = shape[l + 1] * log(l / 200)
            end
            dust_alpha_shapes[frequency] = derivative
        end

        if mode === :TT
            tsz_sed = Dict(f => _fg_tsz_ratio(_PLK_FEFF_TSZ[f]) for f in (100, 143, 217))
            cib_sed = Dict(f => _fg_cib_ratio(_PLK_FEFF_DUST[f], x[5], x[8])
                           for f in (100, 143, 217))
            cib_beta_sed = Dict(f => cib_sed[f] * log(_PLK_FEFF_DUST[f] / 150)
                                for f in (100, 143, 217))
            cib_temperature_sed = Dict(f => cib_sed[f] * (
                _act_planck_logderivative_temperature(_PLK_FEFF_DUST[f], x[8]) -
                _act_planck_logderivative_temperature(150.0, x[8]))
                for f in (100, 143, 217))
            radio_sed = Dict(f => _fg_radio_ratio(_PLK_FEFF_RADIO[f], x[6])
                             for f in (100, 143, 217))
            radio_beta_sed = Dict(f => radio_sed[f] * log(_PLK_FEFF_RADIO[f] / 150)
                                  for f in (100, 143, 217))
            dusty_sed = Dict(f => _fg_cib_ratio(_PLK_FEFF_DUST[f], x[7], x[8])
                             for f in (100, 143, 217))
            dusty_beta_sed = Dict(f => dusty_sed[f] * log(_PLK_FEFF_DUST[f] / 150)
                                  for f in (100, 143, 217))
            dusty_temperature_sed = Dict(f => dusty_sed[f] * (
                _act_planck_logderivative_temperature(_PLK_FEFF_DUST[f], x[8]) -
                _act_planck_logderivative_temperature(150.0, x[8]))
                for f in (100, 143, 217))
            dl_tsz = _fg_tsz_template(templates, lmax, x[9])
            dl_tsz_alpha = zeros(eltype(x), length(dl_tsz))
            for l in 2:lmax
                dl_tsz_alpha[l + 1] = dl_tsz[l + 1] * log(l / 3000)
            end
            dl_ksz = _fg_read_template(templates, :ksz, lmax, 3000)
            dl_cib = _fg_read_template(templates, :cib, lmax, 3000)
            dl_szxcib = _fg_read_template(templates, :szxcib, lmax, 3000)
            dl_ps = _fg_powerlaw(zero(x[1]), lmax, 3000)
            root_ac = sqrt(x[1] * x[2])
            xirt = -x[4] * root_ac
        end

        for (row, (m1, m2)) in enumerate(_PLK_XSPECS)
            f1, f2 = PLK_FREQS[m1], PLK_FREQS[m2]
            frequency = max(f1, f2)
            bar = @view foreground_bar[row, :]
            ad_index = 9 + _plk_dust_index(mode, frequency)
            alpha_index = _planck_alpha_index(mode, frequency)
            d1, d2 = dust_sed[f1], dust_sed[f2]
            db1, db2 = dust_beta_sed[f1], dust_beta_sed[f2]
            dust_pair = d1 * d2
            xbar[ad_index] += dot(bar, dust_shapes[frequency]) * dust_pair
            xbar[alpha_index] += x[ad_index] * dot(bar, dust_alpha_shapes[frequency]) * dust_pair
            xbar[beta_index] += x[ad_index] * dot(bar, dust_shapes[frequency]) *
                                (db1 * d2 + d1 * db2)
            mode === :TT || continue

            t1, t2 = tsz_sed[f1], tsz_sed[f2]
            c1, c2 = cib_sed[f1], cib_sed[f2]
            cb1, cb2 = cib_beta_sed[f1], cib_beta_sed[f2]
            ct1, ct2 = cib_temperature_sed[f1], cib_temperature_sed[f2]
            xbar[1] += dot(bar, dl_tsz) * t1 * t2
            xbar[9] += x[1] * dot(bar, dl_tsz_alpha) * t1 * t2
            xbar[3] += dot(bar, dl_ksz)
            xbar[2] += dot(bar, dl_cib) * c1 * c2
            xbar[5] += x[2] * dot(bar, dl_cib) * (cb1 * c2 + c1 * cb2)
            xbar[8] += x[2] * dot(bar, dl_cib) * (ct1 * c2 + c1 * ct2)

            cross_contract = dot(bar, dl_szxcib)
            cross_sed = t1 * c2 + t2 * c1
            xbar[4] -= root_ac * cross_sed * cross_contract
            xbar[1] += (-x[4] * x[2] / (2root_ac)) * cross_sed * cross_contract
            xbar[2] += (-x[4] * x[1] / (2root_ac)) * cross_sed * cross_contract
            xbar[5] += xirt * (t1 * cb2 + t2 * cb1) * cross_contract
            xbar[8] += xirt * (t1 * ct2 + t2 * ct1) * cross_contract

            r1, r2 = radio_sed[f1], radio_sed[f2]
            rb1, rb2 = radio_beta_sed[f1], radio_beta_sed[f2]
            ps_contract = dot(bar, dl_ps)
            xbar[32] += ps_contract * r1 * r2
            xbar[6] += x[32] * ps_contract * (rb1 * r2 + r1 * rb2)
            p1, p2 = dusty_sed[f1], dusty_sed[f2]
            pb1, pb2 = dusty_beta_sed[f1], dusty_beta_sed[f2]
            pt1, pt2 = dusty_temperature_sed[f1], dusty_temperature_sed[f2]
            xbar[33] += ps_contract * p1 * p2
            xbar[7] += x[33] * ps_contract * (pb1 * p2 + p1 * pb2)
            xbar[8] += x[33] * ps_contract * (pt1 * p2 + p1 * pt2)
        end
        return NoTangent(), NoTangent(), NoTangent(), xbar
    end
    return foreground, planck_foregrounds_flat_pullback
end

@inline _act_mode_indices(mode::Symbol) =
    mode === :TT ? (10, 13, 16) : mode === :TE ? (11, 14, 17) : (12, 15, 18)

# --- Complete ACT foreground mode. The pullback works only on the 15 released
# cross pairs and contracts them into the 26 shared/ACT foreground scalars.
function ChainRulesCore.rrule(
    ::typeof(_act_foregrounds_flat), data::ACTData, templates::SharedTemplates,
    mode::Symbol, x::AbstractVector{<:Real},
)
    foreground, cache = _act_foregrounds_flat_cached(data, templates, mode, x)
    function act_foregrounds_flat_pullback(Δ)
        foreground_bar = unthunk(Δ)
        Tbar = promote_type(eltype(foreground_bar), eltype(x))
        xbar = zeros(Tbar, length(x))
        maps = data.map_names
        lmax = size(first(values(data.beams)), 1) - 1
        ad_index, beta_index, alpha_index = _act_mode_indices(mode)
        shifts = Dict(_ACT_MAPS[i] => x[21 + i] for i in 1:5)

        dust = cache.dust
        dustbar = Dict(m => zeros(Tbar, length(dust[m])) for m in maps)
        dust_shape = cache.dust_shape
        dust_shape_bar = zeros(Tbar, length(dust_shape))

        if mode === :TT
            tsz = cache.tsz
            cib = cache.cib
            radio = cache.radio
            dusty = cache.dusty
            tszbar = Dict(m => zeros(Tbar, length(tsz[m])) for m in maps)
            cibbar = Dict(m => zeros(Tbar, length(cib[m])) for m in maps)
            radiobar = Dict(m => zeros(Tbar, length(radio[m])) for m in maps)
            dustybar = Dict(m => zeros(Tbar, length(dusty[m])) for m in maps)
            dl_tsz = cache.dl_tsz
            dl_ksz = cache.dl_ksz
            dl_cib = cache.dl_cib
            dl_szxcib = cache.dl_szxcib
            dl_ps = cache.dl_ps
            root_ac = sqrt(x[1] * x[2])
            xirt = -x[4] * root_ac
        end

        for row in eachindex(data.cross_pairs)
            m1 = maps[data.cross_pair_i1[row]]
            m2 = maps[data.cross_pair_i2[row]]
            bar = @view foreground_bar[row, :]
            d1, d2 = dust[m1], dust[m2]
            xbar[ad_index] += dot(bar, d1 .* d2 .* dust_shape)
            @views dustbar[m1] .+= bar .* (x[ad_index] .* d2 .* dust_shape)
            @views dustbar[m2] .+= bar .* (x[ad_index] .* d1 .* dust_shape)
            @views dust_shape_bar .+= bar .* (x[ad_index] .* d1 .* d2)
            mode === :TT || continue

            t1, t2 = tsz[m1], tsz[m2]
            c1, c2 = cib[m1], cib[m2]
            r1, r2 = radio[m1], radio[m2]
            p1, p2 = dusty[m1], dusty[m2]
            xbar[1] += dot(bar, t1 .* t2 .* dl_tsz)
            xbar[3] += dot(bar, dl_ksz)
            xbar[2] += dot(bar, c1 .* c2 .* dl_cib)
            xbar[19] += dot(bar, r1 .* r2 .* dl_ps)
            xbar[21] += dot(bar, p1 .* p2 .* dl_ps)
            @views tszbar[m1] .+= bar .* (x[1] .* t2 .* dl_tsz .+ xirt .* c2 .* dl_szxcib)
            @views tszbar[m2] .+= bar .* (x[1] .* t1 .* dl_tsz .+ xirt .* c1 .* dl_szxcib)
            @views cibbar[m1] .+= bar .* (x[2] .* c2 .* dl_cib .+ xirt .* t2 .* dl_szxcib)
            @views cibbar[m2] .+= bar .* (x[2] .* c1 .* dl_cib .+ xirt .* t1 .* dl_szxcib)
            @views radiobar[m1] .+= bar .* (x[19] .* r2 .* dl_ps)
            @views radiobar[m2] .+= bar .* (x[19] .* r1 .* dl_ps)
            @views dustybar[m1] .+= bar .* (x[21] .* p2 .* dl_ps)
            @views dustybar[m2] .+= bar .* (x[21] .* p1 .* dl_ps)

            cross_contract = dot(bar, (t1 .* c2 .+ c1 .* t2) .* dl_szxcib)
            xbar[4] -= root_ac * cross_contract
            xbar[1] += (-x[4] * x[2] / (2root_ac)) * cross_contract
            xbar[2] += (-x[4] * x[1] / (2root_ac)) * cross_contract
        end

        for l in 2:lmax
            xbar[alpha_index] += dust_shape_bar[l + 1] * dust_shape[l + 1] * log(l / 500)
        end
        if mode === :TT
            dl_tsz_bar = zeros(Tbar, length(dl_tsz))
            for row in eachindex(data.cross_pairs)
                m1 = maps[data.cross_pair_i1[row]]
                m2 = maps[data.cross_pair_i2[row]]
                @views dl_tsz_bar .+= foreground_bar[row, :] .* (x[1] .* tsz[m1] .* tsz[m2])
            end
            for l in 2:lmax
                xbar[9] += dl_tsz_bar[l + 1] * dl_tsz[l + 1] * log(l / 3000)
            end
        end

        for m in maps
            shift_index = 21 + findfirst(==(m), _ACT_MAPS)
            _, dust_pullback = rrule(
                _act_sed_rr, data, m, :dust, x[beta_index], zero(x[1]), shifts[m])
            dust_tangent = dust_pullback(dustbar[m])
            xbar[beta_index] += dust_tangent[5]
            xbar[shift_index] += dust_tangent[7]
            mode === :TT || continue

            _, tsz_pullback = rrule(
                _act_sed_rr, data, m, :tsz, zero(x[1]), zero(x[1]), shifts[m])
            _, cib_pullback = rrule(_act_sed_rr, data, m, :cib, x[5], x[8], shifts[m])
            _, radio_pullback = rrule(
                _act_sed_rr, data, m, :radio, x[6], zero(x[1]), shifts[m])
            _, dusty_pullback = rrule(_act_sed_rr, data, m, :dusty, x[7], x[8], shifts[m])
            tsz_tangent = tsz_pullback(tszbar[m])
            cib_tangent = cib_pullback(cibbar[m])
            radio_tangent = radio_pullback(radiobar[m])
            dusty_tangent = dusty_pullback(dustybar[m])
            xbar[5] += cib_tangent[5]
            xbar[6] += radio_tangent[5]
            xbar[7] += dusty_tangent[5]
            xbar[8] += cib_tangent[6] + dusty_tangent[6]
            xbar[shift_index] += tsz_tangent[7] + cib_tangent[7] +
                                 radio_tangent[7] + dusty_tangent[7]
        end
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), xbar
    end
    return foreground, act_foregrounds_flat_pullback
end

# --- Complete ACT residual boundary. Reverse the released windows directly
# into the three theory vectors and three foreground matrices; ET intentionally
# accumulates into the shared TE inputs.
function ChainRulesCore.rrule(
    ::typeof(_act_residual_flat),
    data::ACTData,
    dlth_tt::AbstractVector{<:Real},
    dlth_te::AbstractVector{<:Real},
    dlth_ee::AbstractVector{<:Real},
    dlfg_tt::AbstractMatrix{<:Real},
    dlfg_te::AbstractMatrix{<:Real},
    dlfg_ee::AbstractMatrix{<:Real},
    cal::AbstractVector{<:Real},
)
    residual = _act_residual_flat(
        data, dlth_tt, dlth_te, dlth_ee, dlfg_tt, dlfg_te, dlfg_ee, cal)
    function act_residual_flat_pullback(Δ)
        delta_bar = unthunk(Δ)
        T = promote_type(
            eltype(dlth_tt), eltype(dlth_te), eltype(dlth_ee),
            eltype(dlfg_tt), eltype(dlfg_te), eltype(dlfg_ee),
            eltype(cal), eltype(delta_bar),
        )
        dlth_tt_bar = zeros(T, size(dlth_tt))
        dlth_te_bar = zeros(T, size(dlth_te))
        dlth_ee_bar = zeros(T, size(dlth_ee))
        dlfg_tt_bar = zeros(T, size(dlfg_tt))
        dlfg_te_bar = zeros(T, size(dlfg_te))
        dlfg_ee_bar = zeros(T, size(dlfg_ee))
        calbar = zeros(T, size(cal))

        i0 = 1
        for (i, (W, values, pol, ispec)) in enumerate(zip(
            data.windows, data.window_values, data.spec_pol, data.spec_index))
            nb = size(W, 2)
            i1 = i0 + nb - 1
            idx = values .+ 1
            if pol == "TT"
                theory = dlth_tt
                foreground = dlfg_tt
                theorybar = dlth_tt_bar
                foregroundbar = dlfg_tt_bar
            elseif pol == "EE"
                theory = dlth_ee
                foreground = dlfg_ee
                theorybar = dlth_ee_bar
                foregroundbar = dlfg_ee_bar
            else
                theory = dlth_te
                foreground = dlfg_te
                theorybar = dlth_te_bar
                foregroundbar = dlfg_te_bar
            end

            spectrum = theory[idx] .+ foreground[ispec + 1, idx]
            projected = transpose(W) * spectrum
            projected_bar = -(@view(delta_bar[i0:i1])) ./ cal[i]
            spectrumbar = W * projected_bar
            theorybar[idx] .+= spectrumbar
            @views foregroundbar[ispec + 1, idx] .+= spectrumbar
            calbar[i] = dot(@view(delta_bar[i0:i1]), projected) / cal[i]^2
            i0 = i1 + 1
        end

        return NoTangent(), NoTangent(),
            dlth_tt_bar, dlth_te_bar, dlth_ee_bar,
            dlfg_tt_bar, dlfg_te_bar, dlfg_ee_bar, calbar
    end
    return residual, act_residual_flat_pullback
end

function _planck_select_pullback!(
    rlbar::AbstractMatrix, data::PlanckData, mode::Symbol,
    delta_bar::AbstractVector, i0::Int,
)
    for xf in 1:6
        lmin, lmax = _xfreq_cut(data, mode, xf)
        for (bmin, bmax) in zip(data.bin_lmins, data.bin_lmaxs)
            bmin >= lmin && bmax <= lmax || continue
            dl = bmax - bmin + 1
            @views rlbar[xf, bmin+1:bmax+1] .+= delta_bar[i0] / dl
            i0 += 1
        end
    end
    return i0
end

function _planck_weight_den(data::PlanckData, weight::AbstractMatrix, ::Type{T}) where {T}
    den = zeros(T, 6, size(weight, 2))
    for xs in 1:15
        xf = data.xspec2xfreq[xs] + 1
        @views den[xf, :] .+= weight[xs, :]
    end
    return den
end

function _planck_average_pullback(
    data::PlanckData, weight::AbstractMatrix,
    rlbar::AbstractMatrix, den::AbstractMatrix,
)
    T = promote_type(eltype(rlbar), eltype(weight), eltype(den))
    rspecbar = zeros(T, 15, size(weight, 2))
    for xs in 1:15
        xf = data.xspec2xfreq[xs] + 1
        for l in axes(weight, 2)
            d = den[xf, l]
            iszero(d) && continue
            rspecbar[xs, l] = weight[xs, l] * rlbar[xf, l] / d
        end
    end
    return rspecbar
end

function _planck_model_cal_pullback(
    model::AbstractMatrix, cal::AbstractVector, rspecbar::AbstractMatrix,
)
    T = promote_type(eltype(model), eltype(cal), eltype(rspecbar))
    modelbar = Matrix{T}(undef, size(model))
    calbar = zeros(T, size(cal))
    for xs in axes(model, 1)
        c = cal[xs]
        @views modelbar[xs, :] .= -rspecbar[xs, :] ./ c
        calbar[xs] = dot(@view(rspecbar[xs, :]), @view(model[xs, :])) / c^2
    end
    return modelbar, calbar
end

# --- Complete Planck residual boundary. This is the exact transpose of the
# per-cross-spectrum calibration, fixed weighted cross-frequency averaging,
# TE+ET combination, and released lite-bin selection.
function ChainRulesCore.rrule(
    ::typeof(_planck_residual_flat),
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
    residual = _planck_residual_flat(
        data, model_tt, model_ee, model_te, model_et,
        cal_tt, cal_ee, cal_te, cal_et)
    function planck_residual_flat_pullback(Δ)
        delta_bar = unthunk(Δ)
        T = promote_type(
            eltype(model_tt), eltype(model_ee), eltype(model_te), eltype(model_et),
            eltype(cal_tt), eltype(cal_ee), eltype(cal_te), eltype(cal_et),
            eltype(delta_bar),
        )
        rlbar_tt = zeros(T, 6, 2501)
        rlbar_ee = zeros(T, 6, 2501)
        rlbar_te = zeros(T, 6, 2501)
        i0 = _planck_select_pullback!(rlbar_tt, data, :TT, delta_bar, 1)
        i0 = _planck_select_pullback!(rlbar_ee, data, :EE, delta_bar, i0)
        i0 = _planck_select_pullback!(rlbar_te, data, :TE, delta_bar, i0)
        i0 == length(delta_bar) + 1 || error("Planck residual cotangent length mismatch")

        weight_tt = data.dlweight[:TT]
        weight_ee = data.dlweight[:EE]
        weight_te = data.dlweight[:TE]
        weight_et = data.dlweight[:ET]
        den_tt = _planck_weight_den(data, weight_tt, T)
        den_ee = _planck_weight_den(data, weight_ee, T)
        den_te = _planck_weight_den(data, weight_te, T)
        den_et = _planck_weight_den(data, weight_et, T)

        rspecbar_tt = _planck_average_pullback(data, weight_tt, rlbar_tt, den_tt)
        rspecbar_ee = _planck_average_pullback(data, weight_ee, rlbar_ee, den_ee)

        # TE and ET share the combined numerator and fixed total denominator.
        numbar = zeros(T, size(rlbar_te))
        for i in eachindex(numbar)
            den = den_te[i] + den_et[i]
            iszero(den) || (numbar[i] = rlbar_te[i] / den)
        end
        rspecbar_te = zeros(T, size(model_te))
        rspecbar_et = zeros(T, size(model_et))
        for xs in 1:15
            xf = data.xspec2xfreq[xs] + 1
            @views rspecbar_te[xs, :] .= weight_te[xs, :] .* numbar[xf, :]
            @views rspecbar_et[xs, :] .= weight_et[xs, :] .* numbar[xf, :]
        end

        model_tt_bar, cal_tt_bar = _planck_model_cal_pullback(model_tt, cal_tt, rspecbar_tt)
        model_ee_bar, cal_ee_bar = _planck_model_cal_pullback(model_ee, cal_ee, rspecbar_ee)
        model_te_bar, cal_te_bar = _planck_model_cal_pullback(model_te, cal_te, rspecbar_te)
        model_et_bar, cal_et_bar = _planck_model_cal_pullback(model_et, cal_et, rspecbar_et)

        return NoTangent(), NoTangent(),
            model_tt_bar, model_ee_bar, model_te_bar, model_et_bar,
            cal_tt_bar, cal_ee_bar, cal_te_bar, cal_et_bar
    end
    return residual, planck_residual_flat_pullback
end

# --- Complete SPT instrument boundary. The generic reverse pass through this
# graph records all 21 multipole-space spectra and their beam/leakage
# intermediates. This rule instead performs the exact transpose sequence:
# windows -> model/calibration -> leakage -> polarized beam -> beam modes.
function ChainRulesCore.rrule(::typeof(_spt_residual_flat), data::SPTData,
                              sky::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    residual = _spt_residual_flat(data, sky, x)
    function spt_residual_flat_pullback(Δ)
        delta_bar = unthunk(Δ)
        n_ell = length(data.ells)
        T = promote_type(eltype(sky), eltype(x), eltype(delta_bar))
        skybar = zeros(T, length(sky))
        xbar = zeros(T, length(x))

        block(v, b) = @view v[(b - 1) * n_ell + 1:b * n_ell]
        cal = x[_SPT_CAL]
        calT = @view x[_SPT_CAL_FREQ]
        calE = @view x[_SPT_PE_FREQ]
        beta_pol = @view x[_SPT_BETA_POL]
        temperature_beam = _spt_temperature_beam(data, @view(x[_SPT_BETA_MODES]))
        polarization_beam = _spt_polarization_beam(data, temperature_beam, beta_pol)
        t2p_kernel = _spt_t2p_kernels(data, @view(x[_SPT_T2P]))

        temperature_beambar = zeros(T, size(temperature_beam))
        polarization_beambar = zeros(T, size(polarization_beam))
        t2p_kernelbar = zeros(T, 3, n_ell)

        i0 = 1
        for (b, (mode, f1, f2)) in enumerate(_SPT_SPECS)
            nb = size(data.windows[b], 2)
            i1 = i0 + nb - 1
            modelbar = -(data.windows[b] * @view(delta_bar[i0:i1]))
            s = block(sky, b)
            lo, hi = minmax(f1, f2)

            if mode == "TT"
                u = s
                beam = @view(temperature_beam[f1, :]) .* @view(temperature_beam[f2, :])
                calint = calT[f1] * calT[f2]
            elseif mode == "TE"
                tt_block = _spt_tt_block(f1, f2)
                tt = block(sky, tt_block)
                u = s .+ t2p_kernel[f2] .* tt
                beam = @view(temperature_beam[f1, :]) .* @view(polarization_beam[f2, :])
                calint = calT[f1] * calT[f2] * calE[f2]
            else
                te12_block = _spt_te_block(lo, hi)
                te21_block = _spt_te_block(hi, lo)
                tt_block = _spt_tt_block(f1, f2)
                te12 = block(sky, te12_block)
                te21 = block(sky, te21_block)
                tt = block(sky, tt_block)
                k1 = t2p_kernel[f1]
                k2 = t2p_kernel[f2]
                u = s .+ k1 .* te12 .+ k2 .* te21 .+ (k1 .* k2) .* tt
                beam = @view(polarization_beam[f1, :]) .* @view(polarization_beam[f2, :])
                calint = calT[f1] * calE[f1] * calT[f2] * calE[f2]
            end

            den = cal^2 * calint
            model = u .* beam ./ den
            q = dot(modelbar, model)
            ubar = modelbar .* beam ./ den
            beambar = modelbar .* u ./ den

            block(skybar, b) .+= ubar
            xbar[_SPT_CAL] -= 2q / cal

            if mode == "TT"
                xbar[first(_SPT_CAL_FREQ) + f1 - 1] -= q / calT[f1]
                xbar[first(_SPT_CAL_FREQ) + f2 - 1] -= q / calT[f2]
                @views temperature_beambar[f1, :] .+= beambar .* temperature_beam[f2, :]
                @views temperature_beambar[f2, :] .+= beambar .* temperature_beam[f1, :]
            elseif mode == "TE"
                xbar[first(_SPT_CAL_FREQ) + f1 - 1] -= q / calT[f1]
                xbar[first(_SPT_CAL_FREQ) + f2 - 1] -= q / calT[f2]
                xbar[first(_SPT_PE_FREQ) + f2 - 1] -= q / calE[f2]
                block(skybar, tt_block) .+= ubar .* t2p_kernel[f2]
                @views t2p_kernelbar[f2, :] .+= ubar .* tt
                @views temperature_beambar[f1, :] .+= beambar .* polarization_beam[f2, :]
                @views polarization_beambar[f2, :] .+= beambar .* temperature_beam[f1, :]
            else
                xbar[first(_SPT_CAL_FREQ) + f1 - 1] -= q / calT[f1]
                xbar[first(_SPT_PE_FREQ) + f1 - 1] -= q / calE[f1]
                xbar[first(_SPT_CAL_FREQ) + f2 - 1] -= q / calT[f2]
                xbar[first(_SPT_PE_FREQ) + f2 - 1] -= q / calE[f2]
                block(skybar, te12_block) .+= ubar .* k1
                block(skybar, te21_block) .+= ubar .* k2
                block(skybar, tt_block) .+= ubar .* k1 .* k2
                @views t2p_kernelbar[f1, :] .+= ubar .* (te12 .+ k2 .* tt)
                @views t2p_kernelbar[f2, :] .+= ubar .* (te21 .+ k1 .* tt)
                @views polarization_beambar[f1, :] .+= beambar .* polarization_beam[f2, :]
                @views polarization_beambar[f2, :] .+= beambar .* polarization_beam[f1, :]
            end
            i0 = i1 + 1
        end

        # Polarized beam: p = (main + beta * (temperature - main)) / norm,
        # norm fixed at absolute ell=800 apart from beta.
        i800 = 800 - data.ells[1] + 1
        for f in 1:3
            main = @view data.main_temperature[f, :]
            tbeam = @view temperature_beam[f, :]
            pbar = @view polarization_beambar[f, :]
            beta = beta_pol[f]
            norm = main[i800] + beta * (1 - main[i800])
            num = main .+ beta .* (tbeam .- main)
            @views temperature_beambar[f, :] .+= pbar .* beta ./ norm
            normbar = -dot(pbar, num) / norm^2
            xbar[first(_SPT_BETA_POL) + f - 1] +=
                dot(pbar, tbeam .- main) / norm + normbar * (1 - main[i800])
        end

        # Temperature beam modes and T-to-P amplitudes are linear.
        for m in 1:length(_SPT_BETA_MODES)
            xbar[first(_SPT_BETA_MODES) + m - 1] =
                sum(temperature_beambar .* @view(data.eigenmodes[:, :, m]))
        end
        ell2 = Float64.(data.ells) .^ 2
        for f in 1:3
            xbar[first(_SPT_T2P) + f - 1] =
                dot(@view(t2p_kernelbar[f, :]), _SPT_T2P_SIGMAS[f]^2 .* ell2)
        end

        return NoTangent(), NoTangent(), skybar, xbar
    end
    return residual, spt_residual_flat_pullback
end

# --- np.gradient on a unit-spaced grid. The analytical transpose is a
# tridiagonal stencil, avoiding one reverse-mode tape entry per multipole.
function ChainRulesCore.rrule(::typeof(_np_gradient), y::AbstractVector{<:Real})
    g = _np_gradient(y)
    function np_gradient_pullback(Δ)
        gbar = unthunk(Δ)
        ybar = zeros(promote_type(eltype(y), eltype(gbar)), length(y))
        ybar[1] -= gbar[1]
        ybar[2] += gbar[1]
        ybar[end] += gbar[end]
        ybar[end-1] -= gbar[end]
        for i in 2:length(y)-1
            ybar[i-1] -= gbar[i] / 2
            ybar[i+1] += gbar[i] / 2
        end
        return NoTangent(), ybar
    end
    return g, np_gradient_pullback
end

function ChainRulesCore.rrule(::typeof(_spt_temperature_beam), data::SPTData,
                              beta_modes::AbstractVector{<:Real})
    beam = _spt_temperature_beam(data, beta_modes)
    function spt_temperature_beam_pullback(Δ)
        beambar = unthunk(Δ)
        betabar = [sum(beambar .* @view(data.eigenmodes[:, :, m])) for m in eachindex(beta_modes)]
        return NoTangent(), NoTangent(), betabar
    end
    return beam, spt_temperature_beam_pullback
end

function ChainRulesCore.rrule(::typeof(_spt_polarization_beam), data::SPTData,
                              temperature_beam::AbstractMatrix{<:Real}, beta_pol::AbstractVector{<:Real})
    beam = _spt_polarization_beam(data, temperature_beam, beta_pol)
    i800 = 800 - data.ells[1] + 1
    function spt_polarization_beam_pullback(Δ)
        pbar = unthunk(Δ)
        tbar = zeros(promote_type(eltype(pbar), eltype(temperature_beam)), size(temperature_beam))
        βbar = zeros(promote_type(eltype(pbar), eltype(beta_pol)), length(beta_pol))
        for f in 1:3
            main = @view data.main_temperature[f, :]
            β = beta_pol[f]
            norm = main[i800] + β * (1 - main[i800])
            num = main .+ β .* (@view(temperature_beam[f, :]) .- main)
            rowbar = @view pbar[f, :]
            @views tbar[f, :] .= rowbar .* β ./ norm
            nbar = -sum(rowbar .* num) / norm^2
            βbar[f] = sum(rowbar .* (@view(temperature_beam[f, :]) .- main)) / norm + nbar * (1 - main[i800])
        end
        return NoTangent(), NoTangent(), tbar, βbar
    end
    return beam, spt_polarization_beam_pullback
end

function ChainRulesCore.rrule(::typeof(_spt_t2p_kernels), data::SPTData, t2p::AbstractVector{<:Real})
    kernels = _spt_t2p_kernels(data, t2p)
    ell2 = Float64.(data.ells) .^ 2
    function spt_t2p_pullback(Δ)
        bars = unthunk(Δ)
        tbar = [dot(bars[f], _SPT_T2P_SIGMAS[f]^2 .* ell2) for f in 1:3]
        return NoTangent(), NoTangent(), tbar
    end
    return kernels, spt_t2p_pullback
end


end
