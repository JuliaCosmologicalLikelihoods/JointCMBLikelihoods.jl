# Prior layer — arXiv:2511.04733 Tables A.1 (foregrounds) and A.2 (nuisance).
#
# Conventions (plan §4.4):
#   - the data likelihood never includes priors implicitly;
#   - bounded priors return -Inf outside support;
#   - Gaussian normalisation constants are omitted;
#   - fixed quantities do not occupy slots in the sampled vector.
#
# The paper's joint model samples SHARED dust spectral indices and TT tilt:
#   beta_dust_tt ~ N(1.51, 0.01), beta_dust_te ~ N(1.59, 0.02),
#   beta_dust_ee ~ N(1.59, 0.02), alpha_dust_tt ~ U[-3, -2],
# shared by Planck/ACT/SPT (the frozen per-experiment yamls fix them; the
# paper Tables A.1/A.2 are authoritative for the joint model).

export JointParameters, logprior, foreground_parameters, instrument_parameters

"""
    JointParameters{T}

The 79 sampled nuisance parameters of the joint model, grouped as in the
paper: 31 foreground (10 shared + 11 Planck + 5 ACT + 5 SPT) and 48 instrument
(12 Planck + 14 ACT + 22 SPT). Fixed quantities (PLK_cal_143A, ACT pa5_f150
cal, SPT3G 150 GHz cal, ACT pe pa4_f220, alpha_dust_te/ee = -2.4, T_dust,
T_cib, alpha_tsz, beta_dusty = beta_cib, aberration) are not stored.
"""
struct JointParameters{T}
    # --- shared foreground (10) ---
    Acib::T
    Atsz::T
    Aksz::T
    xi::T
    beta_cib::T
    beta_radio::T
    beta_dust_tt::T
    beta_dust_te::T
    beta_dust_ee::T
    alpha_dust_tt::T
    # --- Planck foreground (11) ---
    PLK_Adust100TT::T
    PLK_Adust143TT::T
    PLK_Adust217TT::T
    PLK_Adust100TE::T
    PLK_Adust143TE::T
    PLK_Adust217TE::T
    PLK_Adust100EE::T
    PLK_Adust143EE::T
    PLK_Adust217EE::T
    PLK_radio_TT::T
    PLK_cib_ps::T
    # --- ACT foreground (5) ---
    ACT_AdustTT::T
    ACT_AdustTE::T
    ACT_AdustEE::T
    ACT_radio_TT::T
    ACT_cib_ps::T
    # --- SPT foreground (5) ---
    SPT_AdustTT::T
    SPT_AdustTE::T
    SPT_AdustEE::T
    SPT_radio_TT::T
    SPT_cib_ps::T
    # --- Planck instrument (12) ---
    A_planck::T
    PLK_cal_100A::T
    PLK_cal_100B::T
    PLK_cal_143B::T
    PLK_cal_217A::T
    PLK_cal_217B::T
    PLK_pe_100A::T
    PLK_pe_100B::T
    PLK_pe_143A::T
    PLK_pe_143B::T
    PLK_pe_217A::T
    PLK_pe_217B::T
    # --- ACT instrument (14) ---
    ACT_cal::T
    ACT_cal_pa4_f220::T
    ACT_cal_pa5_f090::T
    ACT_cal_pa6_f090::T
    ACT_cal_pa6_f150::T
    ACT_pe_pa5_f090::T
    ACT_pe_pa5_f150::T
    ACT_pe_pa6_f090::T
    ACT_pe_pa6_f150::T
    ACT_band_pa4_f220::T
    ACT_band_pa5_f090::T
    ACT_band_pa5_f150::T
    ACT_band_pa6_f090::T
    ACT_band_pa6_f150::T
    # --- SPT instrument (22) ---
    SPT_cal::T
    SPT_cal_90::T
    SPT_cal_220::T
    SPT_pe_90::T
    SPT_pe_150::T
    SPT_pe_220::T
    SPT_beta_1::T
    SPT_beta_2::T
    SPT_beta_3::T
    SPT_beta_4::T
    SPT_beta_5::T
    SPT_beta_6::T
    SPT_beta_7::T
    SPT_beta_8::T
    SPT_beta_9::T
    SPT_beta_pol_90::T
    SPT_beta_pol_150::T
    SPT_beta_pol_220::T
    SPT_T2P2_90::T
    SPT_T2P2_150::T
    SPT_T2P2_220::T
    SPT_kappa::T
end

n_sampled(::JointParameters) = 79

# ---------------------------------------------------------------------------
# logprior — Tables A.1 / A.2. Gaussian constants omitted.
# ---------------------------------------------------------------------------

_log_gauss(x, mu, sigma) = -0.5 * ((x - mu) / sigma)^2

function logprior(p::JointParameters)
    lp = 0.0
    any(!isfinite, (p.Acib, p.Atsz, p.Aksz, p.xi, p.beta_cib, p.beta_radio,
        p.beta_dust_tt, p.beta_dust_te, p.beta_dust_ee, p.alpha_dust_tt)) && return -Inf

    # --- shared foreground (Table A.1) ---
    0 ≤ p.Acib ≤ 50 || return -Inf
    0 ≤ p.Atsz ≤ 50 || return -Inf
    0 ≤ p.Aksz ≤ 50 || return -Inf
    -1 ≤ p.xi ≤ 1 || return -Inf
    1 ≤ p.beta_cib ≤ 3 || return -Inf
    -1.5 ≤ p.beta_radio ≤ 0 || return -Inf
    lp += _log_gauss(p.beta_dust_tt, 1.51, 0.01)
    lp += _log_gauss(p.beta_dust_te, 1.59, 0.02)
    lp += _log_gauss(p.beta_dust_ee, 1.59, 0.02)
    -3 ≤ p.alpha_dust_tt ≤ -2 || return -Inf

    # --- Planck foreground (Table A.1) ---
    all(0 .≤ (p.PLK_Adust100TT, p.PLK_Adust143TT, p.PLK_Adust217TT) .≤ 100) || return -Inf
    all(0 .≤ (p.PLK_Adust100TE, p.PLK_Adust143TE, p.PLK_Adust217TE,
              p.PLK_Adust100EE, p.PLK_Adust143EE, p.PLK_Adust217EE) .≤ 10) || return -Inf
    0 ≤ p.PLK_radio_TT ≤ 150 || return -Inf
    0 ≤ p.PLK_cib_ps ≤ 100 || return -Inf

    # --- ACT foreground ---
    lp += _log_gauss(p.ACT_AdustTT, 7.95, 0.32)
    lp += _log_gauss(p.ACT_AdustTE, 0.423, 0.030)
    lp += _log_gauss(p.ACT_AdustEE, 0.168, 0.017)
    0 ≤ p.ACT_radio_TT ≤ 20 || return -Inf
    0 ≤ p.ACT_cib_ps ≤ 20 || return -Inf

    # --- SPT foreground ---
    lp += _log_gauss(p.SPT_AdustTT, 1.88, 0.960)
    lp += _log_gauss(p.SPT_AdustTE, 0.12, 0.051)
    lp += _log_gauss(p.SPT_AdustEE, 0.05, 0.022)
    0 ≤ p.SPT_radio_TT ≤ 10 || return -Inf
    0 ≤ p.SPT_cib_ps ≤ 10 || return -Inf

    # --- Planck instrument (Table A.2) ---
    lp += _log_gauss(p.A_planck, 1.00, 0.0025)
    for c in (p.PLK_cal_100A, p.PLK_cal_100B, p.PLK_cal_143B, p.PLK_cal_217A, p.PLK_cal_217B)
        lp += _log_gauss(c, 1.00, 0.01)
    end
    all(0.8 .≤ (p.PLK_pe_100A, p.PLK_pe_100B, p.PLK_pe_143A, p.PLK_pe_143B,
                p.PLK_pe_217A, p.PLK_pe_217B) .≤ 1.2) || return -Inf

    # --- ACT instrument ---
    0.7 ≤ p.ACT_cal ≤ 1.3 || return -Inf
    lp += _log_gauss(p.ACT_cal_pa4_f220, 1.00, 0.013)
    lp += _log_gauss(p.ACT_cal_pa5_f090, 1.00, 0.0016)
    lp += _log_gauss(p.ACT_cal_pa6_f090, 1.00, 0.0018)
    lp += _log_gauss(p.ACT_cal_pa6_f150, 1.00, 0.0024)
    all(0.8 .≤ (p.ACT_pe_pa5_f090, p.ACT_pe_pa5_f150, p.ACT_pe_pa6_f090,
                p.ACT_pe_pa6_f150) .≤ 1.2) || return -Inf
    lp += _log_gauss(p.ACT_band_pa4_f220, 0.0, 3.6)
    lp += _log_gauss(p.ACT_band_pa5_f090, 0.0, 1.0)
    lp += _log_gauss(p.ACT_band_pa5_f150, 0.0, 1.3)
    lp += _log_gauss(p.ACT_band_pa6_f090, 0.0, 1.2)
    lp += _log_gauss(p.ACT_band_pa6_f150, 0.0, 1.1)

    # --- SPT instrument ---
    0.7 ≤ p.SPT_cal ≤ 1.3 || return -Inf
    lp += _log_gauss(p.SPT_cal_90, 1.00, 0.0003)
    lp += _log_gauss(p.SPT_cal_220, 1.00, 0.001)
    all(0.8 .≤ (p.SPT_pe_90, p.SPT_pe_150, p.SPT_pe_220) .≤ 1.2) || return -Inf
    for b in (p.SPT_beta_1, p.SPT_beta_2, p.SPT_beta_3, p.SPT_beta_4, p.SPT_beta_5,
              p.SPT_beta_6, p.SPT_beta_7, p.SPT_beta_8, p.SPT_beta_9)
        lp += _log_gauss(b, 0.0, 1.0)
    end
    all(0.0 .≤ (p.SPT_beta_pol_90, p.SPT_beta_pol_150, p.SPT_beta_pol_220) .≤ 1.0) || return -Inf
    lp += _log_gauss(p.SPT_T2P2_90, -0.0065, 0.0011)
    lp += _log_gauss(p.SPT_T2P2_150, -0.012, 0.0021)
    lp += _log_gauss(p.SPT_T2P2_220, -0.023, 0.0066)
    lp += _log_gauss(p.SPT_kappa, 0.0, 0.00045)

    return lp
end

# ---------------------------------------------------------------------------
# Expansion of the shared dust parameters into the per-experiment structs
# ---------------------------------------------------------------------------

const _FIXED_ALPHA_DUST_TE = -2.4
const _FIXED_ALPHA_DUST_EE = -2.4
const _FIXED_T_CIB = 25.0
const _FIXED_ALPHA_TSZ = 0.0

"""
    foreground_parameters(p::JointParameters)

Expand the 79-parameter vector into the per-experiment foreground structs.
Shared dust spectral indices `beta_dust_tt/te/ee` and tilt `alpha_dust_tt`
propagate to all three experiments (paper joint model); ET reuses TE.
"""
function foreground_parameters(p::JointParameters)
    T = promote_type(typeof(p.Atsz), typeof(p.Acib))
    t_cib = convert(T, _FIXED_T_CIB)
    alpha_tsz = convert(T, _FIXED_ALPHA_TSZ)
    alpha_te = convert(T, _FIXED_ALPHA_DUST_TE)
    alpha_ee = convert(T, _FIXED_ALPHA_DUST_EE)
    zero_t = zero(T)
    shared = SharedForegroundParameters(
        p.Atsz, p.Acib, p.Aksz, p.xi, p.beta_cib, p.beta_radio,
        p.beta_cib, t_cib, alpha_tsz)

    plk = PlanckForegroundParameters(
        (p.PLK_Adust100TT, p.PLK_Adust143TT, p.PLK_Adust217TT,
         p.PLK_Adust100TE, p.PLK_Adust143TE, p.PLK_Adust217TE,
         p.PLK_Adust100TE, p.PLK_Adust143TE, p.PLK_Adust217TE,
         p.PLK_Adust100EE, p.PLK_Adust143EE, p.PLK_Adust217EE),
        (p.beta_dust_tt, p.beta_dust_te, p.beta_dust_te, p.beta_dust_ee),
        (p.alpha_dust_tt, p.alpha_dust_tt, p.alpha_dust_tt),
        alpha_te, alpha_te, alpha_ee,
        p.PLK_radio_TT, p.PLK_cib_ps)

    act = ACTForegroundParameters(
        p.ACT_AdustTT, p.ACT_AdustTE, p.ACT_AdustEE,
        p.beta_dust_tt, p.beta_dust_te, p.beta_dust_ee,
        p.alpha_dust_tt, alpha_te, alpha_ee,
        p.ACT_radio_TT, zero_t, p.ACT_cib_ps,
        Dict("dr6_pa4_f220" => p.ACT_band_pa4_f220,
             "dr6_pa5_f090" => p.ACT_band_pa5_f090,
             "dr6_pa5_f150" => p.ACT_band_pa5_f150,
             "dr6_pa6_f090" => p.ACT_band_pa6_f090,
             "dr6_pa6_f150" => p.ACT_band_pa6_f150))

    spt = SPTForegroundParameters(
        p.SPT_AdustTT, p.SPT_AdustTE, p.SPT_AdustEE,
        p.beta_dust_tt, p.beta_dust_te, p.beta_dust_ee,
        p.alpha_dust_tt, alpha_te, alpha_ee,
        p.SPT_radio_TT, zero_t, zero_t, p.SPT_cib_ps, p.SPT_kappa)

    return (shared = shared, planck = plk, act = act, spt = spt)
end

"""
    instrument_parameters(p::JointParameters)

Expand into the per-experiment instrument structs, inserting the fixed
reference values (PLK 143A cal, ACT pa5_f150 cal and pa4_f220 pe, SPT 150 cal).
"""
function instrument_parameters(p::JointParameters)
    T = typeof(p.A_planck)
    one_t = one(T)
    plk = PlanckInstrumentParameters(
        p.A_planck,
        (p.PLK_cal_100A, p.PLK_cal_100B, one_t, p.PLK_cal_143B, p.PLK_cal_217A, p.PLK_cal_217B),
        (p.PLK_pe_100A, p.PLK_pe_100B, p.PLK_pe_143A, p.PLK_pe_143B, p.PLK_pe_217A, p.PLK_pe_217B))

    act = ACTInstrumentParameters(
        p.ACT_cal,
        (p.ACT_cal_pa4_f220, p.ACT_cal_pa5_f090, one_t, p.ACT_cal_pa6_f090, p.ACT_cal_pa6_f150),
        (one_t, p.ACT_pe_pa5_f090, p.ACT_pe_pa5_f150, p.ACT_pe_pa6_f090, p.ACT_pe_pa6_f150),
        (p.ACT_band_pa4_f220, p.ACT_band_pa5_f090, p.ACT_band_pa5_f150,
         p.ACT_band_pa6_f090, p.ACT_band_pa6_f150))

    spt = SPTInstrumentParameters(
        p.SPT_cal,
        (p.SPT_cal_90, one_t, p.SPT_cal_220),
        (p.SPT_pe_90, p.SPT_pe_150, p.SPT_pe_220),
        (p.SPT_beta_pol_90, p.SPT_beta_pol_150, p.SPT_beta_pol_220),
        (p.SPT_beta_1, p.SPT_beta_2, p.SPT_beta_3, p.SPT_beta_4, p.SPT_beta_5,
         p.SPT_beta_6, p.SPT_beta_7, p.SPT_beta_8, p.SPT_beta_9),
        (p.SPT_T2P2_90, p.SPT_T2P2_150, p.SPT_T2P2_220))

    return (planck = plk, act = act, spt = spt)
end
