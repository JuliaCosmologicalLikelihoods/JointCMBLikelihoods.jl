# Instrument (calibration) parameter layer.
# Frozen Hillik semantics (d0e455cf):
#   Planck (_calibration, hillipop.py): per cross-spectrum product
#     TT: cal(m1)·cal(m2)              EE: cal·pe (both maps)
#     TE: cal(m1)·cal(m2)·pe(m2)       ET: cal(m1)·pe(m1)·cal(m2)
#     all × A_planck²
#   ACT (_calibration, actpol_full_dr6.py): cal1·cal2 with pe on E-maps,
#     × ACT_cal²;  pol is the pair (P1,P2) of the spectrum's two polarisations.
#   SPT: SPTInstrumentParameters (spt_model.jl), 22 sampled params.

"""
    PlanckInstrumentParameters{T}

12 sampled Planck instrument parameters: absolute calibration `A_planck`,
five detector-set intercalibrations `cal` (143A fixed to 1, not sampled),
and six polarisation efficiencies `pe`. Map order: 100A, 100B, 143A, 143B,
217A, 217B.
"""
struct PlanckInstrumentParameters{T}
    A_planck::T
    cal::NTuple{6,T}
    pe::NTuple{6,T}
end

n_sampled(::PlanckInstrumentParameters) = 12

"""
    ACTInstrumentParameters{T}

14 sampled ACT instrument parameters: absolute calibration `cal`,
four intercalibrations (pa5_f150 fixed to 1, not sampled), four polarisation
efficiencies (pa4_f220 fixed to 1, not sampled), and five bandpass shifts
(consumed by the foreground chromaticity model, carried here so the
instrument layer is the single home of ACT systematics).
"""
struct ACTInstrumentParameters{T}
    cal::T
    cal_map::NTuple{5,T}   # dr6_pa4_f220, dr6_pa5_f090, dr6_pa5_f150, dr6_pa6_f090, dr6_pa6_f150
    pe_map::NTuple{5,T}    # same order
    band_shift::NTuple{5,T}
end

n_sampled(::ACTInstrumentParameters) = 14

const _ACT_MAPS = ("dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150",
    "dr6_pa6_f090", "dr6_pa6_f150")

n_sampled(::SPTInstrumentParameters) = 22

"""
    planck_calibration(plk::PlanckInstrumentParameters, mode::Symbol) -> Vector

15-element calibration product per cross-spectrum (combinations order
100A×100B, 100A×143A, …, 217A×217B), matching the frozen `_calibration`:
TT uses cal·cal, EE cal·pe on both maps, TE cal(m1)·cal·pe(m2),
ET cal·pe(m1)·cal(m2); all × A_planck².
"""
function planck_calibration(plk::PlanckInstrumentParameters, mode::Symbol)
    c = zeros(typeof(plk.A_planck), 15)
    for (i, (m1, m2)) in enumerate(_PLK_XSPECS)
        cal1 = plk.cal[m1]
        cal2 = plk.cal[m2]
        pe1 = plk.pe[m1]
        pe2 = plk.pe[m2]
        if mode === :TT
            c[i] = cal1 * cal2
        elseif mode === :EE
            c[i] = cal1 * pe1 * cal2 * pe2
        elseif mode === :TE
            c[i] = cal1 * cal2 * pe2
        elseif mode === :ET
            c[i] = cal1 * pe1 * cal2
        else
            throw(ArgumentError("unknown mode $mode"))
        end
    end
    c .*= plk.A_planck^2
    return c
end

"""
    act_calibration(act::ACTInstrumentParameters, data::ACTData) -> Vector

Per selected (spec, pol) entry of the ACT data vector: `cal1·cal2` with the
polarisation efficiency applied to the E leg of the pair, × `ACT_cal²`.
The pair is read from `data.spec_pol` ("TT"/"TE"/"ET"/"EE": first/second
letter = polarisation of experiment 1/2).
"""
function act_calibration(act::ACTInstrumentParameters, data::ACTData)
    cal = zeros(typeof(act.cal), length(data.spec_pol))
    for i in eachindex(cal)
        i1 = data.spec_e1_idx[i]
        i2 = data.spec_e2_idx[i]
        ct1 = act.cal_map[i1]
        ct2 = act.cal_map[i2]
        if data.spec_p1_E[i]
            ct1 *= act.pe_map[i1]
        end
        if data.spec_p2_E[i]
            ct2 *= act.pe_map[i2]
        end
        cal[i] = act.cal * act.cal * ct1 * ct2
    end
    return cal
end
