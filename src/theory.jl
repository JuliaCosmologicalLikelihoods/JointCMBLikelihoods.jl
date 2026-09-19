"""
    JointCMBTheory{T}

CMB theory spectra shared by all three experiments: Dℓ in μK².
`ell` must be contiguous starting at some `ell_min` (typically 0).
"""
struct JointCMBTheory{T<:Real}
    ell::Vector{Int}
    TT::Vector{T}
    TE::Vector{T}
    EE::Vector{T}
end

function JointCMBTheory(ell::Vector{Int}, TT, TE, EE)
    n = length(ell)
    (length(TT) == length(TE) == length(EE) == n) ||
        throw(DimensionMismatch("theory arrays must share one multipole grid"))
    all(diff(ell) .== 1) || throw(ArgumentError("theory multipoles must be contiguous"))
    return JointCMBTheory(ell, Vector{Float64}(TT), Vector{Float64}(TE), Vector{Float64}(EE))
end

"""Theory Dℓ at absolute multipole ℓ (0 outside the stored grid)."""
function theory_at(theory::JointCMBTheory, mode::Symbol, ℓ::Int)
    i = ℓ - first(theory.ell) + 1
    (i < 1 || i > length(theory.ell)) && return 0.0
    return getfield(theory, mode)[i]
end

"""
    theory_slice(theory, mode, ℓmin, ℓmax) -> Vector

Theory Dℓ on the contiguous absolute-multipole range `ℓmin:ℓmax`, zero outside
the stored grid. One allocation + one pass — use instead of per-`theory_at`
comprehensions inside AD-traced code.
"""
function theory_slice(theory::JointCMBTheory, mode::Symbol, ℓmin::Int, ℓmax::Int)
    lo = first(theory.ell)
    hi = last(theory.ell)
    out = zeros(Float64, ℓmax - ℓmin + 1)
    a = max(ℓmin, lo)
    b = min(ℓmax, hi)
    a <= b && (out[a-ℓmin+1:b-ℓmin+1] .= view(getfield(theory, mode), a-lo+1:b-lo+1))
    return out
end
