# Fixed data for the three experiments, loaded from the converted artifacts.

# ---------------------------------------------------------------------------
# Planck (HiLLiPoP v4.2 PR4, paper-era Hillik d0e455cf)
# ---------------------------------------------------------------------------

const PLK_MAPNAMES = ["100A", "100B", "143A", "143B", "217A", "217B"]
const PLK_FREQS = [100, 100, 143, 143, 217, 217]

"""
    PlanckData

Frozen Hillik Planck data: 4872-element vector (TT 1646 + EE 1466 + TE/ET 1760),
inverse covariance (float32), unbinned cross-spectra and weights, per-xfreq cuts.
"""
struct PlanckData
    data_vector::Vector{Float64}
    inv_cov::Matrix{Float32}
    inv_cov64::Matrix{Float64}   # cached float64 copy (fixed data)
    dldata::Dict{Symbol,Matrix{Float64}}   # :TT, :EE, :TE, :ET — (15, 2501)
    dlweight::Dict{Symbol,Matrix{Float64}}
    lmins::Dict{Symbol,Vector{Int}}        # per-xfreq (6,)
    lmaxs::Dict{Symbol,Vector{Int}}
    bin_lmins::Vector{Int}                 # lite binning
    bin_lmaxs::Vector{Int}
    xspec2xfreq::Vector{Int}
end

function load_planck_data(data_dir::AbstractString)
    data_vector = vec(npzread(joinpath(data_dir, "data_vector.npy")))
    inv_cov = npzread(joinpath(data_dir, "inv_cov.npy"))
    dldata = Dict{Symbol,Matrix{Float64}}()
    dlweight = Dict{Symbol,Matrix{Float64}}()
    for (sym, name) in ((:TT, "tt"), (:EE, "ee"), (:TE, "te"), (:ET, "et"))
        dldata[sym] = npzread(joinpath(data_dir, "dldata_$(name).npy"))
        dlweight[sym] = npzread(joinpath(data_dir, "dlweight_$(name).npy"))
    end
    lmins = Dict{Symbol,Vector{Int}}()
    lmaxs = Dict{Symbol,Vector{Int}}()
    for (sym, name) in ((:TT, "tt"), (:EE, "ee"), (:TE, "te"))
        lmins[sym] = Int.(vec(npzread(joinpath(data_dir, "lmins_$(name).npy"))))
        lmaxs[sym] = Int.(vec(npzread(joinpath(data_dir, "lmaxs_$(name).npy"))))
    end
    binning = npzread(joinpath(data_dir, "binning.npz"))
    x2f = Int.(vec(npzread(joinpath(data_dir, "xspec2xfreq.npy"))))
    return PlanckData(
        Vector{Float64}(data_vector),
        Matrix{Float32}(inv_cov),
        Matrix{Float64}(inv_cov),
        dldata, dlweight, lmins, lmaxs,
        Int.(vec(binning["lmins"])), Int.(vec(binning["lmaxs"])),
        x2f,
    )
end

# ---------------------------------------------------------------------------
# ACT (DR6, paper-era TTTEEE_PACT selection: 1139 elements)
# ---------------------------------------------------------------------------

"""
    ACTData

Frozen Hillik ACT DR6 data with the paper PACT selection: 1139-element vector,
inverse covariance, per-(spectrum, pol) windows and effective multipoles,
chromatic beams and bandpasses.
"""
struct ACTData
    data_vector::Vector{Float64}
    inv_cov::Matrix{Float64}
    leff::Vector{Float64}
    windows::Vector{Matrix{Float64}}      # per selected (spec, pol), (n_ell, n_bin)
    window_values::Vector{Vector{Int}}    # ell grid of each window
    spec_index::Vector{Int}               # original spectrum index (0-based) per entry
    spec_experiments::Vector{Tuple{String,String}}
    spec_pol::Vector{String}
    beams::Dict{String,Matrix{Float64}}    # map -> (n_ell, n_nu), normalized at ℓ=0
    bandpass_nu::Dict{String,Vector{Float64}}
    bandpass::Dict{String,Vector{Float64}}
    cross_pairs::Vector{Tuple{String,String}}  # 15 unique pairs, first-occurrence order
    map_names::Vector{String}                 # sorted beam map names
    cross_pair_i1::Vector{Int}                 # row in map_names per cross pair
    cross_pair_i2::Vector{Int}
    # per-spec precomputed instrument indices (avoids String indexing/Dict
    # lookups inside AD-traced code)
    spec_e1_idx::Vector{Int}                  # index of e1 in _ACT_MAPS
    spec_e2_idx::Vector{Int}
    spec_p1_E::Vector{Bool}                   # first polarization char == 'E'
    spec_p2_E::Vector{Bool}
    # trapezoid weights .* bandpass, precomputed per map (uniform band shifts
    # leave the weights unchanged, so the product is shift-independent)
    weighted_bandpass::Dict{String,Vector{Float64}}
end

function load_act_data(data_dir::AbstractString)
    data_vector = vec(npzread(joinpath(data_dir, "data_vector.npy")))
    inv_cov = npzread(joinpath(data_dir, "inv_cov.npy"))
    leff = vec(npzread(joinpath(data_dir, "leff.npy")))
    spec = JSON.parsefile(joinpath(data_dir, "spec_index.json"))
    windows = Matrix{Float64}[]
    window_values = Vector{Int}[]
    spec_index = Int[]
    spec_experiments = Tuple{String,String}[]
    spec_pol = String[]
    for s in spec["spectra"]
        tag = lpad(s["ispec"], 2, "0")
        pol = s["pol"]
        push!(windows, npzread(joinpath(data_dir, "windows", tag * "_" * pol * ".npy")))
        wvals = npzread(joinpath(data_dir, "windows", tag * "_" * pol * "_values.npy"))
        push!(window_values, Int.(vec(wvals)))
        push!(spec_index, Int(s["ispec"]))
        push!(spec_experiments, (s["experiments"][1], s["experiments"][2]))
        push!(spec_pol, pol)
    end
    beams = Dict{String,Matrix{Float64}}()
    bandpass_nu = Dict{String,Vector{Float64}}()
    bandpass = Dict{String,Vector{Float64}}()
    for f in readdir(joinpath(data_dir, "beams"))
        m = first(splitext(f))
        beams[m] = npzread(joinpath(data_dir, "beams", f))
    end
    bp = npzread(joinpath(data_dir, "bandpasses.npz"))
    for m in keys(beams)
        bandpass_nu[m] = vec(bp["$(m)_nu"])
        bandpass[m] = vec(bp["$(m)_bandpass"])
    end
    cross = Tuple{String,String}[]
    for (m1, m2) in spec_experiments
        p = m1 <= m2 ? (m1, m2) : (m2, m1)
        p in cross || push!(cross, p)
    end
    length(cross) == 15 || error("expected 15 unique ACT cross-frequency pairs")
    map_names = sort!(collect(keys(beams)))
    cross_pair_i1 = [findfirst(==(m1), map_names) for (m1, _) in cross]
    cross_pair_i2 = [findfirst(==(m2), map_names) for (_, m2) in cross]
    e1_idx = [findfirst(==(e1), _ACT_MAPS) for (e1, _) in spec_experiments]
    e2_idx = [findfirst(==(e2), _ACT_MAPS) for (_, e2) in spec_experiments]
    p1_E = [pol[1] == 'E' for pol in spec_pol]
    p2_E = [pol[2] == 'E' for pol in spec_pol]
    wbp = Dict{String,Vector{Float64}}()
    for m in keys(beams)
        nu = bandpass_nu[m]
        n = length(nu)
        w = Vector{Float64}(undef, n)
        w[1] = (nu[2] - nu[1]) / 2
        for j in 2:n-1
            w[j] = (nu[j+1] - nu[j-1]) / 2
        end
        w[n] = (nu[n] - nu[n-1]) / 2
        wbp[m] = w .* bandpass[m]
    end
    return ACTData(
        Vector{Float64}(data_vector), Matrix{Float64}(inv_cov), Vector{Float64}(leff),
        windows, window_values, spec_index, spec_experiments, spec_pol,
        beams, bandpass_nu, bandpass, cross, map_names, cross_pair_i1, cross_pair_i2,
        e1_idx, e2_idx, p1_E, p2_E, wbp,
    )
end

# ---------------------------------------------------------------------------
# SPT (SPT-3G D1, 1392 elements — reuses the SPTLikelihoods converted artifact)
# ---------------------------------------------------------------------------

const SPT_SPECTRUM_ORDER = [
    "TT 90x90", "TE 90x90", "EE 90x90",
    "TT 90x150", "TE 90x150", "TE 150x90", "EE 90x150",
    "TT 90x220", "TE 90x220", "TE 220x90", "EE 90x220",
    "TT 150x150", "TE 150x150", "EE 150x150",
    "TT 150x220", "TE 150x220", "TE 220x150", "EE 150x220",
    "TT 220x220", "TE 220x220", "EE 220x220",
]

"""
    SPTData

SPT-3G D1 fixed data from the SPTLikelihoods converted artifact: 1392-element
vector, covariance, 21 windows, eigenmode and main-temperature beams.
Ell grids start at ℓ=2 (artifact convention; absolute-ℓ indices must subtract 2).
"""
struct SPTData
    data_vector::Vector{Float64}
    cov::Matrix{Float64}
    inv_cov::Symmetric{Float64,Matrix{Float64}}   # precomputed; see _spt_inverse_covariance
    windows::Vector{Matrix{Float64}}   # (4094, n_bin) per spectrum, 1-based files
    ells::Vector{Int}                  # 2:4095
    eigenmodes::Array{Float64,3}       # (3 freqs, 4094 ells, 9 modes)
    main_temperature::Matrix{Float64}  # (3 freqs, 4094)
end

"""
    _spt_inverse_covariance(cov) -> Symmetric

`Σ⁻¹` for the SPT quadratic form, formed **once** at load.

`spt_chi2` used to evaluate `dot(delta, cov \\ delta)`, which refactorizes the
1392x1392 covariance on every call — and the reverse-mode rule did it a second
time, so one gradient paid for two factorizations. The released covariance is
not exactly symmetric (`max|C - Cᵀ| / max|C| = 1.5e-17`, pure round-off), so `\\`
did not even get a Cholesky: it ran a general LU. Precomputing this operator
takes the quadratic form from 37 ms to 0.18 ms.

`Symmetric` is the right reading of the data at that asymmetry, the matrix is
positive definite, and the inverse is formed from the Cholesky factor rather
than by general inversion. The frozen upstream code evaluates
`delta @ inv_bpcov @ delta`, so this is also the closer analogue of the
reference.

Against the previous LU route the quadratic form moves by 1.6e-11 relative,
which is Cholesky-versus-LU at `cond(C) = 9.3e8` and not an artifact of
inverting: a cached `cholesky` factorization moves by the same amount and agrees
with this operator to 2.1e-15. The frozen fixtures assert `rtol = 1e-10`.
"""
function _spt_inverse_covariance(cov::AbstractMatrix)
    symmetric = Symmetric(Matrix{Float64}(cov))
    isposdef(symmetric) || throw(ArgumentError(
        "the SPT covariance must be positive definite to be inverted",
    ))
    return Symmetric(inv(cholesky(symmetric)))
end

function load_spt_data(data_dir::AbstractString)
    data_vector = vec(npzread(joinpath(data_dir, "data_vector.npy")))
    cov = npzread(joinpath(data_dir, "covariance.npy"))
    ells = Int.(vec(npzread(joinpath(data_dir, "ells.npy"))))
    windows = [npzread(joinpath(data_dir, "windows", "$(lpad(i, 2, '0')).npy"))
               for i in eachindex(SPT_SPECTRUM_ORDER)]
    eigenmodes = npzread(joinpath(data_dir, "beams", "eigenmodes.npy"))
    main_temperature = npzread(joinpath(data_dir, "beams", "main_temperature.npy"))
    covariance = Matrix{Float64}(cov)
    return SPTData(
        Vector{Float64}(data_vector), covariance, _spt_inverse_covariance(covariance),
        windows, ells, eigenmodes, main_temperature,
    )
end

# ---------------------------------------------------------------------------
# Shared foreground templates
# ---------------------------------------------------------------------------

"""
    SharedTemplates

The four Hillik shared foreground templates (raw Dℓ, μK², ℓ=2..13500).
Normalization at lnorm happens in the foreground model, not here.
"""
struct SharedTemplates
    tsz::Vector{Float64}
    ksz::Vector{Float64}
    cib::Vector{Float64}
    szxcib::Vector{Float64}
    ell::Vector{Int}
end

function load_shared_templates(data_dir::AbstractString)
    function tpl(name)
        ell = Int.(vec(npzread(joinpath(data_dir, "$(name)_ell.npy"))))
        dl = vec(npzread(joinpath(data_dir, "$(name)_dl.npy")))
        return ell, dl
    end
    ell, tsz = tpl("tsz")
    _, ksz = tpl("ksz")
    _, cib = tpl("cib")
    _, szxcib = tpl("szxcib")
    return SharedTemplates(tsz, ksz, cib, szxcib, ell)
end

# ---------------------------------------------------------------------------
# Artifact-based zero-argument loaders. The explicit-directory forms remain
# the primary API; these resolve the bound Artifacts.toml entries (resolved
# lazily, at call time, so the package loads before artifacts are downloaded).
# ---------------------------------------------------------------------------

load_planck_data() = load_planck_data(artifact"joint_planck_data")
load_act_data() = load_act_data(artifact"joint_act_data")
load_spt_data() = load_spt_data(artifact"joint_spt_data")
load_shared_templates() = load_shared_templates(artifact"joint_fg_templates")
