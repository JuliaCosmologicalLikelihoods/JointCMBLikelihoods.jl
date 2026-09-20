using Test
using JointCMBLikelihoods
using Artifacts

const FIXTURE_DIR = get(ENV, "JOINT_CMB_FIXTURES", artifact"joint_cmb_fixtures")
# default to the bound artifacts; ENV overrides point at explicit directories
const PLANCK_DATA_DIR = get(ENV, "JOINT_PLANCK_DATA", artifact"joint_planck_data")
const ACT_DATA_DIR = get(ENV, "JOINT_ACT_DATA", artifact"joint_act_data")
const SPT_DATA_DIR = get(ENV, "JOINT_SPT_DATA", artifact"joint_spt_data")

include("test_ad.jl")   # first: build the joint Mooncake tapes while the
                       # process is lean; they are freed after this file
GC.gc()
include("test_quadratic_forms.jl")
include("test_planck_model.jl")
include("test_act_model.jl")
include("test_spt_model.jl")
include("test_foregrounds.jl")
include("test_instrument.jl")
include("test_priors.jl")
GC.gc()
include("test_turing.jl")
