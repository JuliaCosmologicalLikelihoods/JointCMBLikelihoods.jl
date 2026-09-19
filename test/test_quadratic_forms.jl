using NPZ

@testset "Quadratic forms vs frozen Hillik" begin
    pd = load_planck_data(PLANCK_DATA_DIR)
    ad = load_act_data(ACT_DATA_DIR)
    sd = load_spt_data(SPT_DATA_DIR)

    @testset "data vector sizes" begin
        @test length(pd.data_vector) == 4872
        @test length(ad.data_vector) == 1139
        @test length(sd.data_vector) == 1392
        @test size(pd.inv_cov) == (4872, 4872)
        @test size(ad.inv_cov) == (1139, 1139)
        @test size(sd.cov) == (1392, 1392)
    end

    # frozen reference chi2 from the exported fixtures (baseline point)
    @testset "baseline chi2" begin
        d_plk = npzread(joinpath(FIXTURE_DIR, "planck_baseline.npz"))
        d_act = npzread(joinpath(FIXTURE_DIR, "act_baseline.npz"))
        d_spt = npzread(joinpath(FIXTURE_DIR, "spt_baseline.npz"))

        chi2_plk = planck_chi2(pd, d_plk["delta_dl"])
        chi2_act = act_chi2(ad, d_act["delta_dl"])
        chi2_spt = spt_chi2(sd, d_spt["delta_dl"])

        @test chi2_plk ≈ Float64(d_plk["chi2"]) rtol = 1e-10
        @test chi2_act ≈ Float64(d_act["chi2"]) rtol = 1e-10
        @test chi2_spt ≈ Float64(d_spt["chi2"]) rtol = 1e-10

        c = chi2_components(pd, ad, sd,
            d_plk["delta_dl"], d_act["delta_dl"], d_spt["delta_dl"])
        @test c.planck + c.act + c.spt ≈ 9163.017639179096 rtol = 1e-9
    end

    # multipoint: every exported point must reproduce its frozen chi2
    @testset "multipoint chi2" begin
        tags = ["common_fg", "plk_cal", "plk_dust", "plk_ps", "act_cal", "act_band",
            "act_fg", "spt_cal", "spt_beam", "spt_leak", "spt_fg", "spt_kappa",
            "multi1", "multi2", "multi3", "multi4", "multi5",
            "boundary_low", "boundary_high"]
        for tag in tags
            d_plk = npzread(joinpath(FIXTURE_DIR, "planck_$(tag).npz"))
            d_act = npzread(joinpath(FIXTURE_DIR, "act_$(tag).npz"))
            d_spt = npzread(joinpath(FIXTURE_DIR, "spt_$(tag).npz"))
            @test planck_chi2(pd, d_plk["delta_dl"]) ≈ Float64(d_plk["chi2"]) rtol = 1e-9
            @test act_chi2(ad, d_act["delta_dl"]) ≈ Float64(d_act["chi2"]) rtol = 1e-9
            @test spt_chi2(sd, d_spt["delta_dl"]) ≈ Float64(d_spt["chi2"]) rtol = 1e-9
        end
    end
end
